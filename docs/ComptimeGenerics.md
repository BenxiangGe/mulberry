# Mulberry 的 Comptime 泛型设计

## 目标

Mulberry 需要一种简单的泛型机制，把 `List<T>`、`Tensor<T>` 这类语言基础设施逐步
迁到标准库源码中，而不是继续扩大 dialect 的职责。底层 storage 能力仍然存在，但它
是 compiler/stdlib/runtime internal detail，不是普通泛型代码的 source surface。

第一阶段只做类型级 comptime：

```mulberry
fn main(): UInt64 {
  const xs: List<UInt64> = list.from([1, 2, 3]);
  return xs.size();
}
```

这里的 `List<UInt64>` 会在 Sema 阶段实例化成 concrete `std.list.List<UInt64>`。
MLIRGen 只看见已经实例化后的 concrete type，不需要知道 generic。

当前还支持 generic struct alias：

```mulberry
comptime Box<T> = struct {
  value: T
};
```

`Box<UInt64>` 会在 Sema 阶段实例化成一个 concrete struct type。后续编译阶段只看到
已经实例化后的普通 concrete type，不会把 generic 语义带进 IR。

当前还支持最小 generic function：

```mulberry
fn identity<T>(value: T): T {
  return value;
}
```

调用处不需要显式写类型参数：

```mulberry
var value: UInt64 = identity(42);
```

Sema 会从实参类型 `UInt64` 推断 `T = UInt64`，再生成一个 concrete function，内部
名字类似 `identity__UInt64`。MLIRGen 只看到 `identity__UInt64` 这种普通函数和普通
`func.call`，不会处理 generic。

如果调用表达式出现在带有 expected type 的位置，Sema 也可以从返回类型反推 `T`：

```mulberry
fn empty<T>(): List<T> {
  return list.withCapacity(0);
}

var values: List<UInt64> = empty();
```

这里变量声明的 expected type 是 `List<UInt64>`，所以 `empty()` 会被实例化为
`empty__UInt64`。这让无参数 factory 可以保持源码简洁，不需要引入
`empty<UInt64>()` 这种显式 generic call 语法。

## M1：method-in-struct

Mulberry 后续只支持 Rust/Zig 风格的 method-in-struct，不做 OOP。method 本质上仍是
普通函数，只是声明位置放在 struct 里，并且 dot call 会把 receiver 放到第一个实参：

```mulberry
comptime Box<T> = struct {
  value: T,

  pub fn get(self: Box<T>): T {
    return self.value;
  }
};
```

使用处：

```mulberry
var box: Box<UInt64> = Box<UInt64> { 42 };
box.get();
```

Sema 处理为普通 generic function instantiation：

```text
box.get()
  -> std.<package>.Box.get__UInt64(box)
```

这个设计保持现有分层：

- Parser 只需要能把 struct 里的 `fn` 解析成 method declaration。
- Sema 根据 receiver type 做 method lookup，并把 dot call 改写成普通 call。
- Source method receiver 写 `self: T`；默认 readonly。需要修改 object 的 method
  写 `mut self: T`。
- Generic method 和普通 generic function 一样单态化。
- MLIRGen 只看到 concrete function 和普通 `func.call`。

第一版的限制要刻意保持小：

- `self` 必须显式写在参数列表里，且只是普通第一个参数。
- 参数和 receiver 默认 readonly；修改 object 的 method 必须显式写 `mut self: T`。
- method 名字只在所属 struct 的 method scope 内查找；不同 struct 可以有同名
  method。
- 不支持继承、虚表、接口、trait bound、动态派发、extension method、method value。
- factory 这类没有 receiver 的函数暂时保持自由函数，例如
  `list.withCapacity(...)`。

## 标准库入口

`stdlib/std/types.mulberry` 里先放最小的类型级 alias：

```mulberry
package std.types;

import list.List;

comptime Vector<T> = List<T>;
```

`prelude.mulberry` 默认 import `list.List` 和 `std.types`。用户代码直接使用裸
`List<T>`；需要类型级别别名时再使用 `types.Vector<T>`。这样可以
保留最后一级包名，既不会太长，也不会把标准库名字直接塞进全局命名空间。

## 类型布局查询

`sizeof(T)` 和 `alignof(T)` 是类型级查询，但结果是普通 `UInt64` 常量：

```mulberry
const itemSize: UInt64 = sizeof(UInt64);
const itemAlign: UInt64 = alignof(UInt64);
```

第一阶段只支持布局已经明确的类型：

- `Bool`、`UInt8`、`UInt64`、`Float32`
- 只包含上述类型或其它已支持 struct 的 `struct`

`String`、`File`、`Tensor`、`List<T>` 暂时不支持 `sizeof/alignof`。这些类型是
object reference，当前强行给出大小会误导后续设计。后面如果要支持它们，应先明确
object layout 语义，再扩大布局查询支持范围。

## 非目标

第一阶段不实现完整 Zig comptime：

- 不在编译期执行任意函数体。
- 不支持 comptime `if` / `for`。
- 不支持 generic runtime function。
- 不新增新的 MLIR dialect。

这样做是为了避免把类型系统、解释器和 lowering 过早混在一起。当前要解决的是标准库
类型能不能用 Mulberry 自己描述，而不是一次做完整元编程系统。

## 语义模型

`comptime` 声明是一个类型级 alias function：

```mulberry
comptime Name<T> = Type;
```

使用处：

```mulberry
Name<ConcreteType>
```

会在 Sema 阶段展开为：

```text
substitute(T := ConcreteType, Type)
```

例如：

```mulberry
comptime Vector<T> = List<T>;
Vector<UInt64>
```

实例化结果为：

```mulberry
List<UInt64>
```

## 编译器分层

当前分层保持简单：

- Parser 只构建 `ComptimeTypeAliasDecl` 和 `GenericTypeNode`。
- Sema 负责解析 alias、维护类型参数作用域、实例化 concrete type/function。
- MLIRGen 不处理 generic，只处理 Sema 后的 concrete type。
- Mulberry core dialect 不承载 generic 语义。

这点很重要：generic 是语言类型系统能力，不是运行时 IR 语义。

## 和旧 `mulberry.list` 的关系

`List<T>` 的实现已经迁到标准库/comptime 路径。旧 `mulberry.list` 路径已经删除。
这依赖以下已经打开的基础能力：

- generic struct
- generic function

当前 generic struct alias 已经可以生成 ordinary concrete type。后续能否完整执行，
取决于 concrete type 中包含的字段和函数边界是否已经被当前 compiler pipeline 支持；
不要为了让某个 generic struct 例子“看起来能跑”而把 workaround 塞回 MLIRGen。

当前 generic function 是最小单态化：

- 只支持单个类型参数：`fn name<T>(...)`。
- 类型参数从普通调用实参中推断；在有 expected type 的位置，也可以从返回类型反推。
- 不支持 `name<UInt64>(...)` 显式调用语法。
- 实例化结果会追加成普通 concrete function，generic template 本身不会进入 MLIR。
- 支持从 `List<T>`、Tensor element、以及 generic struct alias body 中推断 `T`。

统一 object reference / heap storage 模型见 `docs/StdlibObjects.md`；
标准库 `List<T>` 的 public API 和 layout 边界见 `docs/StdlibList.md`。

后续不要再把 generic/list 语义塞回 dialect。缺少容器能力时，应继续补
stdlib/comptime/source type system。

## Tensor 所需的 comptime 能力已打开

`List<T>` 只需要一个类型参数；Tensor header 也只需要元素类型。它们的 storage
layout 是 stdlib/internal 细节，用户层只依赖 public type 和 method API。

这里 `T` 是类型参数。`rank`、`sizes` 和 `strides` 都是 runtime metadata。普通元素
访问直接使用 `tensor[i]` / `tensor[i, j]`。source-level ranked tensor bridge 已删除；
后续 memref/linalg interop 应在 package/lowering 内部直接消费 `Tensor<T>` object。

C7 已经打开 type alias 多参数和 `UInt64` comptime integer 参数：

```text
C7.1  引入 ComptimeParam / ComptimeArg AST。
C7.2  Parser 支持 alias 参数列表和 generic type 实参列表。
C7.3  Sema 支持多类型参数 substitution。
C7.4  Sema 支持 UInt64 comptime integer 参数和整数实参 mangle/cache key。
C7.5  增加多参数 comptime alias 正向 lit。
C7.6  把 Tensor 落到 stdlib source。
```

## C7 设计：多参数 comptime generic

C7 的目标不是完整 Zig comptime，而是补齐标准库对象需要的最小类型级能力。核心需求
只有两个：

- 一个 alias 可以有多个 comptime 参数。
- 参数可以是类型，也可以是编译期整数。

当时验证用的多参数目标语法：

```mulberry
comptime StaticShape<Rank: UInt64> = struct {
  rank: UInt64
};
```

使用处：

```mulberry
StaticShape<2>
Tensor<Float32>
```

这里保留 `List<T>` 现有写法：没有类型标注的参数默认是类型参数。带 `: UInt64` 的参数
是编译期整数参数。这样旧代码不需要改，同时 `Rank` 的语义也足够明确。

### AST 形态

当前 AST 是单参数模型：

```text
ComptimeTypeAliasDecl(name, parameterName, bodyTypeNode)
GenericTypeNode(name, argumentTypeNode)
```

C7 后应改为：

```text
ComptimeParam {
  name
  kind: Type | UInt64
}

ComptimeArg {
  kind: Type | UInt64
  typeNode?
  uint64Value?
}

ComptimeTypeAliasDecl(name, params[], bodyTypeNode)
GenericTypeNode(name, args[])
```

这个改动要保持兼容：`params.size() == 1 && params[0].kind == Type` 就是今天的
`comptime List<T>`。

### Parser 改造

Parser 需要把 `<...>` 从“只解析一个 type”改成逗号分隔列表：

```text
< T >
< T, U >
< T, Rank: UInt64 >
< Float32, 2 >
```

参数声明里只允许 identifier 或 `identifier: UInt64`。先不要支持任意类型标注，
否则会过早引入 type-level value kind 系统。

实参列表里第一阶段只允许：

- 普通 type，例如 `Float32`、`List<Float32>`。
- 十进制整数 literal，例如 `2`。

不要在 C7.1 支持表达式实参，例如 `Rank + 1`。这会把 compile-time expression evaluator
提前拉进来，不符合当前目标。

### Sema 改造

Sema 需要把现在的单参数 substitution：

```text
substitute(T := ConcreteType, body)
```

改成 map-based substitution：

```text
substitute({
  T := Type(Float32),
  Rank := UInt64(2)
}, body)
```

类型参数可以替换 `NamedTypeNode`。整数参数第一阶段只需要参与 alias 实例化 identity 和
mangle；如果后续 type grammar 允许在 tensor shape 或 fixed array length 中引用
`Rank`，再把整数 substitution 扩展到这些节点。

generic struct cache key 也必须包含所有实参：

```text
ShapedBuffer__Float32__2
std.tensor.Tensor__Float32
```

不能只按 layout cache。两个不同 alias origin 即使字段 layout 一样，也可能代表不同
source-level 类型，类型 identity 必须不同。

### Generic function 暂不扩展

当前 generic function inference 是单类型参数模型：

```mulberry
fn push<T>(list: List<T>, value: T): UInt64
```

C7 第一阶段不要同时扩展 generic function 多参数推断。原因是函数推断需要从实参和
expected return type 反推出多个参数，复杂度明显高于 type alias 实例化。当前 stdlib
Tensor 不依赖 generic function 多参数推断，不需要立刻支持：

```mulberry
fn foo<T, Rank: UInt64>(tensor: ShapedBuffer<T, Rank>): UInt64
```

这类函数可以作为 C8 单独设计。C7 只保证 `Tensor<Float32>` 这种类型能在 Sema 后变成
具体 record/ptr 类型。

### C7 实施顺序

```text
C7.1  引入 ComptimeParam / ComptimeArg AST，保持单参数旧测试通过。
C7.2  Parser 支持 alias 参数列表和 generic type 实参列表。
C7.3  Sema 把单参数 substitution 改为 map-based substitution，先支持多类型参数。
C7.4  Sema 支持 UInt64 comptime integer 参数，mangle/cache key 包含 integer 实参。
C7.5  增加正向 lit：Pair<T, U>、ShapedBuffer<T, Rank>。
C7.6  只在上述通过后，把 `stdlib/std/tensor.mulberry` 加进来。
```

这个顺序的重点是保持每一步都能解释清楚。不要为了马上得到 Tensor object 而把 rank
语义塞进 mangled name 或 lowering 特判。

## 后续阶段

- `C1` 已完成：支持 `comptime Name<T> = Type;`
- `C2` 已完成：实例化结果和错误诊断已有基本稳定格式。
- `C3` 已完成：`sizeof(T)` / `alignof(T)` 已支持当前明确 layout 的类型。
- `C4` 已完成：标准库 `List<T>` 已迁到 `std.list`。
- `C5` 持续执行：保持现有 lowering/JIT 路径稳定，不再向 dialect 加新的泛型语义。
- `C6` 已完成：已有 `Vector<T>` 和 stdlib List 相关测试。
- `C7` 已完成：支持多参数 / integer 参数 comptime generic，`std.tensor.Tensor` 已可
  作为 source-level Tensor header alias 使用。
