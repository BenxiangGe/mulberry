# Mulberry 的 Comptime 泛型设计

## 目标

Mulberry 需要一种简单的泛型机制，把 `List<T>`、`Tensor<T>`、后续的
`Ptr<T>` 这类语言基础设施逐步迁到标准库源码中，而不是继续扩大 dialect 的职责。

第一阶段只做类型级 comptime：

```cherry
fn main(): UInt64 {
  const xs: types.Vector<UInt64> = [1, 2, 3];
  size(xs)
}
```

这里的 `types.Vector<UInt64>` 会在 Sema 阶段实例化成 `List<UInt64>`。MLIRGen
只看见已经实例化后的 concrete type，不需要知道 generic。

当前还支持 generic struct alias：

```cherry
comptime List<T> = struct {
  length: UInt64,
  capacity: UInt64,
  data: Ptr<T>
};
```

`List<UInt64>` 会在 Sema 阶段实例化成一个 concrete struct type。MLIRGen
仍然只看到普通 `mulberry.record`，不会把 generic 语义带进 IR。

当前还支持最小 generic function：

```cherry
fn first<T>(ptr: Ptr<T>): T {
  ptr[0]
}
```

调用处不需要显式写类型参数：

```cherry
var data: Ptr<UInt64> = heap.alloc<UInt64>(1);
first(data)
```

Sema 会从实参类型 `Ptr<UInt64>` 推断 `T = UInt64`，再生成一个 concrete function，
内部名字类似 `first__UInt64`。MLIRGen 只看到 `first__UInt64` 这种普通函数和普通
`func.call`，不会处理 generic。

如果调用表达式出现在带有 expected type 的位置，Sema 也可以从返回类型反推 `T`：

```cherry
fn make<T>(): Ptr<T> {
  heap.alloc<T>()
}

var value: Ptr<UInt64> = make();
```

这里变量声明的 expected type 是 `Ptr<UInt64>`，所以 `make()` 会被实例化为
`make__UInt64`。这让 `new<T>()` 这类无参数 factory 可以保持源码简洁，不需要引入
`make<UInt64>()` 这种显式 generic call 语法。

## 标准库入口

`stdlib/std/types.cherry` 里先放最小的类型级 alias：

```cherry
package std.types;

comptime Vector<T> = List<T>;
comptime Matrix<T> = T[?, ?];
```

`prelude.cherry` 默认 import `std.types`。用户代码使用 `types.Vector<T>`、
`types.Matrix<T>`，而不是裸 `Vector<T>`。这样可以保留最后一级包名，既不会太长，
也不会把标准库名字直接塞进全局命名空间。

## 类型布局查询

`sizeof(T)` 和 `alignof(T)` 是类型级查询，但结果是普通 `UInt64` 常量：

```cherry
const itemSize: UInt64 = sizeof(UInt64);
const itemAlign: UInt64 = alignof(UInt64);
```

第一阶段只支持布局已经明确的类型：

- `Bool`、`UInt8`、`UInt64`、`Float32`
- 只包含上述类型或其它已支持 struct 的 `struct`

`String`、`File`、`Tensor`、`List<T>` 暂时不支持 `sizeof/alignof`。这些类型的真实
运行时表示涉及 descriptor、指针或 lowering ABI，现在强行给出大小会误导后续设计。
后面把 `List<T>` 迁到标准库时，会先补齐需要的 pointer/generic struct 能力，再扩大
布局查询支持范围。

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

```cherry
comptime Name<T> = Type;
```

使用处：

```cherry
Name<ConcreteType>
```

会在 Sema 阶段展开为：

```text
substitute(T := ConcreteType, Type)
```

例如：

```cherry
comptime Matrix<T> = T[?, ?];
Matrix<Float32>
```

实例化结果为：

```cherry
Float32[?, ?]
```

## 编译器分层

当前分层保持简单：

- Parser 只构建 `ComptimeTypeAliasDecl` 和 `GenericTypeNode`。
- Sema 负责解析 alias、维护类型参数作用域、实例化 concrete type/function。
- MLIRGen 不处理 generic，只处理 Sema 后的 concrete type。
- Mulberry dialect 不承载 generic 语义。

这点很重要：generic 是语言类型系统能力，不是运行时 IR 语义。

## 和 `mulberry.list` 的关系

短期内，现有 `List<T>` 仍然是 compiler-known 类型，并继续 lower 到现有
`mulberry.list` 路径。`comptime` 先让用户代码可以定义别名和类型构造器，例如
`types.Vector<T>`、`types.Matrix<T>`。

长期方向是补齐以下能力后，把 `List<T>` 的实现迁到标准库：

- `Ptr<T>`
- `sizeof(T)` / `alignof(T)`
- heap allocation primitive
- generic struct
- generic function

当前 generic struct alias 已经可以生成高层 MLIR/`mulberry.record`。Lowering/JIT
是否能跑通，仍取决于 concrete struct 里包含的字段类型以及函数边界是否已经有对应的
LowerMulberry 支持；不要为了让某个 generic struct 例子 JIT 而把 lowering workaround
塞回 MLIRGen。

当前 generic function 是最小单态化：

- 只支持单个类型参数：`fn name<T>(...)`。
- 类型参数从普通调用实参中推断；在有 expected type 的位置，也可以从返回类型反推。
- 不支持 `name<UInt64>(...)` 显式调用语法。
- 实例化结果会追加成普通 concrete function，generic template 本身不会进入 MLIR。
- 支持从 `Ptr<T>`、`List<T>`、Tensor element、以及 generic struct alias body 中推断
  `T`。

统一 `Ptr<T>` / heap object handle 模型见 `docs/PtrAndHeapObjectModel.md`；
标准库 `List<T>` 的目标源码形态见 `docs/StdlibList.md`。

到那时，`mulberry.list` 可以逐步退化为 lowering helper，或者完全消失。

## 后续阶段

- `C1`：支持 `comptime Name<T> = Type;`
- `C2`：给实例化结果和错误诊断补稳定格式
- `C3`：设计 `sizeof(T)` / `alignof(T)`，先用于文档和后续标准库 `List<T>`
- `C4`：设计标准库 `List<T>` 的源码形态
- `C5`：保持现有 lowering/JIT 路径稳定，不再向 dialect 加新的泛型语义
- `C6`：增加 demo，展示 `Vector<T>` / `Matrix<T>` 的实际用法
