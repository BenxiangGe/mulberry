# 标准库 `List<T>` 设计

本文档记录当前标准库 `List<T>` 的目标形态和迁移结果。旧的 dialect-level
`mulberry.list` 路径已经删除；`List<T>` 现在走 stdlib/comptime object 模型。

本文档建立在统一 object model 上：目标 source 语义是 `Scalar = value`、
`Object = reference`。底层 pointer/storage 细节只属于 compiler/stdlib/runtime
内部实现，不属于用户 source surface。

## 目标

`List<T>` 应该是标准库容器，而不是 Dialect API。用户应该理解它类似 `std::vector<T>`
或 Rust `Vec<T>`：

- 元素类型 `T` 可以是 builtin、struct、Tensor、其它未来支持的类型。
- `List<T>` 有运行时 `length` 和 `capacity`。
- `List<T>` 是 growable dynamic container，不是普通 `[]` literal 的默认目标。
  普通 `[]` literal 后续默认是语言 `Array`；如果需要把 Array 变成 List，应使用显式
  helper，例如 `list.from(array)`。
- `List<T>` 是 public list object。目标语义下，赋值、传参和返回复制 reference；
  可变共享由 `const` / `mut` 这类修饰符约束。
- `push`、`get`、`set`、`size` 这类操作应该优先用 Mulberry 标准库源码表达，
  并在用户层暴露成 `xs.push(...)`、`xs.size()` 这种 method call。
- `mulberry.list.*` 不再是实现路径。

## 用户可见形态

源码里的 `List<T>` 拼写应该保持稳定：

```mulberry
fn sum(xs: List<UInt64>): UInt64 {
  var i: UInt64 = 0;
  var result: UInt64 = 0;
  while i < xs.size() {
    result = result + xs[i];
    i = i + 1;
  }
  result
}
```

长期目标不是让用户写很长的 package-qualified type。标准库实现放在
`std.list`，prelude 把 `list.List` 直接引进来，使普通用户继续写
裸 `List<T>`。

## Reference semantics

目标 source 语义下，`List<T>` 变量保存 object reference。复制 `List<T>` 只复制
reference，不复制元素内容：

```mulberry
var xs: List<UInt64> = list.from([1, 2, 3]);
var ys: List<UInt64> = xs;
ys[0] = 9;
io.print(xs[0]); // 9
```

如果需要真正复制元素，应通过后续的 `clone()` / `copy()` 这类显式 API 表达。

## M1：method-in-struct 目标 API

`List<T>` 后续应该优先暴露为 method API：

```mulberry
var xs: List<UInt64> = list.from([1, 2, 3]);
xs.push(4);
io.print(xs.size());
io.print(xs[0]);
```

这不是 OOP。method declaration 只是在 struct 作用域里声明普通函数；
`obj.method(a, b)` 只是 receiver-first 的普通函数调用糖：

```text
xs.size()       -> std.list.List.size(xs)
xs.push(value)  -> std.list.List.push(xs, value)
xs[index]       -> std.list.List.get(xs, index)
xs[index] = v   -> std.list.List.set(xs, index, v)
```

实际内部 symbol 名字由 Sema 的 generic monomorphization 和 mangling 决定。用户源码不应
看见这些长名字。

第一版 method 语义：

- `self` 是普通第一个参数，不是隐藏对象。
- receiver 默认 mutable；只读方法写 `const self: List<T>`。
- 不做继承、虚函数、接口、trait、动态派发或 method value。
- MLIRGen 不认识 method；Sema 在进入 MLIRGen 前已经把 dot call 解析成普通
  function call。

当前实现已经走 source object reference。method 声明在 `List<T>` 内，source receiver
写成 `self: List<T>`；Sema/MLIRGen 内部使用 object reference ABI 操作同一个 list
object：

```mulberry
package std.list;

comptime List<T> = struct {
  pub fn size(const self: List<T>): UInt64 {
    ...
  }

  pub fn get(const self: List<T>, index: UInt64): T {
    ...
  }

  pub fn push(self: List<T>, const value: T): UInt64 {
    ...
  }
};
```

目标模型里 `var xs: List<UInt64>` 是 object reference；`xs.size()` 和
`xs.push(value)` 直接以 receiver reference 操作同一个 List object。如果把 `xs`
赋值给另一个变量，复制的是 reference，两个变量指向同一个 List object。

## 当前 API

第一批标准库 API 应保持很小：

```mulberry
package std.list;

fn withCapacity<T>(capacity: UInt64): List<T>;

comptime List<T> = struct {
  pub fn size(const self: List<T>): UInt64;
  pub fn get(const self: List<T>, index: UInt64): T;
  pub fn set(self: List<T>, index: UInt64, const value: T): UInt64;
  pub fn push(self: List<T>, const value: T): UInt64;
};
```

`withCapacity` 是没有 receiver 的 factory function。普通 `[]` 现在默认是 `Array`，
所以 `List<T>` 必须通过显式 factory 创建：

```mulberry
var empty: List<UInt64> = list.withCapacity(0);
var xs: List<UInt64> = list.from([1, 2, 3]);
```

`size/push`
已经放进 `List<T>`，用户层写法是 `xs.size()`、`xs.push(value)`。
读取和写入元素使用下标语法：`xs[index]`、`xs[index] = value`。

语法层面的 `xs[i]` 已经在 Sema/MLIRGen 里走 `List.get`：

```mulberry
xs[i]
```

当前 generic function inference 已经能从普通参数和 expected return type 推断 `T`，
用户不应该经常手写类型参数：

```mulberry
xs.push(42)
```

`list.withCapacity(capacity)` 依赖 expected type 推断 `T`，适合需要预分配
capacity 的场景。普通空 list 不需要额外 factory。

## 实现边界

`List<T>` 依赖 generic struct、generic function、GC-managed object 和 package
re-export。普通用户只需要使用 `List<T>`、`list.from(...)`、`list.withCapacity(...)`
以及 method API；具体实现细节只记录在 internal object/lowering 文档里。

## 单态化

`List<T>` 标准库实现应采用 Zig/C++ template 风格的单态化：

```text
List<UInt64>             -> concrete List type/functions for UInt64
List<Tensor<Float32>>    -> concrete List type/functions for Tensor<Float32>
```

用户源码不应该出现很长的单态化名字。长名字只允许存在于编译器内部 symbol mangling
或 MLIR symbol 中，源码层必须保持短而清楚。

## 和旧 `mulberry.list` 的关系

旧路径曾经把 `List<T>` 做成一套 compiler-known dialect type/op。当前路径改成
stdlib/comptime object 和普通 method API，源码层只看到 `List<T>`。

旧 `mulberry.list` / `mulberry.list_storage` / `mulberry.list_desc` 路径已经删除。后续
不要恢复这条 list descriptor 边界路径。缺少能力时应补语言/stdlib 能力，而不是把
list-specific workaround 塞回 compiler implementation。

## 当前迁移状态

generic struct alias 和 generic function 单态化已经支持，并能生成高层 MLIR。函数可以写成：

```mulberry
fn push<T>(xs: List<T>, value: T): UInt64 {
  ...
}
```

调用时 Sema 从 `xs` 或 `value` 的 concrete type 推断 `T`，生成内部 concrete
函数，例如 `push__UInt64`。这一步仍然只发生在 Parser/Sema 层；MLIRGen 只处理普通
函数和普通 `func.call`。

当前 `std.list` 已经实现 public `List<T>` alias、`withCapacity<T>()`、
`from<T, N>()`、`size<T>()`、`get<T>()`、`set<T>()`、`swap<T>()`、`grow<T>()`、
`push<T>()`。其中 `push<T>()` 在容量不足时会按 `0 -> 1 -> 2x` 策略增长容量并复制
旧元素。

当前 C4.9 已经完成：旧 `mulberry.list` / `mulberry.list_storage` /
`mulberry.list_desc` type/op 和直接测试这些 IR 的 lit 已删除。

如果某一步需要靠 implementation-only 捷径才能“看起来能跑”，先停下来补语言/IR 能力，
不要把 workaround 塞回 MLIRGen。
