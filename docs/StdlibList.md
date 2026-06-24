# 标准库 `List<T>` 设计

本文档记录当前标准库 `List<T>` 的目标形态和迁移结果。旧的 dialect-level
`mulberry.list` 路径已经删除；`List<T>` 现在走 stdlib/comptime
`List<T>` header struct 模型。

本文档建立在 `docs/PtrAndHeapObjectModel.md` 的统一模型上：`Ptr<T>` 按 C/C++
typed pointer 理解；需要共享的 runtime storage 用 Boehm GC 管理。

## 目标

`List<T>` 应该是标准库容器，而不是 Dialect API。用户应该理解它类似 `std::vector<T>`
或 Rust `Vec<T>`：

- 元素类型 `T` 可以是 builtin、struct、Tensor、其它未来支持的类型。
- `List<T>` 有运行时 `length` 和 `capacity`。
- `List<T>` 是 public list handle；可变共享语义通过浅拷贝 header + 共享 data buffer 表达。
- `push`、`get`、`set`、`size` 这类操作应该优先用 Mulberry 标准库源码表达，
  并在用户层暴露成 `xs.push(...)`、`xs.size()` 这种 method call。
- `mulberry.list.*` 不再是实现路径。

## 用户可见形态

源码里的 `List<T>` 拼写应该保持稳定：

```cherry
fn sum(xs: List<UInt64>): UInt64 {
  var i: UInt64 = 0;
  var result: UInt64 = 0;
  while i lt xs.size() {
    result = result + xs[i];
    i = i + 1;
  };
  result
}
```

长期目标不是让用户写很长的 `collections.List<T>`。标准库实现可以放在
`std.collections`，但 prelude 应该把 `collections.List` 直接引进来，使普通用户继续写
裸 `List<T>`。

## 目标源码形态

下面是目标方向，不是当前 Cherry 已经支持的语法。

```cherry
package std.collections;

comptime List<T> = struct {
  length: UInt64,
  capacity: UInt64,
  data: Ptr<T>
};
```

这里 `List<T>` 自己就是 public header struct，包含 `length/capacity/data`。`data` 指向 element buffer。复制 `List<T>` 是浅拷贝：header 会复制一份，但 `data` buffer 共享。

## M1：method-in-struct 目标 API

`List<T>` 后续应该优先暴露为 method API：

```cherry
var xs: List<UInt64> = [1, 2, 3];
xs.push(4);
io.print(xs.size());
io.print(xs[0]);
```

这不是 OOP。method declaration 只是在 struct 作用域里声明普通函数；
`obj.method(a, b)` 只是 receiver-first 的普通函数调用糖：

```text
xs.size()       -> std.collections.List.size(xs)
xs.push(value)  -> std.collections.List.push(xs, value)
xs[index]       -> std.collections.List.get(xs, index)
xs[index] = v   -> std.collections.List.set(xs, index, v)
```

实际内部 symbol 名字由 Sema 的 generic monomorphization 和 mangling 决定。用户源码不应
看见这些长名字。

第一版只支持最小语义：

- `self` 是普通第一个参数，不是隐藏对象。
- 不区分 const receiver / mutable receiver；后续需要 readonly 语义时再补。
- 不做继承、虚函数、接口、trait、动态派发或 method value。
- MLIRGen 不认识 method；Sema 在进入 MLIRGen 前已经把 dot call 解析成普通
  function call。

在当前浅拷贝 header 模型下，method 声明在 `List<T>` 内，receiver 显式写成 `Ptr<List<T>>`，这样 `push` 可以修改调用者的 header：

```cherry
package std.collections;

comptime List<T> = struct {
  length: UInt64,
  capacity: UInt64,
  data: Ptr<T>,

  pub fn size(self: Ptr<List<T>>): UInt64 {
    self.length
  }

  pub fn get(self: Ptr<List<T>>, index: UInt64): T {
    self.data[index]
  }

  pub fn push(self: Ptr<List<T>>, value: T): UInt64 {
    ...
  }
};
```

这样 `var xs: List<UInt64>` 的运行时语义是一个 by-value header struct。`xs.size()`
和 `xs.push(value)` 会把 `xs` 的地址作为 receiver 传给 method，所以 `push` 能修改
`xs.length/capacity/data`。如果把 `xs` 赋值给另一个变量，header 会浅拷贝，两个
header 仍共享同一个 `data` buffer。

## 当前 API

第一批标准库 API 应保持很小：

```cherry
package std.collections;

fn withCapacity<T>(capacity: UInt64): List<T>;

comptime List<T> = struct {
  pub fn size(self: Ptr<List<T>>): UInt64;
  pub fn get(self: Ptr<List<T>>, index: UInt64): T;
  pub fn set(self: Ptr<List<T>>, index: UInt64, value: T): UInt64;
  pub fn push(self: Ptr<List<T>>, value: T): UInt64;
};
```

`withCapacity` 是没有 receiver 的 factory function。空 list 直接使用 `[]`：

```cherry
var xs: List<UInt64> = [];
```

`size/push`
已经放进 `List<T>`，用户层写法是 `xs.size()`、`xs.push(value)`。
读取和写入元素使用下标语法：`xs[index]`、`xs[index] = value`。

语法层面的 `xs[i]` 已经在 Sema/MLIRGen 里走 `List.get`：

```cherry
xs[i]
```

当前 generic function inference 已经能从普通参数和 expected return type 推断 `T`，
用户不应该经常手写类型参数：

```cherry
xs.push(42)
```

`collections.withCapacity(capacity)` 依赖 expected type 推断 `T`，适合需要预分配
capacity 的场景。普通空 list 不需要额外 factory。

## 需要补齐的语言能力

迁移 `List<T>` 之前至少需要以下能力：

- `Ptr<T>`：一等类型，当前支持 `*p` 读取、`*p = value` 写入，以及 `p[i]`
  连续元素索引；null 或等价的空指针表达后续再补。
- generic struct：允许标准库根据 `T` 生成 `List<T>` concrete type。
- generic function：允许 `new<T>`、`push<T>`、`get<T>` 这类函数按 `T` 单态化。
- heap allocation primitive：基于当前 Boehm GC 路线，已经有单对象
  `heap.alloc<T>()` 和连续元素 storage `heap.alloc<T>(count)`。
- `sizeof(T)` / `alignof(T)`：用于计算 element storage 大小和对齐。
- package re-export：prelude 能把标准库里的 `collections.List<T>` 作为裸 `List<T>` 暴露。

## Heap primitive

当前已经接入 Boehm GC，因此第一阶段不需要 `free`。已经实现的最小 primitive 是：

```cherry
package std.heap;

fn alloc<T>(): Ptr<T>;
```

它用于分配一个 `T` 对象。连续元素 storage 使用同名带 count 的形式：

```cherry
package std.heap;

fn alloc<T>(count: UInt64): Ptr<T>;
```

语义：

- `count` 是元素个数，不是 byte 数。
- 编译器或 runtime 使用 `sizeof(T)` / `alignof(T)` 计算实际分配大小。
- 返回值指向连续的 `T` storage。
- 分配失败策略后续再设计；当前可以 fail-fast 或交给 runtime abort。

如果以后需要 resize，可以先用 allocate-copy 的方式实现：

```cherry
fn grow<T>(oldData: Ptr<T>, oldLength: UInt64, newCapacity: UInt64): Ptr<T>
```

暂时不要设计复杂 allocator trait。Mulberry 当前需要的是可用、清楚、可替换的最小
heap primitive。

## 单态化

`List<T>` 标准库实现应采用 Zig/C++ template 风格的单态化：

```text
List<UInt64>      -> concrete List storage/function set for UInt64
List<Tensor<Float32>>
                  -> concrete List storage/function set for that Tensor header
```

用户源码不应该出现很长的单态化名字。长名字只允许存在于编译器内部 symbol mangling
或 MLIR symbol 中，源码层必须保持短而清楚。

## 和旧 `mulberry.list` 的关系

旧路径曾经是：

```text
source List<T>
  -> compiler-known ListType
  -> mulberry.list / mulberry.list_storage / mulberry.list_desc
  -> lowering
```

当前路径已经变成：

```text
source List<T>
  -> prelude import collections.List
  -> generic struct/function monomorphization
  -> heap object pointer: Ptr<record { length, capacity, data }>
  -> lowering
```

已经删除：

- `mulberry.list`
- `mulberry.list_storage`
- `mulberry.list_desc`
- `ListCreateOpLowering` / `ListGetOpLowering` / `ListDesc*` lowering
- boundary rewrite 里针对 list descriptor 的特殊逻辑

后续不要恢复这条 descriptor bridge 路径。缺少能力时应补语言/stdlib 能力，而不是把
list-specific workaround 塞回 MLIRGen 或 lowering。

## 迁移顺序

建议后续按下面顺序推进：

```text
C4.1  本文档：固定 Ptr / heap storage 方向
C4.2  实现 source-level Ptr<T>
C4.3  实现最小 `*p` / `*p = value` 解引用读写能力
C4.4  实现 Boehm-backed heap.alloc<T>()
C4.5  实现 `heap.alloc<T>(count)` 和 `p[i]` 指针索引
C4.6  支持 generic struct，用 Mulberry 表达 List<T>
C4.7  支持 generic function，用 Mulberry 表达 List<T> API
C4.8  把 List<T> 迁到 std.collections
C4.9  删除 mulberry.list 相关 Dialect/op/lowering
```

当前 C4.6 已经支持 generic struct alias，并能生成高层 MLIR。

当前 C4.7 已经支持最小 generic function 单态化。函数可以写成：

```cherry
fn push<T>(xs: collections.List<T>, value: T): UInt64 {
  ...
}
```

调用时 Sema 从 `xs` 或 `value` 的 concrete type 推断 `T`，生成内部 concrete
函数，例如 `push__UInt64`。这一步仍然只发生在 Parser/Sema 层；MLIRGen 只处理普通
函数和普通 `func.call`。

当前 `std.collections` 已经实现最小 `List<T>` layout，以及 public
`List<T>` alias，和 `new<T>()`、`withCapacity<T>()`、`size<T>()`、`get<T>()`、
`set<T>()`、`push<T>()`。其中 `withCapacity<T>()` 会分配 list header 和 element
buffer；`push<T>()` 在容量不足时会按 `0 -> 1 -> 2x` 策略分配新 buffer 并复制旧元素。

当前 C4.9 已经完成：旧 `mulberry.list` / `mulberry.list_storage` /
`mulberry.list_desc` type、op、lowering 和直接测试这些 IR 的 lit 已删除。

如果某一步需要靠 lowering bridge 才能“看起来能跑”，先停下来补语言/IR 能力，不要
把 workaround 塞回 MLIRGen。
