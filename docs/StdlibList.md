# 标准库 `List<T>` 设计

本文档记录把当前 compiler-known `List<T>` 迁到标准库源码的目标形态。当前阶段只做
设计，不立刻替换 `mulberry.list`。原因很直接：缺少 `Ptr<T>`、generic struct、
generic function 和 heap allocation primitive 时，强行实现只会重新制造 workaround。

本文档建立在 `docs/PtrAndHeapObjectModel.md` 的统一模型上：`Ptr<T>` 按 C/C++
typed pointer 理解，复杂对象默认是 heap object handle，由 Boehm GC 管理。

## 目标

`List<T>` 应该是标准库容器，而不是 Dialect API。用户应该理解它类似 `std::vector<T>`
或 Rust `Vec<T>`：

- 元素类型 `T` 可以是 builtin、struct、Tensor、其它未来支持的类型。
- `List<T>` 有运行时 `length` 和 `capacity`。
- `List<T>` value 本身只是 handle，复制 handle 会共享同一份 heap storage。
- `push`、`get`、`set`、`size` 这类操作应该优先用 Mulberry 标准库源码表达。
- `mulberry.list.*` 最终应该消失，或者只保留为短期 lowering helper。

## 用户可见形态

源码里的 `List<T>` 拼写应该保持稳定：

```cherry
fn sum(xs: List<UInt64>): UInt64 {
  var i: UInt64 = 0;
  var result: UInt64 = 0;
  while i lt size(xs) {
    result = result + xs[i];
    i = i + 1;
  };
  result
}
```

长期目标不是让用户写很长的 `collections.List<T>`。标准库实现可以放在
`std.collections`，但 prelude 应该提供受控 re-export，使普通用户继续写 `List<T>`。

## 目标源码形态

下面是目标方向，不是当前 Cherry 已经支持的语法。

```cherry
package std.collections;

comptime ListStorage<T> = struct {
  length: UInt64,
  capacity: UInt64,
  data: Ptr<T>
};

comptime List<T> = struct {
  storage: Ptr<ListStorage<T>>
};
```

这里 `List<T>` 本身只是一个 handle，真正可变状态放在 heap 上的
`ListStorage<T>` 里。函数参数、返回值和赋值都复制这个 handle；`ListStorage<T>` 和
element data 由 Boehm GC 管理。

不要把 public `List<T>` 直接设计成：

```cherry
struct {
  length: UInt64,
  capacity: UInt64,
  data: Ptr<T>
}
```

原因是 Mulberry 当前设计方向是复杂对象具有引用语义。赋值、函数参数和返回值都应该
复制 handle。如果 `length/capacity/data` 直接放在 public descriptor value 里，复制后
`push(xs, value)` 只会更新当前 descriptor 的 `length/capacity`，其它 alias 看到的
descriptor 可能不同步。用 `storage: Ptr<ListStorage<T>>` 可以把可变 header 放在同一
块 heap storage 里，alias 语义更清楚。

## 目标 API

第一批标准库 API 应保持很小：

```cherry
package std.collections;

fn new<T>(): List<T>;
fn withCapacity<T>(capacity: UInt64): List<T>;

fn size<T>(xs: List<T>): UInt64;
fn get<T>(xs: List<T>, index: UInt64): T;
fn set<T>(xs: List<T>, index: UInt64, value: T): ();
fn push<T>(xs: List<T>, value: T): ();
fn pop<T>(xs: List<T>): T;
```

语法层面的 `xs[i]` 和 `size(xs)` 可以先继续是 builtin sugar，后续再 desugar 到：

```cherry
collections.get(xs, i)
collections.size(xs)
```

等 generic function type inference 稳定以后，用户不应该经常手写类型参数：

```cherry
collections.push(xs, 42)
```

如果第一阶段还没有类型参数推断，可以先允许显式形式：

```cherry
collections.push<UInt64>(xs, 42)
```

但这只是过渡语法，不应该成为最终用户体验。

## 需要补齐的语言能力

迁移 `List<T>` 之前至少需要以下能力：

- `Ptr<T>`：一等类型，支持 load、store、index、null 或等价的空指针表达。
- generic struct：允许标准库根据 `T` 生成 `ListStorage<T>` 和 `List<T>` concrete type。
- generic function：允许 `new<T>`、`push<T>`、`get<T>` 这类函数按 `T` 单态化。
- heap allocation primitive：基于当前 Boehm GC 路线，先提供 `heap.alloc<T>(count)`。
- `sizeof(T)` / `alignof(T)`：用于计算 element storage 大小和对齐。
- package re-export：prelude 能把标准库里的 `collections.List<T>` 作为裸 `List<T>` 暴露。

## Heap primitive

当前已经接入 Boehm GC，因此第一阶段不需要 `free`。最小 primitive 可以是：

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
List<Float32[?]>  -> concrete List storage/function set for that Tensor type
```

用户源码不应该出现很长的单态化名字。长名字只允许存在于编译器内部 symbol mangling
或 MLIR symbol 中，源码层必须保持短而清楚。

## 和当前 `mulberry.list` 的关系

迁移期间保留当前路径：

```text
source List<T>
  -> compiler-known ListType
  -> mulberry.list / mulberry.list_storage / mulberry.list_desc
  -> lowering
```

等标准库能力补齐后，目标路径变成：

```text
source List<T>
  -> prelude alias to std.collections.List<T>
  -> generic struct/function monomorphization
  -> heap object handle: record { storage: Ptr<ListStorage<T>> }
  -> lowering
```

这时再删除：

- `mulberry.list`
- `mulberry.list_storage`
- `mulberry.list_desc`
- `ListCreateOpLowering` / `ListGetOpLowering` / `ListDesc*` lowering
- boundary rewrite 里针对 list descriptor 的特殊逻辑

删除必须等标准库路径能覆盖当前正向能力之后再做，不要在半路制造桥接 workaround。

## 迁移顺序

建议后续按下面顺序推进：

```text
C4.1  本文档：固定 Ptr / heap object handle 方向
C4.2  实现 source-level Ptr<T>
C4.3  实现最小 pointer load/store/index 能力
C4.4  实现 Boehm-backed heap.alloc<T>(count)
C4.5  支持 generic struct，用 Mulberry 表达 ListStorage<T>
C4.6  支持 generic function，用 Mulberry 表达 List<T> API
C4.7  把 List<T> 迁到 std.collections
C4.8  删除 mulberry.list 相关 Dialect/op/lowering
```

如果某一步需要靠 lowering bridge 才能“看起来能跑”，先停下来补语言/IR 能力，不要
把 workaround 塞回 MLIRGen。
