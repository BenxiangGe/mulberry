# `Ptr<T>` 与 Object Reference 模型

本文档记录目标方向，不表示所有实现已经完成。当前代码仍有一些 by-value header
路径；后续迁移目标是采用类似 Java 的 object reference 模型：

```text
Scalar = value
Object = reference
```

`Ptr<T>` 仍然存在，但它应该主要服务于 stdlib/runtime/FFI 和底层库作者，而不是普通
用户日常代码。

Mulberry 底层仍采用 C/C++ 风格的 typed pointer 模型：

```text
Ptr<T> = pointer to T
```

也就是说，`Ptr<T>` 的语义就是“指向一个 `T` storage 的地址”。它不是 descriptor，
也不隐含 length、shape、stride 或 ownership metadata。所有 metadata 都应该显式
放在某个 struct / heap object 里。

## 基本原则

- primitive 仍然是 value：`Bool`、`UInt8`、`UInt64`、`Float32` 赋值时复制值。
- 除了 scalar，其它 source-level object 都应该是 reference：`String`、`File`、
  `Array<T, N>`、`List<T>`、`Tensor<T>` 和用户 `struct`。
- object 赋值和传参复制 reference，不复制 object storage。
- object mutation 会影响所有 alias；后续用 `const` / `mut` 约束共享可变性。
- `Ptr<T>` 是底层 typed address，不是默认 object surface。普通用户应该优先使用
  object API，而不是手写 `Ptr<T>`。
- heap object 由 Boehm GC 管理，不设计用户可见的 `free`。
- 函数返回复杂对象时返回 object reference，而不是 by-value descriptor。
- lowering 不应该为了函数边界再发明 escape descriptor 语义。

这条路线牺牲一点 indirection 和 heap allocation，换取更简单的语言语义和更低的心智
负担。对 Mulberry 当前目标来说，这个 tradeoff 是合理的。

## `Ptr<T>` 语义

`Ptr<T>` 等价于 C/C++ 的 `T*`：

```mulberry
var p: Ptr<UInt64>;
```

表示 `p` 保存一个地址，地址指向 `UInt64` storage。

当前 source-level 读写使用 C/C++ 风格的解引用语法：

```mulberry
var p: Ptr<UInt64> = heap.alloc<UInt64>();
*p = 42;
var value: UInt64 = *p;
```

连续 storage 使用带元素个数的 `heap.alloc<T>(count)` 分配，`p[i]` 表示第 `i`
个元素的 lvalue：

```mulberry
var p: Ptr<UInt64> = heap.alloc<UInt64>(3);
p[0] = 10;
p[1] = 32;
var value: UInt64 = p[0] + p[1];
```

`Ptr<T>` 自身仍然不保存长度；`count` 只决定本次 heap allocation 的字节数。
越界检查、空指针表达和分配失败策略都属于后续语言/runtime 设计。

`ptr.load()` / `ptr.store()` 这类函数不作为用户 API 暴露。它们对应的是底层
load/store 语义，源码层应该使用 `*p` 和 `*p = value`。

未来 source-level `Ptr<T>` 在 IR 中映射到：

```text
Ptr<T>
  -> semantic PtrType<T>
  -> !mulberry_core.ptr<lowered(T)>
  -> backend ptr
```

`Ptr<T>` 自身不携带长度或边界信息。比如：

```mulberry
var data: Ptr<Float32>;
```

只表示“指向某个 `Float32` 的地址”。如果要表达一段连续数组，必须另有 metadata：

```mulberry
struct Buffer<T> {
  length: UInt64,
  data: Ptr<T>
}
```

## Source object reference

目标 source 语义接近 Java：

```mulberry
struct Counter {
  value: UInt64
}

var a: Counter = Counter { 0 };
var b: Counter = a;
b.value = 1;      // a.value 也变成 1
```

这里 `a` 和 `b` 都是 object reference。用户看不到 raw pointer，也不能做 pointer
arithmetic。只有 scalar 是真正的 by-value 数据：

```mulberry
var x: UInt64 = 1;
var y: UInt64 = x;
y = 2;            // x 仍然是 1
```

这个模型的好处是：`String`、`Array`、`List`、`Tensor`、`File` 和用户 `struct`
拥有一致的赋值/传参/返回规则。代价是 aliasing 变成语言事实，必须用后续的
`const` / `mut` 规则管理共享可变性。

第一版 const/mut 目标规则：

- `const x: Object` 表示不能通过 `x` mutation object。
- `var x: Object` 表示可以通过 `x` mutation object。
- 多个 reference 可以指向同一个 object；如果要更严格的“唯一 mutable reference”，
  需要后续单独设计，不在第一版实现。
- 显式深拷贝用 `clone()` / `copy()` 这类 API，不让赋值隐式 deep copy。

## Heap object 与共享 storage

复杂对象的 runtime storage 放在 heap 上，由 object reference 指向。`Ptr<T>` 是实现
这种 reference 的底层能力，但 source type 不应该到处写成 `Ptr<Object>`。

List 的形态：

```mulberry
comptime List<T> = struct {
  length: UInt64,
  capacity: UInt64,
  data: Ptr<T>
};
```

source 层仍写：

```mulberry
var xs: List<UInt64> = list.from([1, 2, 3]);
var ys: List<UInt64> = xs;     // reference copy, not header deep copy
ys.push(4);                    // xs 也观察到同一个 list object
```

动态 Tensor 的形态：

```mulberry
comptime Tensor<T> = struct {
  data: Ptr<T>,
  rank: UInt64,
  numel: UInt64,
  sizes: List<UInt64>,
  strides: List<UInt64>
};
```

注意：旧的 `!mulberry.tensor_handle` / `tensor.handle_from_desc` 实验 IR 已删除。
真正的 Tensor object 已经采用上面这种 source-level record header 形态，不能复用旧
实验 IR 偷换语义。

`Tensor<T>` 是 source object reference，不是 descriptor surface。这里不重复 layout 细节，统一
引用 `docs/StdlibObjects.md` 和 `docs/MulberryLowering.md` 中的约定：

- `Tensor<T>` 的 source-level 形态是 ndarray-style object。
- 多维/动态 `T[...]` source surface 已删除；Tensor value 统一使用 `Tensor<T>`。

当前的设计重点不是把所有 legacy tensor 行为一次删干净，而是避免把 Tensor 语义、
memref view、函数边界 ABI 和 runtime ownership 混成一层。

String 的形态就是：

```mulberry
struct String {
  length: UInt64,
  data: Ptr<UInt8>
}
```

`String` source value 是 reference。复制 `String` 复制 reference，不复制 bytes。

string literal 直接分配 heap byte buffer，然后构造 `String` value：

```text
heap UInt8[4] = "abc\0" + String{length = 3, data = bytes}
```

这样 `String` 的 runtime object 仍然可以复用 `heap.alloc`、`ptr.index`、
`record.get_field` 和 `store` 的通用路径。这里不
再需要旧 pointer-storage alias 带来的额外 indirection。

## 函数边界

函数返回复杂对象时，返回 object reference：

```mulberry
fn make(): List<UInt64> {
  ...
}
```

动态 Tensor 同理：

```mulberry
fn makeTensor(): Tensor<Float32> {
  ...
}
```

caller 拿到的是 Tensor object reference。函数边界不需要专门的 descriptor escape
机制，也不应该把 memref descriptor 暴露成 source value。

## Alias 语义

复制 object reference 会共享同一个 heap object：

```mulberry
var a = makeList();
var b = a;
b.push(1);
```

如果 `makeList()` 返回 `List<UInt64>` object reference，`a` 和 `b` 指向同一个 list heap object，
因此 `a.size()` 能看到 `b.push(1)` 的结果。

如果用户需要深拷贝，应该显式调用：

```mulberry
var b = clone(a);
```

不要让赋值隐式深拷贝复杂对象。隐式深拷贝成本高，而且会让性能和 alias 行为变得
不透明。

## GC 与 ownership

heap object 由 Boehm GC 分配和回收：

```mulberry
var xs: List<UInt64> = list.from([1, 2, 3]);
```

只要 `xs` reference 仍能从栈、全局变量、其它 heap object 等 root 找到，Boehm 就不会
回收 list object 以及它能到达的 element buffer。扩容时，新 data buffer 也用 GC 分配；
旧 buffer 如果不再被引用，后续由 GC 回收。

当前阶段不设计：

- explicit `free`
- borrow checker
- region/arena
- allocator trait
- moving GC

这些都可以后续再做。当前优先目标是让语言和 lowering 模型简单、可解释。

## 和 descriptor 的关系

Descriptor 仍可能作为 lowering 内部工具存在，但不再作为 source-level 复杂对象跨函数
边界的主要模型。

可以保留的内部 descriptor：

- memref/linalg 需要的临时 view。
- runtime/FFI 需要的 ABI wrapper。
- lowering pass 内部的局部 helper。

不应该继续扩大的模型：

```text
source List<T>
  -> list_desc<T>
  -> escape_storage
  -> boundary rewrite
```

这条路径已经证明可以工作，但解释困难、心智负担高。新方向应逐步迁移到：

```text
source List<T>
  -> header struct
  -> ordinary record passing + shared heap data buffer
```

旧 `mulberry.list` / `list_desc` 路径已经删除。后续如果需要跨 module ABI 或 FFI
descriptor，也应作为普通 `Ptr<T>` object 的 ABI wrapper 重新设计，而不是恢复
list-specific boundary rewrite。

## 后续实现顺序

当前已完成：

```text
C4.9   引入 `*p` / `*p = value`，不把 ptr.load / ptr.store 暴露为用户 API
C4.10  清理 Ptr API 文档和测试命名
C4.11  删除旧的 Tensor handle 实验 IR
C4.12  实现 `heap.alloc<T>(count)` 和 `p[i]` 指针索引
C4.13  支持 generic struct，用 Mulberry 表达 List<T>
C4.14  支持 generic function，用 Mulberry 表达 List<T> API
C4.15  把 List<T> 迁到 std.list
C4.16  删除旧 list descriptor / escape_storage / boundary rewrite
```

后续建议：

```text
P4.5   继续验证 nested List、record field List 和 Tensor element List 的正向路径
P4.6   设计 List grow / capacity 策略，只在 training 需要时实现
P4.7   已完成设计检查：禁止把 tensor descriptor 伪装成 handle；真正的 Tensor object
       只能是 source-level `Tensor<T>` header
P4.8   如果要继续缩小 Mulberry core dialect，逐项迁移 string/file/heap/record，而不是整块删除
```

每一步都应该优先保持模型简单。如果某一步需要引入很难解释的桥接层，就说明底层能力
还没补齐，应该先停下来补能力，而不是继续 workaround。
