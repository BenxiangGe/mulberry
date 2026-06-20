# `Ptr<T>` 与 Heap Object 模型

Mulberry 后续统一采用 C/C++ 风格的 typed pointer 模型：

```text
Ptr<T> = pointer to T
```

也就是说，`Ptr<T>` 的语义就是“指向一个 `T` storage 的地址”。它不是 descriptor，
也不隐含 length、shape、stride 或 ownership metadata。所有 metadata 都应该显式
放在某个 struct / heap object 里。

## 基本原则

- primitive 仍然是 value：`Bool`、`UInt8`、`UInt64`、`Float32` 赋值时复制值。
- `struct` 默认也是 value；需要共享时显式使用 `Ptr<StructType>`。
- `List<T>`、动态 Tensor、String 这类复杂对象默认采用 heap object pointer。
- heap object 由 Boehm GC 管理，不设计用户可见的 `free`。
- 函数返回复杂对象时返回 handle，而不是 by-value descriptor。
- lowering 不应该为了函数边界再发明 escape descriptor 语义。

这条路线牺牲一点 indirection 和 heap allocation，换取更简单的语言语义和更低的心智
负担。对 Mulberry 当前目标来说，这个 tradeoff 是合理的。

## `Ptr<T>` 语义

`Ptr<T>` 等价于 C/C++ 的 `T*`：

```cherry
var p: Ptr<UInt64>;
```

表示 `p` 保存一个地址，地址指向 `UInt64` storage。

当前 source-level 读写使用 C/C++ 风格的解引用语法：

```cherry
var p: Ptr<UInt64> = heap.alloc<UInt64>();
*p = 42;
var value: UInt64 = *p;
```

连续 storage 使用带元素个数的 `heap.alloc<T>(count)` 分配，`p[i]` 表示第 `i`
个元素的 lvalue：

```cherry
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
  -> !mulberry.ptr<lowered(T)>
  -> backend ptr
```

`Ptr<T>` 自身不携带长度或边界信息。比如：

```cherry
var data: Ptr<Float32>;
```

只表示“指向某个 `Float32` 的地址”。如果要表达一段连续数组，必须另有 metadata：

```cherry
struct Buffer<T> {
  length: UInt64,
  data: Ptr<T>
}
```

## Heap object handle

复杂对象应该显式使用 `Ptr<T>` 表达共享和可变语义。`T` 本身描述对象 layout，
`Ptr<T>` 才是 heap object handle。

List 的形态：

```cherry
comptime List<T> = struct {
  length: UInt64,
  capacity: UInt64,
  data: Ptr<T>
};
```

动态 Tensor 的形态：

```cherry
comptime TensorStorage<T, Rank> = struct {
  data: Ptr<T>,
  sizes: Ptr<UInt64>,
  strides: Ptr<UInt64>
};

comptime Tensor<T, Rank> = struct {
  storage: Ptr<TensorStorage<T, Rank>>
};
```

注意：旧的 `!mulberry.tensor_handle` / `tensor.handle_from_desc` 实验 IR 已删除。
真正的 Tensor heap object handle 应该是上面这种 source-level record/Ptr 形态，不能
复用旧 marker 偷换语义。

String 的形态可以类似：

```cherry
struct StringStorage {
  length: UInt64,
  data: Ptr<UInt8>
}

struct String {
  storage: Ptr<StringStorage>
}
```

具体字段可以后续调整，但核心原则不变：value 是 handle，metadata 和 data 都通过
pointer 找到。

## 函数边界

函数返回复杂对象时，只返回 handle：

```cherry
fn make(): List<UInt64> {
  ...
}
```

如果需要返回可变共享 list object，函数应该返回：

```text
Ptr<List<UInt64>>
```

不再返回：

```text
{ length: i64, data: ptr }
```

动态 Tensor 同理：

```cherry
fn makeTensor(): Float32[?, ?] {
  ...
}
```

底层返回 Tensor handle。caller 顺着 handle 找到 `TensorStorage`，再取出 data、sizes
和 strides。

这样函数边界只需要传递普通 value / pointer / record，不需要专门的 descriptor escape
机制。

## Alias 语义

复制 handle 会共享同一个 heap object：

```cherry
var a = makeList();
var b = a;
push(b, 1);
```

如果 `makeList()` 返回 `Ptr<List<UInt64>>`，`a` 和 `b` 指向同一个 list heap object，
因此 `size(a)` 能看到 `push(b, 1)` 的结果。

如果用户需要深拷贝，应该显式调用：

```cherry
var b = clone(a);
```

不要让赋值隐式深拷贝复杂对象。隐式深拷贝成本高，而且会让性能和 alias 行为变得
不透明。

## GC 与 ownership

heap object 由 Boehm GC 分配和回收：

```cherry
var list: Ptr<List<T>> = heap.alloc<List<T>>();
```

只要 `list` 仍能从栈、全局变量、其它 heap object 等 root 找到，Boehm 就不会回收
它。扩容时，新 data buffer 也用 GC 分配；旧 buffer 如果不再被引用，后续由 GC 回收。

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
  -> heap object handle
  -> ordinary pointer/record passing
```

## 后续实现顺序

建议后续按下面顺序推进：

```text
C4.9  引入 `*p` / `*p = value`，移除临时 ptr.load / ptr.store API
C4.10 清理 Ptr API 文档和测试命名
C4.11 删除旧的 Tensor handle marker IR，避免把 descriptor reconstruction 当成 handle
C4.12 实现 `heap.alloc<T>(count)` 和 `p[i]` 指针索引
C4.13 支持 generic struct，用 Mulberry 表达 List<T>
C4.14 支持 generic function，用 Mulberry 表达 List<T> API
C4.15 把 List<T> 迁到 std.collections
C4.16 设计 Tensor heap object handle
C4.17 删除旧 list descriptor / escape_storage / boundary rewrite
```

每一步都应该优先保持模型简单。如果某一步需要引入很难解释的桥接层，就说明底层能力
还没补齐，应该先停下来补能力，而不是继续 workaround。
