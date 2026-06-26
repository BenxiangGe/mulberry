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
- `List<T>` 是浅拷贝 header struct；真正共享的是 header 里的 heap data buffer。
- 动态 Tensor 这类复杂共享对象默认采用 heap object pointer。
- `String` / `File` 当前是小的 by-value wrapper；它们内部的 `data` / `handle` 字段
  才是 pointer。
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

## Heap object 与共享 storage

复杂对象可以显式使用 `Ptr<T>` 表达共享和可变语义。`T` 本身描述对象 layout，
`Ptr<T>` 才是 heap object handle。`List<T>` 是例外：它自身是一个小 header struct，
共享的是 `data` 指向的连续元素 storage。

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
marker 偷换语义。

`Tensor<T>` 是 by-value header，不是 descriptor marker。`rank` 是 runtime metadata；
`sizes` 和 `strides` 是 `List<UInt64>` header；`data` 指向 dense row-major payload。
第一版即使只支持 contiguous Tensor，也保留
`strides` 字段，避免后面支持 slice/view 时重排对象 layout。

当前 C8 的设计结论是：先实现 Tensor header 和显式 view bridge，但不把所有
source-level `Float32[?, ?]` 一次性改成 Tensor header。原因是现有 Tensor value
lowering 仍以 MLIR `memref` 为核心；如果强行整体替换，会把 Tensor 语义、memref
view、函数边界 ABI 和 runtime ownership 一次性混在一起，风险很高。

因此现阶段保留：

- source-level `Float32[?, ?]` 仍是可写 Tensor value，codegen 输出
  `mulberry.tensor.*`。
- source-level `std.tensor.Tensor<T>` 是普通 record header。
- `tensor.view(value)` 是显式 unwrap 入口，会把 Tensor header 转成当前旧 Tensor
  value 路径使用的 memref view。
- lowering 继续把 Tensor value 转成 `memref`，供 tensor load/store、`linalg` interop
  和后续外部 NN package 使用。
- 公开 Tensor descriptor type/op 已删除；内部 Tensor ABI descriptor 只作为
  LowerMulberry 的局部 C++ helper 存在。
- descriptor helper 不允许变成函数边界 ABI，也不能被包装成“伪 Tensor header”。

迁移仍然分阶段做：

1. 已补齐 rank/shape metadata 的 source-level 表达。
2. 已定义 Tensor header 的 layout 和 alias 语义。
3. 已实现 Tensor header -> memref view 的显式 unwrap lowering 入口：`tensor.view(value)`。
4. 已实现 memref allocation result -> Tensor header 的显式 owning pack 入口：
   `tensor.pack(value)`。
5. 最后删除仅为旧 memref boundary 服务的 descriptor helper。

String 的形态就是：

```cherry
struct String {
  length: UInt64,
  data: Ptr<UInt8>
}
```

`String` value 自身是 by-value record wrapper。复制 `String` 会复制 `length` 和
`data` 两个字段；真正共享的是 `data` 指向的 byte buffer。

string literal 直接分配 heap byte buffer，然后构造 `String` value：

```text
heap UInt8[4] = "abc\0" + String{length = 3, data = bytes}
```

这样 `String` 的 source-level 语义就是普通 struct value，literal materialization 也
直接复用 `heap.alloc`、`ptr.index`、`record.get_field` 和 `store` 的通用路径。这里不
再需要旧 pointer-storage alias 带来的额外 indirection。

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

动态 Tensor 的最终对象模型也应该显式返回 handle：

```cherry
fn makeTensor(): Tensor<Float32> {
  ...
}
```

caller 拿到 header 后可以直接读取 `data/rank/numel/sizes/strides`。当前
`Float32[?, ?]` 函数边界仍然保留为 memref-backed Tensor value 兼容路径；不要把它
描述成 heap object ABI。

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
因此 `a.size()` 能看到 `b.push(1)` 的结果。

如果用户需要深拷贝，应该显式调用：

```cherry
var b = clone(a);
```

不要让赋值隐式深拷贝复杂对象。隐式深拷贝成本高，而且会让性能和 alias 行为变得
不透明。

## GC 与 ownership

heap object 由 Boehm GC 分配和回收：

```cherry
var xs: List<UInt64> = [1, 2, 3];
```

只要 `xs.data` 仍能从栈、全局变量、其它 heap object 等 root 找到，Boehm 就不会回收
它指向的 element buffer。扩容时，新 data buffer 也用 GC 分配；旧 buffer 如果不再被
引用，后续由 GC 回收。`List<T>` 的 header 本身按普通 struct value 传递和复制。

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
C4.11  删除旧的 Tensor handle marker IR
C4.12  实现 `heap.alloc<T>(count)` 和 `p[i]` 指针索引
C4.13  支持 generic struct，用 Mulberry 表达 List<T>
C4.14  支持 generic function，用 Mulberry 表达 List<T> API
C4.15  把 List<T> 迁到 std.collections
C4.16  删除旧 list descriptor / escape_storage / boundary rewrite
```

后续建议：

```text
P4.5   继续验证 nested List、record field List 和 Tensor element List 的正向路径
P4.6   设计 List grow / capacity 策略，只在 training 需要时实现
P4.7   已完成设计检查：禁止把 tensor descriptor 伪装成 handle；真正的 Tensor object
       只能是 source-level `Tensor<T>` header
P4.8   如果要继续缩小 Mulberry dialect，逐项迁移 string/file/heap/record，而不是整块删除
```

每一步都应该优先保持模型简单。如果某一步需要引入很难解释的桥接层，就说明底层能力
还没补齐，应该先停下来补能力，而不是继续 workaround。
