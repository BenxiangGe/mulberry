# Mulberry Lowering

本文档记录当前 Mulberry lowering 的边界、ABI 形状和仍然保留的限制。

核心原则：

- MLIRGen 只生成自洽的高层 Mulberry IR。
- 源语言 `List<T>` 已经走 stdlib/comptime 路径，语义上是普通 object reference；
  赋值、传参和返回复制 reference，不复制 `length/capacity/data` header。
- 旧的 `!mulberry.list` / `!mulberry.list_storage` / `!mulberry.list_desc`
  类型、op 和 lowering 已删除。
- 函数边界不要再引入 list-specific descriptor rewrite。后续复杂对象 ABI 应该基于
  统一的 internal pointer heap object 模型设计。

## 当前边界

`--lower-mulberry` 现在做的是 storage-level lowering；完整 driver pipeline 会继续
lower 到 LLVM dialect、LLVM IR，并在当前正向子集上执行 JIT：

- `mulberry_core.tensor` lower 成 MLIR `memref`，供 core Tensor load/store 和后续外部
  NN package 复用。
- stdlib/comptime `List<T>` lower 成普通 record/ptr/heap 操作，不再走
  `mulberry.list.*` op。
- scalar 和 record storage 在能直接映射的地方使用 LLVM dialect。
- `-dump=mlir-llvm`、`-dump=llvm` 和 JIT 已经打开，用来验证当前支持的正向路径。
- object generation 仍然关闭，后续需要单独设计 emission path。

## 剩余 Mulberry 职责

当前 `mulberry_core` dialect 仍然是必要的高层对象模型 IR，不能简单按“非
NN dialect”整块删除。它现在承载的是源码对象语义到 backend IR 之间的
过渡层：

- `mulberry_core.record` / `mulberry_core.record.get_field` / `mulberry_core.record.extract`：
  表达用户 struct 和 generic struct alias 实例化后的 concrete record。
- `mulberry_core.ptr` / `mulberry_core.heap.alloc` / `mulberry_core.alloca` / `mulberry_core.load` /
  `mulberry_core.store` / `mulberry_core.ptr.index`：表达 compiler/stdlib internal typed
  pointer storage 模型。
- safetensors runtime API 不再有 dedicated Mulberry op。`safetensors.readTensor(file, name)`
  是 stdlib 函数，内部使用 Mulberry cursor JSON parser、`tensor.zeros([shape])`
  和 `io.read` 读取 payload。
  File 已经是 stdlib `std.io.File` object；String 已经是 stdlib
  `std.string.String` object，不再有 `mulberry.string.*` 专用 op。
  `io.open/close/read/write` 和 `safetensors.readTensor` 都不再依赖 file-specific
  dialect op。
- `mulberry_core.tensor.*`：表达可写 Tensor 和当前 memref lowering 所需的显式 Tensor
  value 过渡层。Tensor ABI descriptor 只存在于
  `LowerMulberry.cpp` 的 C++ helper 中，不再是 Mulberry core dialect type/op。

这里的“core”只表示 compiler-owned IR boundary，不表示一个像 `mulberry.nn` 那样
可独立 import/link 的外部 package。`mulberry_core` 目前随 compiler 构建，提供
`PtrType`、`RecordType`、heap/ptr/record ops 和 internal Tensor lowering path。

`mulberry_core` 也不是长期 public ABI。它更像当前 MLIR core/CIR reusable
infrastructure 还不够成熟时的临时 bridge：如果未来 MLIR core 提供可复用的
record/ptr/object model，或者 ClangIR/CIR 的相关能力被拆成通用 MLIR 组件，
Mulberry 应该逐步把对应职责迁过去，而不是把 `mulberry_core` 固化成永久基础设施。

也就是说，当前已经删除的是不该属于 dialect 的 `mulberry.list.*` 容器语义；剩余
Mulberry op/type 暂时仍是 codegen 和 lowering 之间的清晰边界。后续如果要继续减少
Mulberry 的职责，应该逐项迁移到 source/stdlib 能力或更通用的 object ABI，而不是
再引入隐藏 ABI pass 或 list-specific descriptor。

## Array

`Array<T, N>` 的目标定位是 Mulberry 的通用语言数组。普通 `[]` literal 后续应该默认
生成 Array，而不是 Tensor 或 List。Tensor 是显式 ndarray/bufferization 类型；List 是
growable container，不再和 array literal 抢默认解释权。

当前实现已经从 memref handle 挪到普通语言对象：一维静态 `T[N]` 会在 Sema 里变成
`Array<T, N>`，MLIRGen 把它表示成一个 `{ length, data }` header record。`data` 是
internal pointer storage，由 heap allocation 分配。非 extern 参数、local binding、
assignment 和 return 都按 object reference 传递，不再复制 header。

这条路径刻意保持简单：

- array literal 分配 heap data buffer 和 heap header object。
- `array[i]` 通过 header.data 做 `ptr.index`，不再使用 `memref.load/store`。
- 非 extern 函数参数和返回值传递 object reference。
- `Array<T, N>` 可以进入 `struct` field 和 nested Array；raw pointer/heap spelling
  不再是 user source surface。
- 当前不引入 `mulberry_core.array.*` op/type；Array 复用 `record/ptr/heap/load/store`
  lowering，避免扩大 dialect surface。

多维或动态 `T[...]` source surface 已删除。source 层只写 `Tensor<T>`，由
`tensor.from(array)`、`tensor.zeros(shape)` 或 IO API 显式构造 Tensor。

## List 分层

当前 source-level `List<T>` 由 `stdlib/std/list.mulberry` 定义：

```mulberry
comptime List<T> = struct {
  length: UInt64,
  capacity: UInt64,
  data: Ptr<T>
};
```

因此 `List<T>` 和其它用户 struct 一样，走 `mulberry_core.record`、`mulberry_core.ptr`、
`mulberry_core.heap.alloc`、`mulberry_core.load/store` 和 field access 的普通 lowering
路径。

旧 `!mulberry.list<T>` / `!mulberry.list_storage<T>` / `!mulberry.list_desc<T>`
曾用于 list literal、list storage 和函数边界 descriptor。这个模型已经被
stdlib/comptime header struct 模型取代，相关 type、op、lowering pattern 和旧 IR
测试已经删除。

## Tensor Lowering Boundary

Source-level `Tensor<T>` 现在是 ndarray-style header，rank 和 shape 都是 runtime
metadata。它应该按 NumPy `ndarray` 来理解：Tensor 是显式数值计算对象，不是普通数组
literal 的默认解释。

多维或动态 `T[...]` source surface 已删除。一维静态 `Float32[N]` 和其它 `T[N]`
一样表示 fixed-size `Array<T, N>`。Tensor value 统一通过 `Tensor<T>` header 表达。

LowerMulberry 内部的 Tensor ABI descriptor 当前按 memref rank 固定：

```text
TensorABI<T, rank> = {
  data: ptr<#llvm.address_space<0>>,
  sizes: array<rank x i64>,
  strides: array<rank x i64>
}
```

`data` 指向连续元素 storage；`sizes` 是每个维度的运行时大小；`strides` 是按
row-major layout 访问元素时的步长。这个 layout 借鉴 MLIR memref descriptor，
但没有直接把完整 memref ABI 暴露成 Mulberry 语言 ABI，也不是 source-level
`Tensor<T>` 的定义。

LowerMulberry 内部会在需要时把本地 Tensor/memref 打包成这个 ABI descriptor：
当前实现从 memref 中提取 data pointer、sizes 和 strides。

反方向是从 Tensor ABI descriptor 重建可被后续 tensor lowering 或外部
NN package 使用的 memref view。当前实现使用：

- `llvm.extractvalue` 取出 descriptor 的 data、sizes 和 strides。
- `ptr.from_ptr` 把 ABI data pointer 变成 MLIR ptr dialect value。
- `memref.reinterpret_cast` 用 metadata 重建 ranked memref view。
- `memref.memory_space_cast` 回到普通 tensor lowering 使用的 memref type。

这里有一个重要约束：重建 view 不拥有 data 的生命周期。它只重建 view；data
必须由函数参数、heap object storage 或未来 runtime 保证仍然有效。

### Tensor raw-byte view

`read(file, tensor)` / `write(file, tensor)` 需要的是一个更小的 lowering-only 边界：

```text
TensorByteView = {
  data: ptr,
  byteSize: i64
}
```

这个 view 只表达“把 tensor 当作连续字节块读写”的 runtime ABI，不表达 shape、
stride 或生命周期。

当前 source 层 `io.read<T>` / `io.write<T>` 是 stdlib generic wrapper。它们接收
`Tensor<T>` header，通过 `buffer.data` 和 `buffer.numel * sizeof(T)` 计算 runtime
需要的 raw-byte view，再用 `ptr.asUInt8(buffer.data)` 传给
`mulberry_file_read` / `mulberry_file_write` runtime wrapper。这条边界只服务 raw file IO。

旧的实验性 `!mulberry.tensor_handle` / `tensor.handle_from_desc` 已删除，避免和
新的 stdlib Tensor header 方向混淆。当前 source-level Tensor object 形态是
`std.tensor.Tensor<T>` record header。

P4.7 已明确：不能把旧 descriptor surface 伪装成 Tensor header。当前边界是：

- `mulberry_core.tensor.*` 是 compiler-owned internal Tensor lowering value。
- Tensor ABI descriptor 只是 LowerMulberry 内部 C++ helper，不是公开 Mulberry type/op。
- 函数边界不允许通过隐藏 descriptor 边界暗中改变语义。

真正的 Tensor header 已经可以用 source-level `std.tensor.Tensor<T>` 表达。源码不再
暴露 ranked tensor view/pack bridge。需要接入 memref/linalg 或外部 NN package 时，后续
应在 package/lowering 内部消费 `Tensor<T>` header，而不是重新引入公开 descriptor
surface。

`std.tensor` 的 internal source layout 是：

```mulberry
comptime Tensor<T> = struct {
  data: Ptr<T>,
  rank: UInt64,
  numel: UInt64,
  sizes: List<UInt64>,
  strides: List<UInt64>
};
```

LowerMulberry 目前收敛两个 source-level 构造入口：

- `Array -> Tensor<T>`：`tensor.from(array)` 已提供显式构造入口。MLIRGen 会把
  fixed-size Array value 或直接传入的 array literal 按 row-major 顺序 flatten 到
  Tensor payload，并构造 `sizes/strides` metadata。
- `Tensor<T>` zero 初始化：`tensor.zeros([shape])` 用显式 runtime shape 创建
  zero-filled Tensor header。

普通 `[]` 不再根据 expected type 直接变成 `Tensor<T>` 或 `List<T>`。需要
`Tensor<T>` 时通过 `tensor.from(array)` / `tensor.zeros(shape)` 显式进入，需要
`List<T>` 时通过 `list.from(array)` / `list.withCapacity(0)` 显式进入。

`Tensor<T>` header 的普通元素访问由 MLIRGen 直接读取 header 的 `data` 和 `strides`
字段，生成 linear-offset pointer load/store。这条路径是 source-level object model 的
一部分。

### C8.1.19 剩余 `mulberry_core.tensor.*` 边界

source-level ranked tensor surface 已删除，测试也不再维护旧负向路径。
`mulberry_core.tensor.*` 后续只应作为 compiler-owned internal Tensor lowering 机制逐步
收缩；不要再从 Mulberry 源码暴露对应语法。

## 字符串 ABI

Mulberry `String` 现在是 stdlib object，不再是 builtin value，也不是旧的 heap
object handle alias：

```mulberry
struct String {
  length: UInt64,
  data: Ptr<UInt8>
}
```

`length` 是源码字符串的字节数，不包含结尾的 `\0`。`data` 是 internal pointer field，
指向字节数据。函数参数、返回值和 record field 里的 `String` 都按 object reference
传递。

字符串字面量会 materialize 成一个 heap byte buffer，并在末尾额外放一个 `\0`：

```text
"data/mnist.bin" -> heap UInt8[15] = "data/mnist.bin\0"
```

这个额外的 NUL 不属于 Mulberry `String` 的长度语义，只是 ABI/runtime 便利：
后续 `open/read/write/close` 这类 runtime API 如果走 C ABI，可以直接复用同一个
data pointer 作为 C string。

string literal 不再使用 string-specific op，也不再创建 global byte array。MLIRGen
直接生成 `heap.alloc<UInt8>(length + 1)`、逐 byte store，然后构造
`String { length, data }` object。这样 `String` 只是普通 stdlib struct，
lowering 复用 generic heap/ptr/record/store 路径。

`io.open`、`safetensors.readTensor` 等 runtime 边界是普通 stdlib wrapper。它们在 Mulberry
源码里显式读取 `File.handle` / `String` value，再调用对应 extern runtime helper。

现在 source-level `String` 已经搬到 stdlib struct，所以这里不再需要旧的 builtin
string 语义，也不需要旧 pointer-storage alias 带来的额外 indirection。

## 文件 ABI

当前 `File` 是 stdlib object：

```mulberry
struct File {
  handle: Ptr<UInt8>
}
```

`handle` 是 internal pointer field，保存 `fopen` 返回的 opaque C `FILE*`。`io.open` 调用 runtime wrapper
`mulberry_file_open` 得到 raw handle，再包装成 `File { handle }`。`io.close` 取出
`file.handle` 后调用 `mulberry_file_close`。`io.read/write` 是普通 stdlib generic
函数：它们在 stdlib 内部取出 `File.handle` 和 `Tensor<T>.data`，把 data pointer 用 `ptr.asUInt8`
重解释成 byte pointer，再调用 `mulberry_file_read` /
`mulberry_file_write` runtime wrapper。

旧 `!mulberry.file` type 和 `mulberry.file.*` op 已经删除。

## 旧 Boundary Preparation 已移除

`prepare-mulberry-boundaries` 曾经把 source-level `!mulberry.list<T>` 函数边界
改写成 `!mulberry.list_desc<T>`。这条路已经和当前方向冲突：`List<T>` 现在由
stdlib/comptime 定义成普通 object；非 extern 函数参数/返回 ABI 现在按内部 object
reference 传递。不再需要单独的 list descriptor boundary pass。

后续如果需要重新设计跨 module ABI，应该基于统一的 object reference ABI 做，
不要恢复旧的 `!mulberry.list -> !mulberry.list_desc` rewrite。

## 当前正向支持

当前已经支持的核心路径：

- 函数内部 stdlib/comptime List/Tensor lowering。
- `List<T>` / `Tensor<T>` / `String` / `File` / 用户 struct 的非 extern 参数在
  source 层写 `T`，MLIR 函数签名内部使用 `Ptr<T>` ABI。
- `List<T>` / `Tensor<T>` / `String` / `File` / 用户 struct 的局部 binding 和
  assignment 使用 reference slot；object literal 分配 GC heap storage。
- `List<T>` / `Tensor<T>` / `String` / `File` / 用户 struct 的非 extern 函数返回
  也使用内部 `Ptr<T>` ABI；source 层仍写 `T`。
- record field 中的 `Tensor<T>` 也是普通 record field，field access 不触发 Tensor
  descriptor 边界改写。
- list literal 已经通过 `std.list.withCapacity` / `List.push` 初始化，不再生成
  `mulberry.list.create`。
- list 读取和长度查询已经通过 `xs[index]` / `xs.size()`，不再生成
  `mulberry.list.get` / `mulberry.list.size`。

源码级正向例子：

```mulberry
fn make(): List<Tensor<Float32>> {
  const w: Tensor<Float32> =
      tensor.from([[1.0, 0.0], [0.0, 1.0]]);
  return list.from([w]);
}

fn main(): UInt64 {
  const x: Tensor<Float32> = tensor.from([[0.2], [0.8]]);
  return x[0, 0];
}
```

lowering 后的关键路径是：

```text
List<T> non-extern parameter
  -> source `List<T>`
  -> MLIR function ABI `mulberry_core.ptr<record<List<T>>>`
  -> LLVM pointer

List<T> method receiver
  -> source `self: List<T>`
  -> Sema readonly object reference, or `mut self: List<T>` for mutation
  -> mulberry_core.ptr / mulberry_core.record field access
  -> LLVM pointer to temporary/list slot
```

## Ownership

当前复杂对象统一倾向于 Java-style object reference 模型：source 层写 `T`，内部
ABI 可以用 typed pointer 指向 heap storage，生命周期由 Boehm GC 管理。非 extern
参数、local binding、assignment 和 function return 对 struct-shaped object 与
`Array<T, N>` 都已经按 reference 传递。
旧的 list-specific escape descriptor 已经不再需要。

旧的 `mulberry.list.escape_storage`、`mulberry.list.dealloc`、
`mulberry.list.desc_dealloc` 以及其它 `mulberry.list.*` op 已删除。

## 仍然限制

以下场景仍然不应该硬凑：

- 直接写旧 `mulberry.list.*` IR 的测试和例子已删除，当前不应再新增使用点。
- external function、indirect symbol use、function pointer 或跨 module boundary
  的复杂对象 ABI。
- 完整 ownership，包括 arena、precise GC、move/copy 语义。
- memref allocation result -> Tensor heap object handle 的 pack 入口。
- unsupported source patterns 的完整 LLVM/JIT 支持。
- object generation。

这些限制是有意保留的。缺少底层 ABI 或 ownership 设计时，继续 fail-fast 比生成
表面可用、实际语义不成立的 IR 更好。

## 借鉴 `std::vector<T>`

Mulberry 不需要完整 C++ template 语言特性，但 `List<T>` 的 stdlib/comptime 实现
可以借鉴 monomorphization 思想：

- `List<T>` 在语义层是泛型容器别名。
- `List<UInt64>`、`List<Tensor<Float32>>`、`List<List<UInt64>>` 在 Sema 阶段会得到
  不同的 concrete `List<T>` record type。
- 每个 concrete `List<T>` 都走普通 record/ptr lowering。

可以把当前 `List<T>` 理解成语言标准库里的 `std::vector<T>` 简化版本：

```text
用户层：List<T>
语义层：List<T> header struct
lowering 层：record/ptr/heap object
runtime 层：Boehm-managed heap object
```

当前不引入 allocator、异常安全、move/copy 语义或复杂 grow 策略。等 training
需要更完整的容器语义时，再单独设计。

## 已完成清理

当前清理结果：

- Nielsen for-loop 推理脚本可以 lower 到 LLVM dialect / LLVM IR，并通过 JIT 直接
  运行，输出 `7`。
- `mulberry.nn` primitives 不再放在 core codegen / lowering 里扩展，改由独立
  `MulberryNNPackage` 提供 dialect、op、pass 和 lowering pipeline。
- Boehm-only 方向已接受，旧 local `List<T>` auto dealloc 插入逻辑和
  `mulberry.list.dealloc` / `mulberry.list.desc_dealloc` 已删除。
- 旧 scalar/list descriptor function-boundary 方案已经废弃；`List<T>` 统一使用
  stdlib/comptime header struct，非 extern 参数走内部 reference ABI。

后续等 training script 需要时，再补更完整的 ownership / dealloc / runtime 策略，
不要提前为负向场景堆 workaround。

P4.1 已完成：source surface 不再生成 `mulberry.list.create/get/size`，而是走
stdlib/comptime `List<T>`。

P4.2 已完成：删除旧 `prepare-mulberry-boundaries` pass。

P4.3 已完成：删除旧 `mulberry.list.*` type/op/lowering/test。source-level
`List<T>` 保留为 stdlib/comptime header struct 路径。

P4.4 已完成：明确剩余 `mulberry` dialect 边界。当前不能整块删除 `mulberry`，
因为它仍承载 record/ptr/heap/string/file/tensor 的高层对象模型；后续清理必须按
具体 op/type 逐项迁移。

P4.7 已完成设计检查：当前不把旧 descriptor surface 伪装成 source-level handle。
P5.7 已完成：公开 Tensor descriptor type/op 已删除；LowerMulberry 内部仍保留
Tensor ABI descriptor helper，用于 internal Tensor/memref 转换。
