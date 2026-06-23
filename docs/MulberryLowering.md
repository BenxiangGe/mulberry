# Mulberry Lowering

本文档记录当前 Mulberry lowering 的边界、ABI 形状和仍然保留的限制。

核心原则：

- MLIRGen 只生成自洽的高层 Mulberry IR。
- 源语言 `List<T>` 已经走 stdlib/comptime 路径，语义上是普通
  `Ptr<record>` heap object。
- 旧的 `!mulberry.list` / `!mulberry.list_storage` / `!mulberry.list_desc`
  类型、op 和 lowering 已删除。
- 函数边界不要再引入 list-specific descriptor rewrite。后续复杂对象 ABI 应该基于
  统一的 `Ptr<T>` heap object 模型设计。

## 当前边界

`--lower-mulberry` 现在做的是 storage-level lowering；完整 driver pipeline 会继续
lower 到 LLVM dialect、LLVM IR，并在当前正向子集上执行 JIT：

- `mulberry.tensor` lower 成 MLIR `memref`，方便 `cherry_nn` 继续 lower 到
  `linalg`、`math`、`arith` 和 `memref`。
- stdlib/comptime `List<T>` lower 成普通 record/ptr/heap 操作，不再走
  `mulberry.list.*` op。
- scalar 和 record storage 在能直接映射的地方使用 LLVM dialect。
- `-dump=mlir-llvm`、`-dump=llvm` 和 JIT 已经打开，用来验证当前支持的正向路径。
- object generation 仍然关闭，后续需要单独设计 emission path。

## 剩余 Mulberry 职责

当前 `mulberry` dialect 仍然是必要的高层对象模型 IR，不能简单按“非
`cherry_nn` dialect”整块删除。它现在承载的是源码对象语义到 backend IR 之间的
过渡层：

- `mulberry.record` / `mulberry.record.get_field` / `mulberry.record.extract`：
  表达用户 struct 和 generic struct alias 实例化后的 concrete record。
- `mulberry.ptr` / `mulberry.heap.alloc` / `mulberry.alloca` / `mulberry.load` /
  `mulberry.store` / `mulberry.ptr.index`：表达统一的 C/C++ typed pointer 模型。
- safetensors runtime API 不再有 dedicated Mulberry op。`readTensor(file, name)`
  通过 temporary `func.call @__cherry_safetensor_read_*` marker 进入 LowerMulberry，
  再展开成 shape query、Tensor allocation 和 payload read。File 已经是 stdlib
  `Ptr<FileStorage>`；String 已经是 stdlib `Ptr<StringStorage>`，不再有
  `mulberry.string.*` 专用 op。`open/close/read/write/readTensor` 都不再依赖
  file-specific dialect op。
- `mulberry.tensor.*`：表达可写 Tensor 和当前 memref / `cherry_nn` lowering
  所需的显式 Tensor value 过渡层。Tensor ABI descriptor 只存在于
  `LowerMulberry.cpp` 的 C++ helper 中，不再是 Mulberry dialect type/op。

也就是说，当前已经删除的是不该属于 dialect 的 `mulberry.list.*` 容器语义；剩余
Mulberry op/type 暂时仍是 codegen 和 lowering 之间的清晰边界。后续如果要继续减少
Mulberry 的职责，应该逐项迁移到 source/stdlib 能力或更通用的 object ABI，而不是
再引入 bridge pass 或 list-specific descriptor。

## List 分层

当前 source-level `List<T>` 由 `stdlib/std/collections.cherry` 定义：

```cherry
comptime ListStorage<T> = struct {
  length: UInt64,
  capacity: UInt64,
  data: Ptr<T>
};

comptime List<T> = Ptr<ListStorage<T>>;
```

因此 `List<T>` 和其它用户 struct 一样，走 `mulberry.record`、`mulberry.ptr`、
`mulberry.heap.alloc`、`mulberry.load/store` 和 field access 的普通 lowering
路径。

旧 `!mulberry.list<T>` / `!mulberry.list_storage<T>` / `!mulberry.list_desc<T>`
曾用于 list literal、list storage 和函数边界 descriptor。这个模型已经被
stdlib/comptime `Ptr<record>` 模型取代，相关 type、op、lowering pattern 和旧 IR
测试已经删除。

## Tensor ABI

Mulberry Tensor 是可写数组，rank 静态，dimension 可以是静态或动态。Tensor ABI
descriptor 当前按 rank 固定：

```text
TensorABI<T, rank> = {
  data: ptr<#llvm.address_space<0>>,
  sizes: array<rank x i64>,
  strides: array<rank x i64>
}
```

`data` 指向连续元素 storage；`sizes` 是每个维度的运行时大小；`strides` 是按
row-major layout 访问元素时的步长。这个 layout 借鉴 MLIR memref descriptor，
但没有直接把完整 memref ABI 暴露成 Mulberry 语言 ABI。

LowerMulberry 内部会在需要时把本地 Tensor/memref 打包成这个 ABI descriptor：
当前实现从 memref 中提取 data pointer、sizes 和 strides。

反方向是从 Tensor ABI descriptor 重建可被后续 Tensor / `cherry_nn` op 使用的
memref view。当前实现使用：

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
stride 或生命周期。当前 lowering 通过：

- `createMemRefDataPointer()` 取出 memref 的 raw aligned data pointer。
- `createTensorByteSize()` 计算整个 tensor payload 的总字节数。

MLIRGen 先生成 temporary `func.call @__cherry_file_read_*` /
`@__cherry_file_write_*` marker。LowerMulberry 识别 marker 后创建这个 raw-byte view，
再把它传给 `fread` / `fwrite`，并删除 marker declaration。这条边界和完整
Tensor ABI descriptor 不同：后者服务 Tensor view / `cherry_nn`，前者只服务 raw
file IO。marker 不是 runtime ABI，不能泄露到 lowered MLIR 之后。

旧的实验性 `!mulberry.tensor_handle` / `tensor.handle_from_desc` 已删除，避免和
新的 heap object handle 方向混淆。当前真正保留的 handle 形态是 source-level
`std.tensor.Tensor<T, Rank>`，也就是 `Ptr<TensorStorage<T, Rank>>`。

P4.7 已明确：不能把旧 descriptor surface 伪装成 Tensor handle。原因是当前
`cherry_nn` 正向路径需要 Tensor lowering 成 `memref`，再继续 lower 到 `linalg`。
因此现阶段的边界是：

- `mulberry.tensor.*` 表示高层可写 Tensor value。
- Tensor ABI descriptor 只是 LowerMulberry 内部 C++ helper，不是公开 Mulberry type/op。
- Tensor 函数边界继续 lower 成 `memref`。
- 函数边界不允许通过隐藏 descriptor bridge 暗中改变语义。

真正的 Tensor heap object handle 已经可以用 source-level `std.tensor.Tensor<T, Rank>`
表达，并通过显式 `tensor.view(handle)` 接入现有 `Float32[?, ?]` Tensor lowering。
MLIRGen 为它生成 `mulberry.tensor.view`，LowerMulberry 从 `TensorStorage` 的
`data/sizes/strides` 字段重建 memref view。不要复活旧 `tensor.handle_from_desc`，
也不要增加新的 descriptor marker。

`std.tensor` 的 source-level layout 是：

```cherry
comptime TensorStorage<T, Rank: UInt64> = struct {
  rank: UInt64,
  data: Ptr<T>,
  sizes: Ptr<UInt64>,
  strides: Ptr<UInt64>
};

comptime Tensor<T, Rank: UInt64> = Ptr<TensorStorage<T, Rank>>;
```

LowerMulberry 目前收敛两个显式边界：

- `Ptr<TensorStorage<T, Rank>> -> memref view`：`tensor.view(handle)` 已提供这条
  正向路径，给 `cherry_nn`、`linalg` 和 tensor load/store 使用。
- `memref allocation -> Ptr<TensorStorage<T, Rank>>`：把当前 tensor allocation 结果包装回
  source-level heap object handle；`tensor.pack(value)` 已提供这条 owning pack 路径。

这两个边界必须是 Tensor object model 的一部分，不能通过隐藏 descriptor 暗中
bridge。

### C8.1.19 剩余 `mulberry.tensor.*` 边界

当前不能把 `mulberry.tensor.*` 整块删除，因为这些 op 分成两类职责：

- `tensor.alloc` / `tensor.dim` / `tensor.cast` / `tensor.load` / `tensor.store`
  仍然表示 source-level `Float32[...]` 这种可写 Tensor value。LowerMulberry 把它们
  直接变成 `memref.alloc`、`memref.dim`、`memref.cast`、`memref.load` 和
  `memref.store`，这是 `cherry_nn -> linalg` 正向路径的基础。
- `tensor.view` / `tensor.pack` 是 stdlib Tensor handle 的显式边界。`tensor.view`
  把 `Ptr<TensorStorage<T, Rank>>` unwrap 成 memref-backed Tensor value；`tensor.pack`
  把一个 memref-backed Tensor value 拷贝到 Boehm heap，并返回
  `Ptr<TensorStorage<T, Rank>>`。

后续删除顺序应该保持这个分层：

1. 先让 source-level Tensor value 本身也统一成 `Ptr<TensorStorage<T, Rank>>`。
2. 再把 `cherry_nn` / raw file IO / safetensors runtime boundary 都改成显式接受
   Tensor handle 或显式 view。
3. 先删掉已经被 handle / multi-dimensional tests 重复覆盖的旧 value-path lit；保留
   `tensor.alloc/dim/cast/load/store` 的唯一正向覆盖点。旧 `mulberry.tensor.*`
   负向 verifier lit 不再维护。
4. `tensor.view/pack` 只能在有更通用的 stdlib/runtime lowering 入口后再删除。

不要用新的 descriptor bridge 替代这些 op。缺少通用 Tensor object lowering 能力时，
应该先补 source/stdlib 能力，而不是在 MLIRGen 或 LowerMulberry 里再加一条隐藏 ABI
捷径。

`readTensor(file, name)` 根据 expected type 选择返回形式：

- expected type 是 `Float32[?, ...]` 时，返回现有 Tensor value。
- expected type 是 `std.tensor.Tensor<Float32, Rank>` 时，先读取 Tensor value，再通过
  `tensor.pack` 打包成 heap Tensor handle。

这样 safetensors runtime 仍只负责读取 payload；Tensor object layout 仍由
`tensor.pack`/`tensor.view` 这两个显式边界管理。

## 字符串 Storage

Mulberry `String` 现在是 stdlib alias，不再是 builtin value。源语言语义上它就是
`Ptr<StringStorage>`，lowering 也直接把它当成普通 pointer 处理：

```text
StringStorage = { length: i64, data: ptr }
```

`length` 是源码字符串的字节数，不包含结尾的 `\0`。`data` 指向字节数据。这个
layout 是第一阶段迁移结果：函数边界和 runtime helper 传递普通 pointer，不再传
`{ length, data }` by-value descriptor。

字符串字面量会 materialize 成一个 heap byte buffer，并在末尾额外放一个 `\0`：

```text
"data/mnist.bin" -> heap UInt8[15] = "data/mnist.bin\0"
```

这个额外的 NUL 不属于 Mulberry `String` 的长度语义，只是 ABI/runtime 便利：
后续 `open/read/write/close` 这类 runtime API 如果走 C ABI，可以直接复用同一个
data pointer 作为 C string。

P5.2.1 后续收敛为更简单的方案：string literal 不再使用 string-specific op，也不再
创建 global byte array。MLIRGen 直接生成 `heap.alloc<UInt8>(length + 1)`、
逐 byte store、`heap.alloc<StringStorage>()` 和字段写入。这样 `String` 只是普通
`Ptr<StringStorage>` heap object，lowering 也复用 generic heap/ptr/record/store 路径。

`io.open`、`readTensor` 等 runtime 边界在 lowering call site 显式读取
`StringStorage.data`，再传给 `fopen` 或 safetensors runtime helper。

现在 source-level `String` 已经搬到 stdlib alias，所以这里不再需要旧的 builtin
string 语义。

## 文件 ABI

当前 `File` 是 stdlib alias，源语言语义是普通 `Ptr<FileStorage>`：

```text
FileStorage = { handle: ptr }
File = Ptr<FileStorage>
```

`handle` 保存 `fopen` 返回的 opaque C `FILE*`。`io.open/read/write/close` 已经在源码层
放在 `std.io` 下；其中 `open/close` 通过普通 runtime wrapper function 调用
`fopen`/`fclose`，`read/write` 通过 temporary `func.call` marker 进入
LowerMulberry，再改写成 `fread`/`fwrite`。

旧 `!mulberry.file` type 和 `mulberry.file.*` op 已经删除。

## 旧 Boundary Preparation 已移除

`prepare-mulberry-boundaries` 曾经把 source-level `!mulberry.list<T>` 函数边界
改写成 `!mulberry.list_desc<T>`。这条路已经和当前方向冲突：`List<T>` 现在由
stdlib/comptime 定义成普通 `Ptr<record>` 对象，函数参数和返回值走统一的
Mulberry `Ptr<T>` 模型，不再需要单独的 list descriptor boundary pass。

后续如果需要重新设计跨 module ABI，应该基于统一的 `Ptr<T>` heap object 模型做，
不要恢复旧的 `!mulberry.list -> !mulberry.list_desc` rewrite。

## 当前正向支持

当前已经支持的核心路径：

- 函数内部 Tensor、stdlib/comptime List、`cherry_nn` lowering。
- Tensor 函数参数和返回值 lower 成 memref function boundary。
- `List<T>` 参数和返回值按普通 `Ptr<ListStorage<T>>` 传递。
- `Tensor<T, Rank>` handle 参数和返回值按普通 `Ptr<TensorStorage<T, Rank>>`
  传递；只有显式 `tensor.view(handle)` 时才 unwrap 成 memref-backed Tensor value。
- record field 中的 `Tensor<T, Rank>` handle 也是普通 pointer field，field access
  不触发 Tensor descriptor bridge。
- list literal 已经通过 `std.collections.withCapacity` / `push` 初始化，不再生成
  `mulberry.list.create`。
- list 读取和长度查询已经通过 `std.collections.get` / `size` 函数调用，不再生成
  `mulberry.list.get` / `mulberry.list.size`。

源码级正向例子：

```cherry
fn make(): List<Tensor<Float32, 2>> {
  const w: Float32[2, 2] = [[1.0, 0.0], [0.0, 1.0]];
  [tensor.pack(w)]
}

fn main(): UInt64 {
  const x: Float32[2, 1] = [[0.2], [0.8]];
  argmax(matmul(tensor.view(collections.get(make(), 0)), x))
}
```

lowering 后的关键路径是：

```text
List<T> value
  -> Ptr<ListStorage<T>>
  -> mulberry.ptr / mulberry.record field access
  -> LLVM pointer / LLVM struct storage
```

## Ownership

当前复杂对象统一倾向于 C/C++ 指针模型：`Ptr<T>` 是 typed pointer，heap object
由 Boehm GC 管理。`List<T>` 本身就是 `Ptr<ListStorage<T>>`，返回或传参只复制这个
pointer value，不再需要 list-specific escape descriptor。

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
- `List<UInt64>`、`List<Tensor<Float32, 2>>`、`List<List<UInt64>>` 在 Sema 阶段会得到
  不同的 concrete `ListStorage<T>` record type。
- 每个 concrete `ListStorage<T>` 都走普通 record/ptr lowering。

可以把当前 `List<T>` 理解成语言标准库里的 `std::vector<T>` 简化版本：

```text
用户层：List<T>
语义层：Ptr<ListStorage<T>>
lowering 层：record/ptr/heap object
runtime 层：Boehm-managed heap object
```

当前不引入 allocator、异常安全、move/copy 语义或复杂 grow 策略。等 training
需要更完整的容器语义时，再单独设计。

## 下一步

R3.8.88 已完成：Nielsen for-loop 推理脚本可以通过 `--dump=lowered-mlir`，
lowered IR 使用 `scf`、`memref`、`linalg`、`arith` 和 `math`，不再残留
`mulberry` 或 `cherry_nn` op。

R3.8.89 已完成：scalar `List<T>` 函数边界可以通过 descriptor ABI lowering，
当前覆盖 `UInt64`、`Float32` 和 `index` 这类 scalar element。

R3.8.90 已完成：本地 list storage 曾支持内部显式 dealloc，`List<Tensor>` 返回
descriptor 的 heap data 曾使用 caller-side `free`。这只是历史阶段，不是完整
runtime ownership。

R3.8.91 已完成：返回的 scalar `List<T>` 和 `List<Tensor>` 曾统一走 escaping
descriptor storage，并复制到 Boehm-managed ABI data。这是旧阶段的历史记录。

R3.8.92 已完成：当前支持的正向路径可以继续 lower 到 LLVM dialect / LLVM IR，
并通过 JIT 执行。Nielsen for-loop 推理脚本可以直接运行，输出 `7`。

G3 已完成：接受 Boehm-only 方向后，删除 local `List<T>` auto dealloc 插入逻辑。
G4 已完成：删除 `mulberry.list.dealloc` 和 `mulberry.list.desc_dealloc`。List
descriptor data 的生命周期由 Boehm GC 管理，不再保留显式 cleanup op。

R3.8.93：暂时不扩展 object generation；先增加 training 需要的 `matsub`、
`hadamard`、`sigmoidPrime` 等 `cherry_nn` ops，并保持 for-loop 推理 JIT 路径稳定。

R3.8.94：等 training script 需要时，再补更完整的 ownership / dealloc / runtime
策略，而不是提前为负向场景堆 workaround。

P4.1 已完成：source surface 不再生成 `mulberry.list.create/get/size`，而是走
stdlib/comptime `List<T>`。

P4.2 已完成：删除旧 `prepare-mulberry-boundaries` pass。

P4.3 已完成：删除旧 `mulberry.list.*` type/op/lowering/test。source-level
`List<T>` 保留为 stdlib/comptime `Ptr<ListStorage<T>>` 路径。

P4.4 已完成：明确剩余 `mulberry` dialect 边界。当前不能整块删除 `mulberry`，
因为它仍承载 record/ptr/heap/string/file/tensor 的高层对象模型；后续清理必须按
具体 op/type 逐项迁移。

P4.7 已完成设计检查：当前不把旧 descriptor surface 伪装成 source-level handle。
P5.7 已完成：公开 Tensor descriptor type/op 已删除；LowerMulberry 内部仍保留
Tensor ABI descriptor helper，用于 memref 和 `tensor.view(handle)` 之间的局部转换。
