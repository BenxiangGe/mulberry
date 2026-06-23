# 标准库对象模型

本文档记录 P5 阶段的方向：把 `String`、`File`、`Tensor` 这类语言对象逐步从
dialect-specific type/op 拿回到 stdlib/runtime object model。目标不是马上删除
`mulberry` dialect，而是避免继续用 dialect 扩展标准库能力。

## 动机

如果一个语言无法用自己的语言能力实现核心标准库，那么语言基础能力还不够强。旧的
dialect 路径会让每个标准库对象都变成一套低层 IR 机制：

```text
type/op -> verifier -> MLIRGen 特判 -> lowering pattern -> ABI descriptor
```

这相当于用 LLVM/MLIR 这种“汇编级”工具实现标准库对象。它能工作，但抽象层级太低，
也容易把语言语义、runtime ABI 和 lowering workaround 混在一起。

`List<T>` 已经证明了更好的方向：

```text
source type/API
  -> stdlib/comptime object layout
  -> Ptr<T> / record / heap.alloc / load / store
  -> generic lowering
```

后续 `String`、`File`、`Tensor` 应该尽量复用这条路。

## 基本原则

- 标准库对象优先用 Mulberry struct、`Ptr<T>`、`heap.alloc<T>()` 和 generic function
  表达。
- Dialect 只保留当前正向路径必须知道的高层边界，不继续扩大对象语义。
- ABI descriptor 可以存在，但只能是 lowering/runtime helper，不能伪装成 source-level
  object handle。
- 每次迁移一个对象类型，先保证现有 JIT 正向路径不退化。
- 遇到缺失能力时补语言能力，不在 lowering 里加 object-specific workaround。

## 目标对象形态

### List

当前已完成：

```cherry
comptime ListStorage<T> = struct {
  length: UInt64,
  capacity: UInt64,
  data: Ptr<T>
};

comptime List<T> = Ptr<ListStorage<T>>;
```

`List<T>` 是标准库对象，不再有 `mulberry.list.*` type/op/lowering。

### String

当前形态：

```cherry
struct StringStorage {
  length: UInt64,
  data: Ptr<UInt8>
}

comptime String = Ptr<StringStorage>;
```

如果后续需要可变字符串或 byte buffer，再引入 `capacity`：

```cherry
struct ByteBufferStorage {
  length: UInt64,
  capacity: UInt64,
  data: Ptr<UInt8>
}
```

`String` 现在是 stdlib alias，语义上就是 `Ptr<StringStorage>`。

- `length` 是 Mulberry 字符串长度，不包含结尾 NUL。
- `data` 指向 byte storage。第一版继续保证 literal data 以 NUL 结尾，方便 C
  runtime API 复用同一个 pointer。
- string literal 的 bytes 直接用 `heap.alloc<UInt8>(length + 1)` 分配并逐 byte 填充。
- string literal 的 `StringStorage` header 同样用 Boehm heap 分配并填充。

也就是说，literal lowering 的目标形态是：

```text
"abc"
  -> heap UInt8[4] = "abc\0"
  -> heap.alloc<StringStorage>()
  -> storage.length = 3
  -> storage.data = bytes
  -> Ptr<StringStorage>
```

不再经过 string-specific op 的原因是：`String` 已经是普通
`Ptr<StringStorage>` heap object。literal materialization 可以直接复用
`heap.alloc`、`ptr.index`、`record.get_field` 和 `store` 这条通用对象路径。

runtime 边界现在直接接收 `Ptr<StringStorage>`，由 helper 或 lowering 取出 `data` 和
`length`。如果只是调用 C `fopen` 这种需要 NUL-terminated string 的 API，可以只用
`data`；如果后续实现非 C API，则可以同时使用 `length`。

旧 `{ length, data }` String descriptor 已经不再是语言 ABI。

String 的调用链现在是：

```text
String literal
  -> Parser: StringLiteralExpr
  -> Sema: stdlib alias String
  -> TypeConverter: String -> Ptr<StringStorage>
  -> MLIRGen: heap.alloc bytes + heap.alloc StringStorage + field stores
  -> LowerMulberry: generic heap/ptr/record/store lowering
```

`io.open(path, mode)` 和 `readTensor(file, name)` 的 String 参数也走这条链。lowering
里的 file 和 safetensors runtime boundary 都调用 `loadStringDataPointer()`，从
`Ptr<StringStorage>` 里取出 `data`。

当前已经完成的实现：

1. `stdlib/std/string.cherry` 定义了 `StringStorage` 和 `comptime String = Ptr<StringStorage>`。
2. `MLIRGen` 直接为 string literal 创建 heap byte buffer 和 `StringStorage` heap
   object，并写入 `length` 和 `data` 字段。
3. lowering 通过 `loadStringDataPointer()` 从 `Ptr<StringStorage>` 里取 `data`，
   供 `fopen` 和 safetensors runtime helper 使用。
4. 旧 `mulberry.string.literal` op 已删除；String 不再有专用 dialect 路径。

### File

目标形态：

```cherry
struct FileStorage {
  handle: Ptr<UInt8>
}

comptime File = Ptr<FileStorage>;
```

这里 `handle` 只是 opaque runtime handle。第一阶段不需要暴露 `FILE*` 的真实类型。
`io.open/read/write/close` 是 stdlib API；其中 `open/close` 已改为普通 runtime
wrapper function，`read/write` 通过 temporary `func.call` marker 进入 LowerMulberry，
再改写成 C stdio 的 `fread`/`fwrite`。

当前已经完成：

1. `stdlib/std/io.cherry` 定义了 `FileStorage` 和 `comptime File = Ptr<FileStorage>`。
2. `std.io.open` 现在通过 runtime wrapper `mulberry_file_open` 调用 `fopen`，然后把
   `FILE*` 保存到 `FileStorage.handle`。
3. `std.io.close` 现在通过 runtime wrapper `mulberry_file_close` 读取 `handle` 并调用
   `fclose`。
4. 旧 `!mulberry.file` type 和 `mulberry.file.*` op 已删除；File 不再是 builtin type。

### Tensor

目标形态：

```cherry
comptime TensorStorage<T, Rank> = struct {
  rank: UInt64,
  data: Ptr<T>,
  sizes: Ptr<UInt64>,
  strides: Ptr<UInt64>
};

comptime Tensor<T, Rank> = Ptr<TensorStorage<T, Rank>>;
```

`rank` 是静态 rank 的 runtime mirror，主要服务 runtime/debug/FFI；类型系统仍然负责
保证 rank。`sizes` 和 `strides` 都指向长度为 `Rank` 的连续 `UInt64` 数组。`data`
指向连续 row-major payload。第一版只支持 dense contiguous Tensor，因此 `strides`
可以由 `sizes` 计算出来，但仍保留字段，避免后面要支持 slice/view 时重排 ABI。

当前 `stdlib/std/tensor.cherry` 已经可以直接表达这个形态：

```cherry
comptime TensorStorage<T, Rank: UInt64> = struct {
  rank: UInt64,
  data: Ptr<T>,
  sizes: Ptr<UInt64>,
  strides: Ptr<UInt64>
};

comptime Tensor<T, Rank: UInt64> = Ptr<TensorStorage<T, Rank>>;
```

`Rank` 不是普通 runtime field，而是类型系统信息。Sema 实例化
`TensorStorage<T, Rank>` 时会把 alias origin 保存到 concrete struct type 上，后续
`tensor.view(handle)` 依靠这个 origin 取回 source-level 的 `T` 和 `Rank`，而不是解析
`std_tensor_TensorStorage__Float32__2` 这种 mangled concrete name。

这个 layout 和 MLIR memref descriptor 很接近，但它是 source-level heap object：

```text
Tensor<T, Rank>
  -> Ptr<TensorStorage<T, Rank>>
  -> { rank, data, sizes, strides }
```

也就是说，函数参数、返回值、List element 和 struct field 里保存的都应该是普通
Tensor handle，而不是 `{data, sizes, strides}` by-value descriptor。这样 Tensor 和
List/String/File 的对象模型保持一致：赋值、传参、返回都复制 pointer。

LowerMulberry 当前有一个显式 unwrap 入口：

```text
Ptr<TensorStorage<T, Rank>>
  -> read data/sizes/strides
  -> create memref view
  -> feed cherry_nn / linalg / raw-byte runtime boundary
```

反方向也要明确：

```text
memref allocation result
  -> allocate TensorStorage
  -> store data/sizes/strides
  -> return Ptr<TensorStorage<T, Rank>>
```

当前先不把 source-level `Float32[?, ?]` 整体替换成 heap handle。原因是现有
`cherry_nn -> linalg` 路径依赖 Tensor lowering 成 `memref`。迁移时使用显式入口：

```cherry
var handle: Tensor<Float32, 2> = heap.alloc<tensor.TensorStorage<Float32, 2>>();
var view: Float32[?, ?] = tensor.view(handle);
```

`tensor.view` 是显式 unwrap 入口：Sema 把 `Ptr<TensorStorage<T, Rank>>` 看作旧
`T[?, ...]` TensorType，rank 来自 `Rank`；MLIRGen 生成 `mulberry.tensor.view`；
LowerMulberry 从 `TensorStorage.data/sizes/strides` 重建 memref view，供现有
`cherry_nn` / `linalg` / tensor load-store 路径继续使用。它不拷贝 payload，也不拥有
payload 生命周期。

`tensor.pack(value)` 是反向入口：把当前 Tensor value 包装成
`Ptr<TensorStorage<T, Rank>>`。LowerMulberry 会分配 Boehm heap 上的 storage、
sizes/strides 和 payload，并把原 tensor payload copy 到 heap buffer，保证返回的
handle 可以逃逸。

`readTensor(file, name)` 支持两种 expected type：

```cherry
var x: Float32[?, ?] = readTensor(file, "x");
var xHandle: Tensor<Float32, 2> = readTensor(file, "x");
```

第一种保持现有 memref Tensor value 路径；第二种先读取 payload Tensor value，再通过
`tensor.pack` 打包成 heap Tensor handle。

P5.6/C8 的非目标：

- 不把旧 descriptor surface 改名后伪装成 `TensorStorage`。
- 不让隐藏 descriptor bridge 跨函数边界。
- 不在 MLIRGen 里生成 memref descriptor 或 LLVM descriptor。
- 不破坏当前 `readTensor`、for-loop inference、training safetensors JIT 正向路径。

## P5 迁移顺序

```text
P5.1  固定本文档：标准库对象统一走 Ptr<T> / heap object 方向。
P5.2  已完成第一阶段 StringStorage lowering：String lower 成 Ptr<StringStorage>。
P5.3  已完成第一阶段 FileStorage lowering：File lower 成 Ptr<FileStorage>。
P5.4  设计 TensorStorage；只写 layout 和 unwrap 策略，不立刻替换 memref 正向路径。
P5.5  已完成 string/file/safetensors stdlib runtime boundary 迁移：不再有
      `mulberry.string.*`、`mulberry.file.*` 或 `mulberry.safetensor.read` op。
P5.6  TensorStorage / Tensor 已作为 `std.tensor` 里的 comptime alias 落地；当前只表示
      source-level Ptr<T> / heap object layout，不替换现有 memref Tensor 正向路径。
P5.6.2 `tensor.view(handle)` 已完成 Sema、MLIRGen 和 LowerMulberry 正向路径，能从
       alias origin 恢复旧 TensorType，并从 `TensorStorage` 重建 memref view。
P5.7  已完成：删除公开 Tensor descriptor type/op/lowering/test；保留
      LowerMulberry 内部 Tensor ABI descriptor helper。
P5.8  已完成：`tensor.pack(value)` 提供
      `memref allocation -> Ptr<TensorStorage<T, Rank>>` 的显式 owning pack 入口。
P5.9  已完成：`readTensor(file, name)` 在 expected type 为
      `Tensor<Float32, Rank>` 时返回 heap Tensor handle；expected type 为
      `Float32[?, ...]` 时仍返回 Tensor value。
C8.1.19 已完成边界审计：`mulberry.tensor.*` 当前不能整块删除。旧
      `tensor.alloc/dim/cast/load/store` 仍是 `Float32[...] -> memref/linalg` 正向路径；
      新 `tensor.view/pack` 是 `Ptr<TensorStorage<T, Rank>>` 和 memref-backed Tensor
      value 之间的显式边界。后续要先统一 source-level Tensor object model，再删除
      旧 value-path op。
```

这个顺序是故意保守的。当前推理、training 和 safetensors JIT 已经能跑，不能为了清理
dialect 把正向路径打碎。

## 非目标

- 不在 P5.1 直接删除 `mulberry.tensor.*`。
- 不把旧 descriptor surface 伪装成 Tensor heap handle。
- 不为 String/File/Tensor 分别设计独立 bridge pass。
- 不引入 allocator trait、异常、borrow checker 或复杂 ownership。

当前阶段的目标是让标准库对象模型变简单、可解释、可迁移。
