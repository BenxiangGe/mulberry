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
comptime TensorStorage<T> = struct {
  rank: UInt64,
  data: Ptr<T>,
  sizes: Ptr<UInt64>,
  strides: Ptr<UInt64>
};
```

当前先不把 source-level `Float32[?, ?]` 改成 heap handle。原因是现有
`cherry_nn -> linalg` 路径依赖 Tensor lowering 成 `memref`。真正迁移时，应先定义
`TensorStorage` 的 source-level layout，再让 lowering 入口把 handle 显式 unwrap 成
memref view。

## P5 迁移顺序

```text
P5.1  固定本文档：标准库对象统一走 Ptr<T> / heap object 方向。
P5.2  已完成第一阶段 StringStorage lowering：String lower 成 Ptr<StringStorage>。
P5.3  已完成第一阶段 FileStorage lowering：File lower 成 Ptr<FileStorage>。
P5.4  设计 TensorStorage；只写 layout 和 unwrap 策略，不立刻替换 memref 正向路径。
P5.5  已完成 string/file/safetensors stdlib runtime boundary 迁移：不再有
      `mulberry.string.*`、`mulberry.file.*` 或 `mulberry.safetensor.read` op。
P5.6  在 TensorStorage 准备好后，设计 tensor handle 到 memref view 的 lowering 入口。
P5.7  逐项删除不再需要的 `tensor_desc` 边界。
```

这个顺序是故意保守的。当前推理、training 和 safetensors JIT 已经能跑，不能为了清理
dialect 把正向路径打碎。

## 非目标

- 不在 P5.1 直接删除 `mulberry.tensor.*`。
- 不把 `tensor_desc` 伪装成 Tensor heap handle。
- 不为 String/File/Tensor 分别设计独立 bridge pass。
- 不引入 allocator trait、异常、borrow checker 或复杂 ownership。

当前阶段的目标是让标准库对象模型变简单、可解释、可迁移。
