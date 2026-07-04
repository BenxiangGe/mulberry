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
- 目标 source 语义采用 Java-style reference object model：除了 scalar，其它
  source-level object 赋值、传参和返回都复制 reference。
- `Ptr<T>` 是 stdlib/runtime/FFI 使用的底层 typed address；普通用户应该通过 object
  API 使用 `String`、`File`、`Array`、`List`、`Tensor` 和用户 `struct`。
- Dialect 只保留当前正向路径必须知道的高层边界，不继续扩大对象语义。
- ABI descriptor 可以存在，但只能是 lowering/runtime helper，不能伪装成 source-level
  object handle。
- 每次迁移一个对象类型，先保证现有 JIT 正向路径不退化。
- 遇到缺失能力时补语言能力，不在 lowering 里加 object-specific workaround。

## 目标对象形态

### List

当前已完成：

```mulberry
comptime List<T> = struct {
  length: UInt64,
  capacity: UInt64,
  data: Ptr<T>
};
```

`List<T>` 是标准库对象，不再有 `mulberry.list.*` type/op/lowering。目标 source
语义下，`List<T>` 变量保存 object reference；`length/capacity/data` 是 heap object
layout 字段，不是用户赋值时被复制的 by-value header。

### String

当前形态：

```mulberry
struct String {
  length: UInt64,
  data: Ptr<UInt8>
}
```

如果后续需要可变字符串或 byte buffer，再引入 `capacity`：

```mulberry
struct ByteBufferStorage {
  length: UInt64,
  capacity: UInt64,
  data: Ptr<UInt8>
}
```

`String` 现在是 stdlib object，不再是 builtin，也不再是旧 pointer-storage alias。

- `length` 是 Mulberry 字符串长度，不包含结尾 NUL。
- `data` 指向 byte storage。第一版继续保证 literal data 以 NUL 结尾，方便 C
  runtime API 复用同一个 pointer。
- string literal 的 bytes 直接用 `heap.alloc<UInt8>(length + 1)` 分配并逐 byte 填充。
- string literal 构造一个普通 `String { length, data }` record value；只有 byte
  storage 在 Boehm heap 上。

也就是说，literal lowering 的目标形态是：

```text
"abc"
  -> heap UInt8[4] = "abc\0"
  -> String { length = 3, data = bytes }
```

不再经过 string-specific op 的原因是：`String` 已经是普通 stdlib record value。
literal materialization 可以直接复用 `heap.alloc`、`ptr.index`、`record.get_field`
和 `store` 这条通用对象路径。

runtime 边界现在直接接收 by-value `String` wrapper，由 helper 或 lowering 取出
`data` 和 `length`。如果只是调用 C `fopen` 这种需要 NUL-terminated string 的 API，
可以只用 `data`；如果后续实现非 C API，则可以同时使用 `length`。

String 的调用链现在是：

```text
String literal
  -> Parser: StringLiteralExpr
  -> Sema: stdlib struct String
  -> TypeConverter: String -> record { length, data }
  -> MLIRGen: heap.alloc bytes + local String record stores
  -> LowerMulberry: generic heap/ptr/record/store lowering
```

`io.open(path, mode)` 和 `safetensors.readTensor(file, name)` 的 String 参数也走这条链。
lowering 不再为 String 保留专用 helper；`String` 和 `File` 现在通过普通 record/ptr
路径 lower 到 runtime boundary。目标 reference model 会把这些 record storage 放在
heap object 后面，source 层只传递 reference。

当前已经完成的实现：

1. `stdlib/std/string.mulberry` 定义了 `struct String { length, data }`。
2. `MLIRGen` 直接为 string literal 创建 heap byte buffer，再构造 `String` record，
   并写入 `length` 和 `data` 字段。后续 reference model 会把这一步变成 heap object
   construction。
3. lowering 通过通用 record/ptr 路径把 `String` 的 `data` 和 `File` 的 `handle`
   传给 runtime helper。
4. 旧 `mulberry.string.literal` op 已删除；String 不再有专用 dialect 路径。

### File

目标形态：

```mulberry
struct File {
  handle: Ptr<UInt8>
}
```

这里 `handle` 只是 opaque runtime handle。第一阶段不需要暴露 `FILE*` 的真实类型。
`io.open/read/write/close` 是 stdlib API；其中 `open/close` 已改为普通 runtime
wrapper function。`read/write` 现在也是 stdlib generic wrapper：它们从
`Tensor<T>` header 取出 `data/numel`，用 `ptr.asUInt8` 得到 raw byte pointer，
再调用 `mulberry_file_read` / `mulberry_file_write` runtime wrapper。具体 C stdio
的 `fread`/`fwrite` 细节留在 runtime 里。

当前已经完成：

1. `stdlib/std/io.mulberry` 定义了 `struct File { handle: Ptr<UInt8> }`。
2. `std.io.open` 现在通过 runtime wrapper `mulberry_file_open` 调用 `fopen`，然后把
   raw `FILE*` handle 包装成 `File { handle }`。
3. `std.io.close` 现在读取 `file.handle`，再通过 runtime wrapper
   `mulberry_file_close` 调用 `fclose`。
4. 旧 `!mulberry.file` type 和 `mulberry.file.*` op 已删除；File 不再是 builtin type。

### Array / List / Tensor 分层

下一阶段的 source-level 规则：

- 普通 `[]` literal 默认是 `Array`。它是 Mulberry 的通用语言数组。
- `List<T>` 是 growable container，适合 push/pop 和动态长度数据，不再作为普通数组
  literal 的默认目标。
- `Tensor<T>` 类似 NumPy `ndarray`，是显式 numeric buffer / bufferization 对象。
  只有在需要 dense numeric storage、MLIR memref/linalg 或 `mulberry.nn` interop 时使用。

也就是说：

```mulberry
var data = [[1.0, 2.0], [3.0, 4.0]];        // Array
var tensor: Tensor<Float32> = tensor.from(data);
var zeros: Tensor<Float32> = tensor.zeros([30, 784]);
```

这里 `tensor.zeros([30, 784])` 里的 `[30, 784]` 也应该先是普通 Array。只有运行时才知道
shape 时，才需要用 `List<UInt64>` 表达动态 shape。

当前实现已经收掉旧的 expected-type magic：普通 `[]` 总是先成为 `Array`。如果要构造
`Tensor<T>`，使用 `tensor.from(array)` 或 `tensor.zeros(shape)`；如果要构造
`List<T>`，使用 `list.from(array)` 或 `list.withCapacity(0)`。
`tensor.from(array)` 接受 fixed-size `Array` value 或直接传入的 array literal，按
row-major 顺序递归 flatten nested Array，构造 `Tensor<T>` header，并填好 runtime
rank、shape、numel 和 strides metadata。

### Tensor

目标形态：

```mulberry
comptime Tensor<T> = struct {
  data: Ptr<T>,
  rank: UInt64,
  numel: UInt64,
  sizes: List<UInt64>,
  strides: List<UInt64>
};
```

这是 NumPy `ndarray` 风格的 high-level Tensor header。`data` 指向连续 row-major
payload；`rank` 是 runtime rank；`numel` 是元素总数；`sizes` 和 `strides` 是普通
`List<UInt64>` metadata header。第一版只支持 dense contiguous Tensor，因此
`strides` 可以由 `sizes` 计算出来，但仍保留字段，避免后面支持 slice/view 时重排
source-level object layout。用户层建议通过 `ndim()`、`numel()`、`shape()`、
`dim(i)`、`stride(i)` 访问这些 metadata，而不是直接碰字段。

当前 `stdlib/std/tensor.mulberry` 已经可以直接表达这个形态：

```mulberry
comptime Tensor<T> = struct {
  data: Ptr<T>,
  rank: UInt64,
  numel: UInt64,
  sizes: List<UInt64>,
  strides: List<UInt64>,

  pub fn numel(self: Ptr<Tensor<T>>): UInt64 {
    self.numel
  }
};
```

`Tensor<T>` 不再有 compile-time `Rank` 参数。rank/shape 是 runtime metadata。
`Tensor<T>` 的目标 source 语义是 object reference；源码不再暴露 ranked tensor
view/pack bridge。

也就是说，函数参数、返回值、List element 和 struct field 里保存 Tensor reference。
赋值、传参、返回复制 reference；payload 和 metadata list 的 underlying buffer 由
Tensor object 管理。显式复制数据应通过后续 `clone()` / `copy()` 这类 API 表达。

多维/动态 `T[...]` source surface 已删除。一维静态 `Float32[N]` 已经是普通
fixed-size Array。新代码显式构造 `Tensor<T>` header。推荐写法是 `tensor.from(array)` 或
`tensor.zeros(shape)`：

```mulberry
var data = [[1.0, 2.0], [3.0, 4.0]];
var tensor2d: Tensor<Float32> = tensor.from(data);
var zeros: Tensor<Float32> = tensor.zeros([2, 2]);
var value: Float32 = tensor2d[1, 1];
```

也可以直接把 literal 作为 `tensor.from` 的参数：

```mulberry
var tensor2d: Tensor<Float32> =
    tensor.from([[1.0, 2.0], [3.0, 4.0]]);
var zeros: Tensor<Float32> = tensor.zeros([2, 2]);
```

`Tensor<T>` header 可以直接用 `tensor[i]` 或 `tensor[i, j]` 访问元素。MLIRGen
根据 header 里的 `strides` 计算 row-major linear offset，再通过 `data` 指针读写。

`safetensors.readTensor(file, name)` 直接返回 `Tensor<Float32>`：

```mulberry
var xTensor: Tensor<Float32> = safetensors.readTensor(file, "x");
var first: Float32 = xTensor[0, 0];
```

`Tensor<T>` object 承载 shape、rank、numel、sizes 和 strides。
普通元素访问直接用 index 语法；需要 metadata 时，用 `ndim()`、`numel()`、
`shape()`、`dim(i)`、`stride(i)`。

P5.6/C8 的非目标：

- 不把旧 descriptor surface 改名后伪装成 `Tensor<T>`。
- 不让隐藏 descriptor 边界跨函数边界。
- 不在 MLIRGen 里生成 memref descriptor 或 LLVM descriptor。
- 不破坏当前 `safetensors.readTensor`、core Tensor object 和 bundled `mulberry.nn`
  package 正向路径；真正独立的 shared library / FFI package 后续再拆。

## P5 迁移顺序

```text
P5.1  固定本文档：标准库对象统一走 Ptr<T> / heap object 方向。
P5.2  已完成 String value wrapper lowering：String lower 成 `{length, data}` record。
P5.3  已完成 File value wrapper lowering：File lower 成 `{handle}` record。
P5.4  设计 Tensor header；只写 layout 和 unwrap 策略，不立刻替换 memref 正向路径。
P5.5  已完成 string/file/safetensors stdlib runtime boundary 迁移：不再有
      `mulberry.string.*`、`mulberry.file.*` 或 `mulberry.safetensor.read` op。
P5.6  Tensor 已作为 `std.tensor` 里的 comptime alias 落地；当前表示 source-level
      ndarray-style header，不替换现有 memref Tensor 正向路径。
P5.6.2 已废弃：source-level ranked tensor bridge 已删除。
P5.7  已完成：删除公开 Tensor descriptor type/op/lowering/test；保留
      LowerMulberry 内部 Tensor ABI descriptor helper。
P5.8  已废弃：source-level ranked tensor bridge 已删除。
P5.9  已完成：`safetensors.readTensor(file, name)` 统一返回 `Tensor<Float32>` header；
      后续 NN/linalg interop 应直接消费 Tensor header。
T1.4  已完成：新代码优先使用 `tensor.from(array)` 和 `tensor.zeros([shape])`；
      source-level ranked tensor bridge 已删除。
C8.1.19 已完成边界审计：`mulberry_core.tensor.*` 后续只作为 internal Tensor lowering
      机制逐步收缩，不再有对应源码语法。
```

这个顺序是故意保守的。当前 core Tensor、file IO 和 safetensors object path 是后续
外部 NN package 的地基，不能为了清理 dialect 把它们打碎。

## 非目标

- 不在 P5.1 直接删除 `mulberry_core.tensor.*`。
- 不把旧 descriptor surface 伪装成 Tensor heap handle。
- 不为 String/File/Tensor 分别设计独立 lowering pass。
- 不引入 allocator trait、异常、borrow checker 或复杂 ownership。

当前阶段的目标是让标准库对象模型变简单、可解释、可迁移。
