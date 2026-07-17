# Builtins

## Types

- `()`
- `Bool`
- `UInt8`
- `UInt64`
- `Float32`
- `Char`

普通 object 类型由语言/stdlib 提供，例如 `String`、`Array<T, N>`、`List<T>`、
`Tensor<T>`、`Result<T, E>`、用户 `struct` 和用户 `data`。它们按 object reference
语义传递；底层 storage 不是用户语法。

## Functions  

- `io.print<T>(T): ()`
- `io.println<T>(T): ()`
- `io.open(String, String): Result<File, io.FileError>`
- `io.read<T>(File, mut Tensor<T>): Result<UInt64, io.FileError>`
- `io.readExact<T>(File, mut Tensor<T>): Result<UInt64, io.FileError>`
- `io.write<T>(File, Tensor<T>): Result<UInt64, io.FileError>`
- `io.seek(File, UInt64): Result<UInt64, io.FileError>`
- `io.tell(File): Result<UInt64, io.FileError>`
- `io.close(File): Result<(), io.FileError>`
- `json.Parser.expectChar(Char): Result<(), json.JsonError>`
- `json.Parser.stringEquals(String): Result<Bool, json.JsonError>`
- `json.Parser.parseIntegerArray(): Result<List<UInt64>, json.JsonError>`
- `json.Parser.skipValue(): Result<(), json.JsonError>`
- `safetensors.open(String): Result<TensorFile, safetensors.SafetensorsError>`
- `safetensors.find(TensorFile, String): Result<TensorInfo, safetensors.SafetensorsError>`
- `safetensors.read(TensorFile, String): Result<Tensor<Float32>, safetensors.SafetensorsError>`
- `safetensors.close(TensorFile): Result<(), safetensors.SafetensorsError>`
- `string.formatValue<T>(T): String`
- `boolToUInt64(Bool): UInt64`
- `core.toUInt64(UInt8): UInt64`
- `core.toUInt8(UInt64): UInt8`
- `core.toFloat32(UInt64): Float32`

三个 `core.to*` conversion 是 compiler builtin source function，分别直接生成
`arith.extui`、`arith.trunci` 和 `arith.uitofp`。它们不经过 stdlib wrapper 或 runtime
C bridge。

`string.formatValue<T>()` 是普通 reflection-based stdlib generic。字符串插值和
`String + value` 使用它静态选择 scalar formatter、`toString()` 或 object identity；
这些选择不属于 compiler builtin registry。

`tensor.from<A>()` 也是普通 reflection-based stdlib generic。它在 specialization 中
递归构造 shape、按 row-major 顺序 flatten Array，并建立 `Tensor<T>` object；Sema 和
MLIRGen 不再为 `std.tensor.from` 注册 typecheck/codegen handler。Tensor-only early
disposal 仍将它识别为 owning constructor；这属于生命周期策略，不参与 Tensor 构造。

`Result<T, E>`、`Ok`、`Err`、`FileError` 和 `io.*` 都是普通 stdlib declaration，
不属于 builtin registry。runtime 只返回 byte count 或 Bool 状态；Result 构造和分派
完全使用普通 ADT 与 `match`。`FileError` 当前覆盖 open/read/unexpected-end/write/seek/
tell/close。`read()` 把正常 EOF 短读表示为 `Ok(actualBytes)`；要求填满 buffer 的格式读取
应使用 `readExact()`，短读会得到 `UnexpectedEnd(expected, actual)`。

`close()` 成功时返回 `Ok(())`。generic ADT 保留 Unit payload 的完整语义，但 Unit 不占
runtime storage，也不产生 dummy SSA value。

`JsonError` 是 cursor parser 的普通 ADT，覆盖 unexpected end、expected
character/digit/value/separator、unterminated string 和当前不支持的 string escape。
parser 不构造 DOM，只实现 safetensors metadata 所需的 ASCII JSON subset。

`SafetensorsError` 组合 `IoFailure(io.FileError)`、`JsonFailure(json.JsonError)`、
`TensorNotFound(String)` 和 `InvalidTensor(String)`。safetensors public API 不再返回
`found = false` 或默认 metadata，也不保留每次重新读取 header 的 raw File helper。

`File` 当前还没有 `Disposable`/GC finalizer。成功 open 后，如果后续 `?` 提前返回，调用方
可能跳过原计划的 close；现有 workflow 在成功路径显式 close。统一自动清理应由未来的
`Disposable` object lifecycle 解决，不在每个 I/O 调用点复制手写 cleanup control flow。

postfix `?` 是语言控制流，不是按名字查找的 builtin function。第一版只处理 canonical
`Result<T, E>`，并要求当前函数返回 error type 相同的 `Result<U, E>`。高层
`mulberry_core.result.try` 会在 lowering 中展开为 Ok continuation 和 Err direct
return；当前不提供通用 `Try` trait 或 error conversion。object payload 继承 operand
expression 的 mutability：直接 call 的 Result 可以通过 `?` 产生 mutable object，而
readonly Result variable 解包后仍为 readonly。

## Comptime Reflection

- `typeInfo(T)`：取得 source type 的 comptime Type。
- `typeOf(value)`：取得 value expression 的 comptime Type。
- `const info = typeInfo(T)`：根据 initializer 自动建立 comptime local binding。

reflection query 只在 Sema 求值，不是 runtime function，也不产生 MLIR call。完整
query 列表和边界见 [编译期反射](Reflection.md)。普通 `if` 的 condition 如果能在
Sema 得到 comptime Bool，就只分析并生成选中的 block；否则仍是 runtime `if`。
返回 comptime Type 的 reflection expression 也可以直接用于 local、function
signature 和 generic alias 的 type position。

## Operators
- `UInt64 + UInt64 : UInt64`
- `UInt64 - UInt64 : UInt64`
- `UInt64 * UInt64 : UInt64`
- `UInt64 / UInt64 : UInt64`
- `UInt64 % UInt64 : UInt64`
- `UInt64 < UInt64 : Bool`
- `UInt64 <= UInt64 : Bool`
- `UInt64 > UInt64 : Bool`
- `UInt64 >= UInt64 : Bool`
- `UInt64 == UInt64 : Bool`
- `UInt64 != UInt64 : Bool`
- `Float32 + Float32 : Float32`
- `Float32 - Float32 : Float32`
- `Float32 * Float32 : Float32`
- `Float32 / Float32 : Float32`
- `Float32 < Float32 : Bool`
- `Float32 <= Float32 : Bool`
- `Float32 > Float32 : Bool`
- `Float32 >= Float32 : Bool`
- `Float32 == Float32 : Bool`
- `Float32 != Float32 : Bool`
- `Bool == Bool : Bool`
- `Bool != Bool : Bool`
- `Bool and Bool : Bool`
- `Bool or Bool : Bool`
