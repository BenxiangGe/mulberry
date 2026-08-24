# Builtins

## Types

- `()`
- `Bool`
- `UInt8`
- `UInt64`
- `Int64`
- `Integer`
- `Float32`
- `Float64`
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
- `bigint.abs(Integer): Integer`
- `bigint.gcd(Integer, Integer): Integer`
- `bigint.div(Integer, Integer): Result<Integer, bigint.ArithmeticError>`
- `bigint.mod(Integer, Integer): Result<Integer, bigint.ArithmeticError>`
- `bigint.modInverse(Integer, Integer): Result<Integer, bigint.ArithmeticError>`
- `bigint.modPow(Integer, Integer, Integer): Result<Integer, bigint.ArithmeticError>`
- `bigint.toUInt64(Integer): Result<UInt64, bigint.ArithmeticError>`
- `bigint.toUInt8(Integer): Result<UInt8, bigint.ArithmeticError>`
- `bigint.toString(Integer): String`
- `bigint.toHex(Integer): String`
- `string.format<Args...: Show>(comptime pattern: String, args: Args...): String`
- `boolToUInt64(Bool): UInt64`
- `core.toUInt64(UInt8): UInt64`
- `core.toUInt8(UInt64): UInt8`
- `core.toFloat32(UInt64): Float32`
- `core.toFloat64(UInt64): Float64`

四个 `core.to*` conversion 是 compiler builtin source function，分别直接生成
`arith.extui`、`arith.trunci` 和 `arith.uitofp`（Float32/Float64 各一个）。它们不经过
stdlib wrapper 或 runtime C bridge。

`string.format<Args...: Show>()` 是普通 constrained stdlib generic。它要求 comptime
String pattern，支持顺序 `{}` slot 和 `{{` / `}}` literal brace；interpreter 在
specialization 时执行 pattern parser，只将 concrete `Show.toString()` call 与 String concat
留给 runtime。它不是 compiler builtin，compiler 不知道 formatter function name 或 brace
grammar。`Show` 是 special language trait：String、builtin scalar 和 source object 都满足它，
Unit 不参与 formatting。完整 surface 和限制见 [String formatting](StringFormatting.md)。
`Show.toString()` 是纯 capability contract；String 和 builtin scalar 用 concrete `impl Show`
提供 value formatter，source object 由 conditional generic impl 的 explicit method body 调用
object identity formatter，user object 的 inherent `toString()` 优先。Trait default body 机制
仍保留给适用于所有 implementation 的兜底行为。Sema 在 concrete type 上实例化匹配的
conditional method body、Trait default body 或 concrete witness，最终都是普通 direct call，
不属于 compiler builtin registry，也不生成 runtime trait metadata。完整语义见
[Trait](Traits.md)。

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

`Integer` 是 signed arbitrary-precision builtin object。它的 `+`、`-`、`*`、comparison、
bit operation 和 shift 是 operator；`/` 和 `%` 刻意没有 operator surface。`std.bigint` 的
fallible API 返回 `ArithmeticError`，让 `?` 和 `match` 保持显式 error policy；正 modulus 的
`bigint.mod` 返回 canonical residue。`Integer` 有 concrete `Show` implementation，因此
`print(Integer)` 和 `bigint.toString(Integer)` 输出 canonical signed decimal；`bigint.toHex`
显式输出 lowercase `0x` spelling，并按四个 hex digit 分组。

`JsonError` 是 cursor parser 的普通 ADT，覆盖 unexpected end、expected
character/digit/value/separator、unterminated string 和当前不支持的 string escape。
parser 不构造 DOM，只实现 safetensors metadata 所需的 ASCII JSON subset。

`SafetensorsError` 组合 `IoFailure(io.FileError)`、`JsonFailure(json.JsonError)`、
`TensorNotFound(String)` 和 `InvalidTensor(String)`。safetensors public API 不再返回
`found = false` 或默认 metadata，也不保留每次重新读取 header 的 raw File helper。

`File` 当前还没有 `Disposable`/GC finalizer。成功 open 后，如果后续 `?` 提前返回，调用方
可能跳过原计划的 close；现有 workflow 在成功路径显式 close。统一自动清理应由未来的
`Disposable` object lifecycle 解决，不在每个 I/O 调用点复制手写 cleanup control flow。

postfix `?` 是语言控制流，不是按名字查找的 builtin function。它只处理 canonical
`Result<T, E>`。Sema 允许 enclosing function 的 error type 不同：当 operand 是
`Result<Value, SourceError>`、函数返回 `Result<ReturnValue, TargetError>` 时，同类型 error
直接传播；不同类型必须有 canonical `From<SourceError> for TargetError` concrete witness。
Sema 把 source/target error type 与已解析的 converter symbol 记录在 `TryExpr`，缺少 conversion
会报错；它不把 `TryExpr` 改写为 source-level `match`。`std.convert.From<T>` 同时支持显式
`Target.from(error)` conversion。

这一 Sema/AST contract 与 ER2.6 的 high-level IR contract 已完成。`mulberry_core.result.try`
明确保存 `source_error`、`target_error` 与 optional converter symbol；无 converter 时两种 error
storage type 必须相同，有 converter 时 verifier 检查它引用一个精确的
`SourceError -> TargetError` function。MLIRGen 只使用 Sema 已选择的 symbol，不重新做 Trait
lookup。

ER2.7 已完成：lowered Err path 解包 source error，在存在 converter 时调用一次并构造
`Err(TargetError)`；Ok path 不调用 converter，同类型 fast path 继续完整工作。跨 error type 的
`?` 现在可以进入 lowered CFG。ER2.8 已完成 generic JIT 的 exact-once 验证，并覆盖现有的
Unit/object payload、loop 和 match 矩阵。高层 `mulberry_core.result.try` 会在 lowering 中展开为 Ok continuation
和 Err direct return；不提供通用 `Try` trait。object payload 继承 operand expression 的 mutability：
直接 call 的 Result 可以通过 `?` 产生 mutable object，而 readonly Result variable 解包后仍为 readonly。

ER2.9 已完成真实 safetensors workload 迁移：`mapIoError()` 和 `mapJsonError()` 已删除，普通
I/O 与 JSON parser 调用直接使用 `?`；cleanup 分支继续保留 primary error precedence。

Unit 可作为 Trait 的 type-level argument。因此 `impl From<()> for TargetError` 采用零参数
`fn from(): Self`：Unit 不产生 runtime value，而 `?` 的 Err path 会直接调用这个 converter。

ER2.12 已完成 structured-loop propagation：safetensors 的 `while`/`for` body 可以直接使用
`?`。第一次 core lowering 只把 `ResultTryOp` 的 operand/result 类型转换到 storage 类型；
ownership deallocation 在结构化 loop 仍存在时运行，随后 `lower-result-try` 才把实际的
`ResultTryOp` 和相关 SCF ancestor 展开为 CFG。含 `ResultTryOp` 的模块当前跳过 ownership
deallocation，避免函数级 early return 绕过 cleanup 导致 Tensor double free；Tensor storage
继续依赖 GC/manual-deallocation fallback。该边界将在未来有显式 cleanup edge 后收紧。

ER2.13 已完成 ownership 边界审计。不能简单把 fallback 缩小到 function：NN workload 的
Tensor ownership 会跨函数传递，含 `?` 的函数与其它函数可能共享同一个 storage；恢复其它
函数的自动 deallocation 已实际触发 double free。也不能给 `ResultTryOp` 伪造
`RegionBranchOpInterface`，因为它没有 region，且该 interface 无法表达从 loop 深处跳到函数
返回前必须执行的 lexical cleanup。当前模块级 fallback 是正确性边界，不是最终设计。
后续应让 Err edge 显式携带 ownership，经过统一 cleanup dispatcher 后再返回；详见设计文档
第 56.7 节。

## Comptime Intrinsics

- `#compileError(String)`：只在已执行的 comptime path 中接受 static String，终止当前
  specialization 并产生 compiler diagnostic。

`#compileError()` 不是 formatter-specific API；generic stdlib code 在无法满足 static
contract 时都可以使用它。当前 comptime interpreter 的 executable statement、residualization
和明确 non-goals 见 [Comptime interpreter](ComptimeInterpreter.md)。

## Comptime Reflection

- `#typeInfo(T)`：取得 source type 的 comptime Type。
- `#typeOf(value)`：取得 value expression 的 comptime Type。
- `const info = #typeInfo(T)`：根据 initializer 自动建立 comptime local binding。

reflection query 只在 Sema 求值，不是 runtime function，也不产生 MLIR call。完整
query 列表和边界见 [编译期反射](Reflection.md)。comptime interpreter 可以执行 static
local、ordinary static `if` / `while`、source helper 和 parameter pack，同时将依赖 runtime
value 的 ordinary expression residualize。返回 comptime Type 的 reflection expression 也可以
直接用于 local、function signature 和 generic alias 的 type position。

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
- `Int64 + Int64 : Int64`
- `Int64 - Int64 : Int64`
- `Int64 * Int64 : Int64`
- `Int64 / Int64 : Int64`
- `Int64 % Int64 : Int64`
- `Int64 << UInt64 : Int64`
- `Int64 >> UInt64 : Int64`
- `Int64 & Int64 : Int64`
- `Int64 | Int64 : Int64`
- `Int64 ^ Int64 : Int64`
- `Int64 < Int64 : Bool`
- `Int64 <= Int64 : Bool`
- `Int64 > Int64 : Bool`
- `Int64 >= Int64 : Bool`
- `Int64 == Int64 : Bool`
- `Int64 != Int64 : Bool`
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
- `Float64 + Float64 : Float64`
- `Float64 - Float64 : Float64`
- `Float64 * Float64 : Float64`
- `Float64 / Float64 : Float64`
- `Float64 < Float64 : Bool`
- `Float64 <= Float64 : Bool`
- `Float64 > Float64 : Bool`
- `Float64 >= Float64 : Bool`
- `Float64 == Float64 : Bool`
- `Float64 != Float64 : Bool`
- `Bool == Bool : Bool`
- `Bool != Bool : Bool`
- `Bool and Bool : Bool`
- `Bool or Bool : Bool`

非负 integer literal 默认是 `UInt64`，负 literal 默认是 `Int64`；float literal 默认是
`Float32`。显式 expected type 可以把 literal 定型为 `Int64` 或 `Float64`。两个不同的
fixed-width numeric variable 不会自动混算，只有 literal 会根据另一侧已知类型进行上下文
定型；fixed-width integer 到 `Integer` 是唯一的 implicit numeric widening。
