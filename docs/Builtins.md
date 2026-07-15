# Builtins

## Types

- `()`
- `Bool`
- `UInt8`
- `UInt64`
- `Float32`
- `Char`

普通 object 类型由语言/stdlib 提供，例如 `String`、`Array<T, N>`、`List<T>`、
`Tensor<T>` 和用户 `struct`。它们按 object reference 语义传递；底层 storage 不是
用户语法。

## Functions  

- `io.print<T>(T): ()`
- `io.println<T>(T): ()`
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
