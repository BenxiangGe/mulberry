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
- `boolToUInt64(Bool): UInt64`
- `core.toUInt64(UInt8): UInt64`
- `core.toUInt8(UInt64): UInt8`
- `core.toFloat32(UInt64): Float32`

三个 `core.to*` conversion 是 compiler builtin source function，分别直接生成
`arith.extui`、`arith.trunci` 和 `arith.uitofp`。它们不经过 stdlib wrapper 或 runtime
C bridge。

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
