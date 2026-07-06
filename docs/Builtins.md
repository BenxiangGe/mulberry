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

- `io.print(...)`
- `boolToUInt64(Bool): UInt64`
- `core.toUInt64(UInt8): UInt64`
- `core.toUInt8(UInt64): UInt8`
- `core.toFloat32(UInt64): Float32`

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
