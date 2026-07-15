# Mulberry Grammar

本文档描述当前用户可见 Mulberry source 语法。compiler/stdlib internal 语法不在这里列出。

## Lexical Structure

```text
identifier                -> identifier-head identifier-character*
identifier-head           -> `A` ... `Z` | `a` ... `z` | `_`
identifier-character      -> identifier-head | `0` ... `9`

unit-literal              -> `(` `)`
boolean-literal           -> `true` | `false`
integer-literal           -> decimal-digit+
float-literal             -> decimal-digit+ `.` decimal-digit+
string-literal            -> `"` string-segment* `"`
string-segment            -> string-character
string-segment            -> `\\n` | `\\t` | `\\"` | `\\\\` | `\\{` | `\\}`
string-segment            -> string-interpolation
string-interpolation      -> `{$` interpolation-access `}`
interpolation-access      -> identifier interpolation-suffix*
interpolation-suffix      -> `.` identifier
interpolation-suffix      -> `[` interpolation-index-list `]`
interpolation-index-list  -> interpolation-index (`,` interpolation-index)* `,`?
interpolation-index       -> integer-literal | interpolation-access
char-literal              -> `'` ... `'`
```

只有未转义的 `{$` 开始插值。插值是受限 access grammar，不接受 binary expression、
function call 或 method call。普通 `{` / `}` 是文本；`\{`、`\}` 和 `\\` 分别表示
literal `{`、`}` 和 `\`。

## Module

```text
module                    -> package-declaration? declaration*
package-declaration       -> `package` qualified-name `;`
qualified-name            -> identifier (`.` identifier)*
```

## Declarations

```text
declaration               -> import-declaration
declaration               -> function-declaration
declaration               -> extern-function-declaration
declaration               -> struct-declaration
declaration               -> comptime-type-alias-declaration

import-declaration        -> `import` qualified-name `;`
extern-function-declaration
                           -> `extern` function-prototype `;`
function-declaration      -> function-prototype block
function-prototype        -> `fn` function-name comptime-parameter-clause?
                              function-signature
function-name             -> identifier
function-signature        -> `(` parameter-list? `)` `:` type
parameter-list            -> parameter (`,` parameter)* `,`?
parameter                 -> `mut`? identifier `:` type

struct-declaration        -> `struct` identifier `{` struct-member-list? `}`
struct-member-list        -> struct-member (`,` struct-member)* `,`?
struct-member             -> field-declaration
struct-member             -> method-declaration
field-declaration         -> identifier `:` type
method-declaration        -> `pub`? function-declaration

comptime-type-alias-declaration
                           -> `comptime` identifier comptime-parameter-clause?
                              `=` comptime-alias-body `;`
comptime-alias-body       -> type
comptime-alias-body       -> `struct` `{` struct-member-list? `}`
```

函数参数和 method receiver 默认是 readonly object reference。需要在 callee 里
修改 object 时，参数前写 `mut`：

```mulberry
fn read(xs: List<UInt64>): UInt64 {
  return xs.size();
}

fn append(mut xs: List<UInt64>, value: UInt64): UInt64 {
  return xs.push(value);
}
```

## Comptime Parameters

```text
comptime-parameter-clause -> `<` comptime-parameter-list `>`
comptime-parameter-list   -> comptime-parameter (`,` comptime-parameter)*
comptime-parameter        -> identifier
comptime-parameter        -> identifier `:` `UInt64`

generic-argument-clause   -> `<` generic-argument-list `>`
generic-argument-list     -> generic-argument (`,` generic-argument)*
generic-argument          -> type
generic-argument          -> integer-literal
```

没有 `: UInt64` 的 comptime 参数默认是类型参数：

```mulberry
comptime Pair<T, U> = struct { first: T, second: U };
comptime Buffer<T, N: UInt64> = Array<T, N>;
```

## Comptime Reflection

`typeInfo(T)` 的参数是 type，取得该类型的 semantic type；`typeOf(value)` 的参数是
value expression，取得表达式的 semantic type。二者只产生 comptime Type，不生成
runtime object。

当前 reflection query 和求值规则见 [Reflection](Reflection.md)。

comptime value 使用普通 `const` 保存，例如 `const info = typeInfo(T);`。binding 是否
只存在于编译期由 initializer 自动决定，不增加 `comptime const` 语法。

## Types

```text
type                      -> `(` `)`
type                      -> identifier
type                      -> identifier generic-argument-clause
type                      -> computed-type-expression
type                      -> type array-type-suffix
computed-type-expression  -> call-expression postfix-expression*
array-type-suffix         -> `[` integer-literal `]`
```

`T[N]` 是固定长度 `Array<T, N>` 的 source sugar。多维或动态 ranked tensor spelling
已删除；需要 numeric buffer 时使用 `Tensor<T>`。

`computed-type-expression` 必须在 Sema 得到 comptime Type。例如 generic function
实例化后，局部声明或函数签名可以从 Array type 计算 Tensor 的 element type：

```mulberry
var result: Tensor<typeInfo(A).arrayLeafElementType()> = tensor.from(value);
```

该 expression 只存在于 AST/Sema，不进入 MLIR。普通 runtime value 或 Bool/UInt64/String
comptime value 都不能用作 type。

generic inference 不会仅凭 reflection query 的结果猜测完整 source type。对于
`Tensor<typeInfo(A).arrayLeafElementType()>`，单独一个 leaf type 无法确定 `A` 的
rank 和 shape。普通 value 参数必须提供完整 `A`；Array literal 是唯一的当前特例：
literal 已经提供 rank/shape，expected return type 再提供 leaf type，因此两者可以共同
确定完整 Array type。Sema 随后求出 concrete 参数和返回类型并进行普通 type check。

## Expressions

```text
expression                -> assignment-expression
assignment-expression     -> binary-expression
assignment-expression     -> lvalue `=` expression
lvalue                    -> identifier
lvalue                    -> postfix-expression `.` identifier
lvalue                    -> postfix-expression `[` expression-list `]`

binary-expression         -> postfix-expression
binary-expression         -> binary-expression operator binary-expression
operator                  -> `+` | `-` | `*` | `/` | `%`
operator                  -> `and` | `or`
operator                  -> `==` | `!=` | `<` | `<=` | `>` | `>=`

postfix-expression        -> primary-expression
postfix-expression        -> postfix-expression `.` identifier
postfix-expression        -> postfix-expression `.` identifier call-argument-clause
postfix-expression        -> postfix-expression `[` expression-list `]`

primary-expression        -> literal-expression
primary-expression        -> identifier
primary-expression        -> call-expression
primary-expression        -> struct-literal-expression
primary-expression        -> array-literal-expression
primary-expression        -> block

literal-expression        -> unit-literal
literal-expression        -> boolean-literal
literal-expression        -> integer-literal
literal-expression        -> float-literal
literal-expression        -> string-literal
literal-expression        -> char-literal

call-expression           -> qualified-name call-argument-clause
call-argument-clause      -> `(` expression-list? `)`
expression-list           -> expression (`,` expression)* `,`?

struct-literal-expression -> type `{` expression-list? `}`
array-literal-expression  -> `[` expression-list? `]`
```

Method call 不是 OOP dispatch。它只是 receiver-first 的普通函数调用糖，由 Sema 解析：

```text
receiver.method(args...) -> method(receiver, args...)
```

## Blocks And Statements

```text
block                     -> `{` statement* `}`

statement                 -> variable-declaration `;`
statement                 -> expression `;`
statement                 -> if-statement
statement                 -> while-statement
statement                 -> for-statement
statement                 -> `break` `;`?
statement                 -> `continue` `;`?
statement                 -> return-statement

variable-declaration      -> `var` identifier (`:` type)? `=` expression
variable-declaration      -> `const` identifier (`:` type)? `=` expression
return-statement          -> `return` expression? `;`
```

省略 `: type` 时，局部变量直接采用 initializer 的 semantic type。这个规则只适用于
local variable declaration；函数参数、函数返回类型和 struct field 仍必须显式声明。
如果 initializer 本身不能确定完整类型，程序员需要保留类型注解。

block 不产生 value。函数返回使用 `return` statement。

## Control Flow

```text
if-statement              -> `if` condition block
if-statement              -> `if` condition block `else` block
while-statement           -> `while` condition block
for-statement             -> `for` identifier `in` expression `..` expression block
condition                 -> expression
```

`if` 只有一种语法。如果 condition 能在 Sema 得到 comptime Bool，只分析并生成选中的
block，未选 block 不参与当前 specialization 的类型检查；如果 condition 是 runtime
Bool，则按普通控制流分析两个 block，并在 MLIR 中生成 `scf.if`。

示例：

```mulberry
if value > 0 {
  io.println(value);
}

while index < values.size() {
  index = index + 1;
}

for i in 0 .. values.size() {
  io.println(values[i]);
}
```
