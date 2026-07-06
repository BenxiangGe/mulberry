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
string-literal            -> `"` ... `"`
char-literal              -> `'` ... `'`
```

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
parameter                 -> `const`? identifier `:` type

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

## Types

```text
type                      -> `(` `)`
type                      -> identifier
type                      -> identifier generic-argument-clause
type                      -> type array-type-suffix
array-type-suffix         -> `[` integer-literal `]`
```

`T[N]` 是固定长度 `Array<T, N>` 的 source sugar。多维或动态 ranked tensor spelling
已删除；需要 numeric buffer 时使用 `Tensor<T>`。

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
block                     -> `{` statement* expression? `}`

statement                 -> variable-declaration `;`
statement                 -> expression `;`
statement                 -> if-statement
statement                 -> while-statement
statement                 -> for-statement
statement                 -> `break` `;`?
statement                 -> `continue` `;`?
statement                 -> return-statement

variable-declaration      -> `var` identifier `:` type `=` expression
variable-declaration      -> `const` identifier `:` type `=` expression
return-statement          -> `return` expression? `;`
```

block 的最后一个 expression 可以省略；省略时 block 的值是 `()`。函数返回推荐使用
`return` statement；函数 body 的最后一个 expression 作为返回值仍暂时兼容。

## Control Flow

```text
if-statement              -> `if` condition block
if-statement              -> `if` condition block `else` block
while-statement           -> `while` condition block
for-statement             -> `for` identifier `in` expression `..` expression block
condition                 -> expression
```

示例：

```mulberry
if value > 0 {
  io.print(value);
}

while index < values.size() {
  index = index + 1;
}

for i in 0 .. values.size() {
  io.print(values[i]);
}
```
