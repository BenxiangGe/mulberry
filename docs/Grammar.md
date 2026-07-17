# Mulberry Grammar

本文档描述当前用户可见 Mulberry source 语法。compiler/stdlib internal 语法不在这里列出。

## Lexical Structure

```text
identifier                -> identifier-head identifier-character*
identifier-head           -> `A` ... `Z` | `a` ... `z` | `_`
identifier-character      -> identifier-head | `0` ... `9`
uppercase-identifier      -> (`A` ... `Z`) identifier-character*

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

点式调用的首段如果同时是局部变量和 import package alias，局部变量按词法作用域
优先。例如参数名为 `result` 时，`result.size()` 仍是该对象的 method call；只有当前
作用域没有同名变量时，`result.foo()` 才会按 package alias 解析。

## Declarations

```text
declaration               -> import-declaration
declaration               -> function-declaration
declaration               -> extern-function-declaration
declaration               -> struct-declaration
declaration               -> trait-declaration
declaration               -> impl-declaration
declaration               -> data-declaration
declaration               -> comptime-type-alias-declaration

import-declaration        -> `import` qualified-name `;`
extern-function-declaration
                           -> `extern` function-prototype `;`
function-declaration      -> function-prototype block
function-prototype        -> `fn` function-name function-comptime-parameter-clause?
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

trait-declaration         -> `trait` identifier `{` trait-method-declaration* `}`
trait-method-declaration  -> `fn` identifier `(` trait-receiver
                              (`,` parameter)* `,`? `)` `:` type
                              (`;` | block)
trait-receiver            -> `mut`? `self`
impl-declaration          -> `impl` qualified-name `for` type
                              `{` method-declaration* `}`

data-declaration          -> `data` identifier comptime-parameter-clause?
                              `=` data-constructor-list `;`
data-constructor-list     -> data-constructor (`|` data-constructor)*
data-constructor          -> uppercase-identifier data-constructor-payload?
data-constructor-payload  -> `(` (type (`,` type)* `,`?)? `)`

comptime-type-alias-declaration
                           -> `comptime` identifier comptime-parameter-clause?
                              `=` comptime-alias-body `;`
comptime-alias-body       -> type
comptime-alias-body       -> `struct` `{` struct-member-list? `}`
```

`data` 只在 declaration 起始位置作为 contextual keyword；字段、参数和局部变量仍可
使用 `data` 作为名字。data constructor 名必须以大写字母开头，Parser 据此把
constructor expression 和普通 function call 区分开：

```mulberry
data LookupKey =
    ById(UInt64)
  | ByName(String)
  | ByIdAndName(UInt64, String);

data Tree<T> =
    Empty
  | Node(T, Tree<T>, Tree<T>);
```

一个 constructor 可以没有 payload，也可以有一个或多个 payload；同一 data type 的
不同 constructor、同一 constructor 内的不同 payload 都可以使用不同类型。声明中的
零 payload constructor 不写 `()`，构造 value 时统一使用 call spelling，例如
`Empty()`、`ById(42)` 和 `ByIdAndName(42, "Mulberry")`。

generic data type 复用普通 comptime 参数。generic constructor 第一版由 expected type
确定 specialization，不引入额外 constraint solver：

```mulberry
fn emptyTree(): Tree<UInt64> {
  return Empty();
}
```

data type 是 object reference。高层 MLIR 用 nominal `mulberry_core.data` type 表示
concrete specialization，用 `mulberry_core.data.construct` 保存 constructor 名、tag 和
payload SSA values；递归 payload 不会展开成循环的 MLIR type graph。lowering 为每个
variant 分配独立的 GC-managed object，物理布局为 `{i64 tag, payload...}`，payload 顺序
与 declaration 一致。不同 variant 不需要共享同一个最大 union layout；函数参数和返回值
统一使用 object pointer。generic function inference 可以把 `Tree<T>` 与 concrete
`Tree<UInt64>` 对应起来，并从 nominal data type 的 comptime arguments 推导 `T`。
`()` 可以作为 constructor payload，例如 `Result<(), E>`。它参与 semantic type、generic
specialization 和 constructor arity，但不占 runtime storage，也不产生 dummy SSA value；
因此 `Ok(())` 的物理 variant layout 只有 tag。
标准库 `Result<T, E>` 已使用普通 `data` 实现；postfix `?` 在函数返回的
`Result` 中传播同类型错误。

函数参数和 method receiver 默认是 readonly object reference。需要在 callee 里
修改 object 时，参数前写 `mut`：

```mulberry
fn read(xs: List<UInt64>): UInt64 {
  return xs.size();
}

fn append(mut xs: List<UInt64>, value: UInt64): () {
  xs.push(value);
}
```

## Comptime Parameters

```text
comptime-parameter-clause -> `<` comptime-parameter-list `>`
comptime-parameter-list   -> comptime-parameter (`,` comptime-parameter)*
comptime-parameter        -> identifier
comptime-parameter        -> identifier `:` `UInt64`
function-comptime-parameter-clause
                          -> `<` function-comptime-parameter-list `>`
function-comptime-parameter-list
                          -> function-comptime-parameter
                             (`,` function-comptime-parameter)*
function-comptime-parameter
                          -> comptime-parameter
function-comptime-parameter
                          -> identifier `:` qualified-name

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

函数的类型参数还可以带一个 Trait constraint，例如 `fn show<T: Show>(value: T)`。
第一版只支持单一 constraint；type alias 和 data declaration 的 comptime 参数仍只接受
普通类型参数或 `N: UInt64`。Trait method 的 `self` 类型由 `impl Trait for Type` 中的
`Type` 隐式确定，不提供 `Self` source type。Trait 在 generic specialization 后静态解析
成普通 direct call，不产生 trait object、vtable 或 runtime metadata。

Trait method 可以提供 default body。具体 `impl` 可以省略有 default body 的 method；
Sema 会在 concrete target type 上实例化该 body，并把调用解析为普通 static direct call。
`Show` 是 special language trait：String、builtin scalar 和 source object 都满足
`Show`；Unit 没有可观察值，不参与 formatting。generic API 仍显式写出 `T: Show`，
例如 `fn formatValue<T: Show>(value: T): String`。
完整的 Trait 语义、default witness 和当前边界见 [Trait](Traits.md)。

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
type                      -> function-type
type                      -> computed-type-expression
type                      -> type array-type-suffix
function-type             -> `fn` `(` function-type-parameter-list? `)` `:` type
function-type-parameter-list
                          -> function-type-parameter
                             (`,` function-type-parameter)* `,`?
function-type-parameter   -> `mut`? type
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

函数类型只描述参数和返回类型，不重复参数名：

```mulberry
fn(UInt64): UInt64
fn(mut Box): UInt64
fn(UInt64): ()
```

`mut` 是函数类型的一部分，表示 callee 可以修改对应 object；后端 ABI 即使相同，
`fn(Box): UInt64` 和 `fn(mut Box): UInt64` 也不是同一个 semantic type。

普通 non-extern named function 可以直接作为 value，用于局部变量、参数和返回值；
函数值调用仍使用普通 call syntax：

```mulberry
fn apply(callback: fn(UInt64): UInt64, value: UInt64): UInt64 {
  return callback(value);
}

const callback = increment;
var result = apply(callback, 41);
```

named function 的签名仍必须显式书写。generic function template 本身不是 runtime value；
extern function 也暂不作为 source function value，因为当前 extern object ABI 与普通
Mulberry function ABI 不同。FFI callback 留给后续真实需求设计。

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
postfix-expression        -> postfix-expression `?`

primary-expression        -> literal-expression
primary-expression        -> identifier
primary-expression        -> call-expression
primary-expression        -> data-constructor-expression
primary-expression        -> lambda-expression
primary-expression        -> match-expression
primary-expression        -> struct-literal-expression
primary-expression        -> array-literal-expression

lambda-expression         -> `|` lambda-parameter-list? `|` expression
lambda-parameter-list     -> identifier (`,` identifier)* `,`?

literal-expression        -> unit-literal
literal-expression        -> boolean-literal
literal-expression        -> integer-literal
literal-expression        -> float-literal
literal-expression        -> string-literal
literal-expression        -> char-literal

call-expression           -> qualified-name call-argument-clause
data-constructor-expression
                           -> qualified-name call-argument-clause
call-argument-clause      -> `(` expression-list? `)`
expression-list           -> expression (`,` expression)* `,`?

struct-literal-expression -> type `{` expression-list? `}`
array-literal-expression  -> `[` expression-list? `]`
```

`data-constructor-expression` 的 `qualified-name` 最后一个 component 必须以大写字母
开头；普通 `call-expression` 的 callee 不受此限制。两者共享 `(...)` argument 语法，
最终由 Sema 将 constructor 绑定到 expected data type 的具体 variant。

第一版 lambda 只有 expression body，并且不捕获外层局部变量。参数类型来自 expected
function type，返回类型由 body 推断：

```mulberry
const increment: fn(UInt64): UInt64 = |value| value + 1;
var result = apply(|value| value * 2, 21);
```

lambda 参数不重复书写类型或 `mut`；`fn(mut Box): UInt64` 中的 `mut` 由 contextual
function type 带入。没有 expected function type 的 `const f = |x| x + 1;` 暂不成立，
因为 compiler 无法仅从 body 确定 `x` 的类型。

generic callback 先从其它实参确定 lambda 参数类型，再从 body 反推剩余 generic type：

```mulberry
fn apply<T, U>(callback: fn(T): U, value: T): U {
  return callback(value);
}

var answer = apply(|value| value + 1, 41);
```

Sema 把 noncapturing lambda 提升为普通 private function；lambda expression 只是该函数的
function value，因此继续使用标准 `func.constant` 和 `func.call_indirect`，不引入 closure
object 或 lambda-specific lowering。

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
statement                 -> match-statement
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
match-pattern             -> uppercase-qualified-identifier match-bindings?
match-statement           -> `match` expression `{` match-statement-arm+ `}`
match-statement-arm       -> match-pattern
                              `=>` statement
match-statement-arm       -> match-pattern
                              `=>` block
match-expression          -> `match` expression `{` match-expression-arm+ `}`
match-expression-arm      -> match-pattern `=>` expression `,`
match-expression-arm      -> match-pattern `=>` match-expression-block `,`?
match-expression-block    -> `{` statement* `yield` expression `;` `}`
match-bindings            -> `(` (identifier (`,` identifier)* `,`?)? `)`
while-statement           -> `while` condition block
for-statement             -> `for` identifier `in` expression `..` expression block
condition                 -> expression
```

`if` 只有一种语法。如果 condition 能在 Sema 得到 comptime Bool，只分析并生成选中的
block，未选 block 不参与当前 specialization 的类型检查；如果 condition 是 runtime
Bool，则按普通控制流分析两个 block，并在 MLIR 中生成 `scf.if`。

match statement 不产生 value。scrutinee 必须是 `data` type，每个 constructor 必须
恰好出现一次；不接受遗漏或重复的 arm。pattern 只支持一层 constructor 和直接
payload binding，不支持 wildcard、nested pattern 或 guard。payload binding 默认
readonly，其 concrete type 来自 scrutinee 的 data specialization：

```mulberry
match key {
  ById(id) => {
    io.println(id);
  }
  ByName(name) => io.println(name);
  Missing => io.println(0);
}
```

单条 statement arm 不需要额外大括号；它仍然遵循普通 statement 的分号规则。
Parser 在 AST 中把它规范化为单元素 block，因此 statement 和 block arm 共享同一套
Sema 与 MLIRGen 语义。

match expression 是专用的 value-producing expression，不会恢复通用 block
expression。short arm 直接给出 expression；需要多条 statement 时，arm block 必须以
terminal `yield expression;` 结束：

```mulberry
const score = match value {
  Some(score) => score,
  None() => 0,
};

const adjusted = match value {
  Some(score) => {
    const bonus = calculateBonus(score);
    yield score + bonus;
  }
  None() => 0,
};
```

`yield` 在这里是 contextual keyword，只在 match expression arm block 的末尾具有
特殊含义；其它位置仍可作为普通 identifier。第一版不支持 arm 中间的 early yield，
也不允许用 `return` 代替 terminal yield。arm 不能用 `break` / `continue` 跳到 match
expression 外层的 loop；arm 内部新建的 loop 仍可正常使用 `break` / `continue`，然后
由 terminal yield 产生 arm value。所有 arm result 必须具有同一个 semantic type；
存在 expected type 时逐 arm 检查，否则由第一个 arm result 确定。复杂 arm 中 terminal
yield 可以引用前面 statement 声明的局部变量。

零 payload pattern 可以写成 `Missing` 或 `Missing()`。scrutinee 只求值一次；高层
MLIR 用 `mulberry_core.data.tag` 读取 tag，以 `arith.cmpi` 和 `scf.if` 选择 arm，进入
arm 后再用 `mulberry_core.data.unpack` 取得 payload。MLIRGen 不读取物理字段；lowering
才根据该 constructor 的 `{i64 tag, payload...}` variant layout 生成 GEP 和 load。
match expression 使用 result-producing `scf.if` / `scf.yield` 合并 SSA value；object
result 通过 SCF structural type conversion 降为 backend pointer。

标准库在 `std.result` 中直接定义：

```mulberry
data Result<T, E> =
    Ok(T)
  | Err(E);
```

prelude 默认导入 `Result`、`Ok` 和 `Err`。可以使用 exhaustive `match` 显式消费
Result：

```mulberry
fn openAndClose(path: String): Result<(), io.FileError> {
  match io.open(path, "rb") {
    Ok(file) => return io.close(file);
    Err(error) => return Err(error);
  }
}
```

postfix `?` 用于向当前函数传播错误：

```mulberry
fn openForRead(path: String): Result<File, io.FileError> {
  const file = io.open(path, "rb")?;
  return Ok(file);
}
```

第一版只接受 canonical `std.result.Result<T, E>`。operand 为 `Ok(value)` 时，`?`
expression 的类型和值是 `T`；operand 为 `Err(error)` 时，当前函数立即返回
`Err(error)`。因此当前函数必须返回 `Result<U, E>`，并且两边的 `E` 必须是同一个
semantic type。当前不做 `From` conversion、通用 `Try` trait 或 Monad abstraction。

如果 `T` 是 object，`?` 继承 operand expression 的 mutability。直接 call result 与普通
object-returning call 一样，可以绑定到 mutable local；readonly Result variable 解包后
仍然 readonly：

```mulberry
var tensor = safetensors.read(file, "weights")?;

const saved = safetensors.read(file, "weights");
const readonlyTensor = saved?;
```

operand 只求值一次。Err 分支是真正的 function early return，同一个 expression 中位于
`?` 后面的调用和其它副作用不会执行。`?` 可以出现在普通 expression、match expression
arm 和 loop body 中；它与显式 `return` 不同，Ok path 仍会产生当前 expression 所需的
value。需要保留 structured loop 给 Tensor bufferization 的 stdlib 代码，会在 loop 内
使用显式 `match` 传播错误，而不是把 `?` 放进 loop body。

高层 IR 用 `mulberry_core.result.try` 保留这一控制效果。`lower-result-try` 只把实际包含
该 op 的 SCF 祖先转成 CFG：Ok payload 进入 continuation block，Err path 直接构造
当前函数的 Result 并执行 `func.return`。同一函数中与 `?` 无关的 SCF loop/if
保持 structured，因此 Tensor bufferization 和 ownership pass 仍可以分析它们。

`io.open/read/readExact/write/seek/tell/close` 都返回 `Result<..., io.FileError>`。
`read()` 的正常 EOF 短读是 `Ok(actualBytes)`；`readExact()` 把短读变成
`UnexpectedEnd(expected, actual)`。safetensors 的 header、metadata bytes 和 Tensor
payload 都使用 exact read。cursor parser 返回 `Result<..., json.JsonError>`；它只覆盖
safetensors 所需 ASCII JSON subset，不构造 DOM，也不静默接受 unterminated string、
缺少 punctuation/digit 或 escape。safetensors public API 返回
`Result<..., safetensors.SafetensorsError>`，明确组合 `IoFailure`、`JsonFailure`、
`TensorNotFound` 和 `InvalidTensor`。

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
