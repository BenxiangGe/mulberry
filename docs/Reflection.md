# 编译期反射

## 目标

Mulberry 的 reflection 用来让普通 Mulberry 标准库代码在编译期观察 semantic type，
随后生成已经定型的普通程序。它首先服务两个已经出现的真实需求：

1. 把 String formatting 的 builtin/Array/struct/method 分派从 Sema 搬到 stdlib。
2. 把 `tensor.from(array)` 的 rank、shape、leaf element type 和 row-major flatten
   从 Sema/MLIRGen 搬到 `stdlib/std/tensor.mulberry`。

这不是 runtime reflection。编译器不为 object 生成 `TypeInfo` metadata，也不让
reflection query 进入 MLIR。所有 query 都必须在 Sema 完成求值。

## 当前语法

类型参数通过 `typeInfo(T)` 进入 reflection：

```mulberry
fn inspect<T>(value: T): UInt64 {
  const info = typeInfo(T);
  if info.isUInt64() {
    return value + 1;
  }
  return 0;
}
```

已有 value 的 semantic type 通过 `typeOf(value)` 获得：

```mulberry
if typeOf(value).isArray() {
  return value[0];
}
```

Mulberry 不增加 `if comptime` 语法。普通 `if` 的 condition 如果能在 Sema 得到
comptime Bool，就只对选中的 block 做普通 semantic analysis，未选 block 可以包含只对
其它 specialization 合法的代码；MLIRGen 只生成选中的 block，不产生 reflection call
或 runtime type test。如果 condition 依赖 runtime value，Sema 仍分析两个 block，
MLIRGen 仍生成普通 `scf.if`。

comptime value 可以保存在普通 `const` local 中，不增加 `comptime const` 语法：

```mulberry
const info = typeInfo(T);
const isArray = info.isArray();

if isArray {
  const rank = info.arrayRank();
  return rank;
}
```

initializer 依赖 reflection/type value 时，该 binding 是 comptime-only，声明本身不生成
MLIR。Type 只能继续参与 reflection；Bool、UInt64 和 String 在普通 value position 中
使用时直接物化为常量。普通 `const answer = 42` 仍保留原来的 runtime binding，同时
在 symbol 中缓存 `42`，因此后续编译期条件可以复用该值而不改变对象 identity 或现有
codegen。

当前 type reflection query 包括：

```text
isBool()
isUInt8()
isUInt64()
isFloat32()
isArray()
isStruct()
hasMethod(name)
arrayElementType()
arrayLeafElementType()
arrayLength()
arrayRank()
```

`arrayElementType()` 和 `arrayLeafElementType()` 返回新的 comptime Type，可以继续
调用 query。`arrayLength()` 和 `arrayRank()` 返回 comptime UInt64。comptime
expression 当前支持 Bool/UInt64/String/Type literal value，以及这些值需要的
算术、比较、相等、`and` 和 `or`。

## 类型位置

reflection expression 可以直接产生 generic type argument：

```mulberry
fn makeTensor<A>(value: A): UInt64 {
  var result: Tensor<typeInfo(A).arrayLeafElementType()> = tensor.from(value);
  return result.numel();
}
```

这不是 `Tensor` 的特殊语法。同一机制也可以用于普通 generic type：

```mulberry
const row: Array<typeInfo(A).arrayLeafElementType(), 2> = value[0];
```

parser 用 `ComputedTypeNode` 保存 type position 中的正常 expression AST。generic
function 单态化时，现有 substitution 会同时替换 expression 内的 type parameter；
Sema 随后调用同一个 comptime evaluator，并要求结果是 canonical semantic Type。
computed type 到此被完全消费，MLIRGen 只看到声明上已经确定的类型。

computed type 可以出现在 local type annotation、函数参数、函数返回类型和 generic
alias 中：

```mulberry
comptime LeafTensor<A> =
    Tensor<typeInfo(A).arrayLeafElementType()>;

fn pass<A>(value: LeafTensor<A>, source: A): LeafTensor<A> {
  return value;
}
```

generic call 按下面的顺序分析：

1. 从不含 computed type 的参数和可逆 expected return type 推断 comptime 参数；若
   expected return 通过 `arrayLeafElementType()` 约束 Array literal，则同时记录 leaf
   constraint。
2. 对已经确定的 comptime 参数做 substitution。
3. 求值 computed 参数和返回类型，建立 concrete function signature。
4. 用 concrete signature 验证全部实参和 expected return type。

这个顺序与参数排列无关：上例的 `value` 可以写在 `source` 前面。单独一个
`arrayLeafElementType()` 结果仍不能反推出 `A`，因为多个 rank/shape 可以具有同一 leaf
type；但是 Array literal 自身已经给出 rank/shape，因此 expected `Tensor<UInt8>` 与
`Tensor<typeInfo(A).arrayLeafElementType()>` 可以共同把 `[10, 20]` 唯一定型为
`Array<UInt8, 2>`。非 literal 参数仍必须从实参类型取得完整 `A`，编译器不会猜 shape。

## 实现边界

semantic `Type` 是 reflection 的唯一类型身份。类型名只用于 diagnostic 和
`hasMethod("toString")` 这样的 symbol lookup，不允许用 String 代替类型身份。

`typeInfo(T)` 直接把 `T` 解析为 type position，并在 AST 中成为 `TypeInfoExpr`。
generic function 按具体类型单态化时直接替换该节点持有的 `TypeNode`，不把 type
parameter 伪装成 value expression。`TypeInfoExpr` 由 Sema 消耗，不能进入普通
expression checking 或 MLIRGen。

当前不实现完整 Zig comptime interpreter，也不支持任意编译期 I/O、runtime object
构造、AST macro 或 runtime TypeInfo。recursive generic 不需要额外 interpreter：每次
普通 generic call 都以更具体的 Array element type 单态化，而 comptime `if` 只保留该
specialization 合法的分支。

## String formatting

String formatting 的静态分派已经由 stdlib `formatValue<T>()` 实现：

```mulberry
fn formatValue<T>(value: T): String {
  const info = typeInfo(T);

  if info.isUInt64() {
    return fromUInt64(value);
  }

  const hasToString = info.hasMethod("toString");
  if hasToString {
    return value.toString();
  }

  # 其它 object 进入 internal identity fallback。
}
```

字符串插值和 `String + value` 的 Sema 只负责构造普通
`std.string.formatValue(value)` call，不再根据 value type 选择 scalar formatter、
`toString()` 或 object identity。`formatValue<T>()` 单态化后的普通 Mulberry 代码完成
这些选择。

object identity 仍需要取得隐藏 object reference 和静态类型名，source object model
无法直接表达这条 runtime ABI。因此 stdlib 可以使用 internal `__objectIdentity(value)`
expression；普通用户 source 会被 Sema 拒绝。它只保留必要的 ABI primitive，不负责
formatting policy。

## Tensor 边界

`tensor.from(array)` 已经是 `stdlib/std/tensor.mulberry` 中的普通 source-level generic。
它通过 reflection 取得 rank、leaf type 和每层 Array length，递归生成 shape，再由
`flattenInto<A, T>()` 按 row-major 顺序复制 payload。生成的 MLIR 只包含普通定型函数，
Sema/MLIRGen 不再包含它的 typecheck、payload construction 或 call handler。当前
Tensor-only early-disposal 分析仍需把 `from/zeros` 区分为 owning constructor，把
`reshape/sliceFirst` 区分为共享 storage view；以后由统一的 `Disposable`/ownership effect
替换这项临时分类。

`mulberry_core.tensor.pack` 不是同一类能力。它把 NN/internal tensor 或 lowered
memref descriptor 转成 stdlib Tensor record，是 IR/ABI bridge。source reflection
不能替代这一步。它以后可以移动 package ownership 或重新设计 ABI，但不属于本轮
reflection 清理目标。

## 后续顺序

迁移后的调用者审计已经完成。没有源码需求或测试调用的 `isUnit()` 和旧 Array literal
shape 缓存已经删除。numeric conversion 仍直接对应 primitive IR op，Tensor dispose
仍是当前 Tensor-only lifecycle primitive；它们不是 formatting 或 `tensor.from()` 的
遗留 handler。

后续只根据真实 stdlib 需求增加 field iteration、`compileError()` 或其它 comptime
能力，不预先实现完整 Zig comptime interpreter。
