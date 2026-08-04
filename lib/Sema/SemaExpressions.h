//===--- SemaExpressions.h - Expression semantic analysis -----------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_SEMA_EXPRESSIONS_H
#define MULBERRY_SEMA_EXPRESSIONS_H

#include "SemaData.h"
#include "mulberry/AST/AST.h"
#include "Symbols.h"
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace mulberry {

class SemaImpl;

class ExpressionSema {
public:
  explicit ExpressionSema(SemaImpl &sema) : _sema(sema) {}

  auto sema(Expr *node) -> llvm::LogicalResult;
  auto sema(Expr *node, const Type *type) -> llvm::LogicalResult;
  auto semaExpected(std::unique_ptr<Expr> &node, const Type *type)
      -> llvm::LogicalResult;
  auto sema(DataConstructorExpr *node) -> llvm::LogicalResult;
  auto sema(DataConstructorExpr *node, const DataType *dataType)
      -> llvm::LogicalResult;
  auto semaGenericCall(CallExpr *node, const GenericFunctionSymbol *symbol,
                       const Type *expectedType = nullptr)
      -> llvm::LogicalResult;
  auto sema(UnitExpr *node) -> llvm::LogicalResult;
  auto sema(ComptimeBlockExpr *node) -> llvm::LogicalResult;
  auto sema(LambdaExpr *node) -> llvm::LogicalResult;
  auto sema(MatchExpr *node, const Type *expectedType = nullptr)
      -> llvm::LogicalResult;
  auto sema(TryExpr *node) -> llvm::LogicalResult;
  auto sema(LambdaExpr *node, const FunctionType *functionType)
      -> llvm::LogicalResult;
  auto semaLambda(LambdaExpr *node,
                  const std::vector<const Type *> &parameterTypes,
                  const std::vector<bool> &parameterCanMutateObject,
                  const Type *expectedReturnType,
                  std::string_view packageName) -> llvm::LogicalResult;
  auto sema(CallExpr *node) -> llvm::LogicalResult;
  auto semaIndirectCall(CallExpr *node, const FunctionType *functionType,
                        const Type *expectedType = nullptr)
      -> llvm::LogicalResult;
  auto semaMethodCall(CallExpr *node,
                      const Type *expectedType = nullptr)
      -> llvm::LogicalResult;
  auto semaDottedMethodCall(CallExpr *node,
                            const Type *expectedType = nullptr)
      -> llvm::LogicalResult;
  auto sema(StructLiteralExpr *node) -> llvm::LogicalResult;
  auto sema(VariableExpr *node) -> llvm::LogicalResult;
  auto sema(MemberExpr *node) -> llvm::LogicalResult;
  auto sema(IntegerLiteralExpr *node) -> llvm::LogicalResult;
  auto sema(IntegerLiteralExpr *node, const Type *type)
      -> llvm::LogicalResult;
  auto sema(IntegerWidenExpr *node) -> llvm::LogicalResult;
  auto sema(FloatLiteralExpr *node) -> llvm::LogicalResult;
  auto sema(FloatLiteralExpr *node, const Type *type)
      -> llvm::LogicalResult;
  auto sema(BoolLiteralExpr *node) -> llvm::LogicalResult;
  auto sema(StringLiteralExpr *node) -> llvm::LogicalResult;
  auto sema(InterpolatedStringExpr *node) -> llvm::LogicalResult;
  auto sema(ObjectIdentityExpr *node) -> llvm::LogicalResult;
  auto sema(CharLiteralExpr *node) -> llvm::LogicalResult;
  auto sema(TypeLayoutExpr *node) -> llvm::LogicalResult;
  auto sema(HeapAllocExpr *node) -> llvm::LogicalResult;
  auto sema(AssignExpr *node) -> llvm::LogicalResult;
  auto sema(BinaryExpr *node) -> llvm::LogicalResult;
  auto sema(BinaryExpr *node, const Type *expectedType)
      -> llvm::LogicalResult;
  auto checkStringConcatFunction(Expr *node, const Type *stringType)
      -> llvm::LogicalResult;
  auto semaFormatValueCall(std::unique_ptr<Expr> &expression,
                           const Type *stringType) -> llvm::LogicalResult;
  auto checkAssignable(const Expr *expr) -> llvm::LogicalResult;
  auto canMutateObjectReference(const Expr *expr) -> bool;
  auto checkConstObjectUseAsMutable(const Expr *expr)
      -> llvm::LogicalResult;
  auto checkMutableObjectArgument(const FunctionType *functionType,
                                  size_t index, const Expr *arg)
      -> llvm::LogicalResult;
  auto arrayLiteralTypeWithLeaf(const ArrayLiteralExpr *expr,
                                const Type *leafType) -> const ArrayType *;
  auto semaDefaultArrayLiteral(ArrayLiteralExpr *expr)
      -> llvm::LogicalResult;
  auto semaTensorDisposeCall(CallExpr *node) -> llvm::LogicalResult;
  auto semaTensorIsDisposedCall(CallExpr *node) -> llvm::LogicalResult;
  auto semaTensorStorageAllocCall(CallExpr *node, const Type *expectedType)
      -> llvm::LogicalResult;
  auto sema(ArrayLiteralExpr *expr) -> llvm::LogicalResult;
  auto sema(ArrayLiteralExpr *expr, const ArrayType *type)
      -> llvm::LogicalResult;
  auto semaArrayLiteralElement(std::unique_ptr<Expr> &expr, const Type *type)
      -> llvm::LogicalResult;
  auto sema(IndexExpr *expr) -> llvm::LogicalResult;

private:
  auto resolveFunctionName(std::string_view name) -> std::string;
  auto genericFunctionName(std::string_view name,
                           const Type *argumentType) const -> std::string;
  auto genericFunctionName(
      std::string_view name,
      const std::vector<InferredComptimeArgument> &arguments) const
      -> std::string;
  auto genericFunctionPackageName(std::string_view name) const -> std::string;
  auto sameCallArgumentType(const Type *parameterType, const Type *actualType,
                            bool allowAddressOf) const -> bool;
  auto makeInferredComptimeArguments(
      const std::vector<ComptimeParam> &parameters) const
      -> std::vector<InferredComptimeArgument>;
  auto comptimeSubstitutions(
      const std::vector<ComptimeParam> &parameters,
      const std::vector<InferredComptimeArgument> &arguments) const
      -> std::vector<TypeSubstitution>;
  auto comptimeParameterIndex(const std::vector<ComptimeParam> &parameters,
                              std::string_view name) const
      -> std::optional<size_t>;
  auto hasComputedType(const TypeNode *typeNode,
                       NameSet &visitingAliases) -> bool;
  auto hasComputedType(const TypeNode *typeNode) -> bool;
  auto bindComptimeTypeArgument(const Type *type,
                                InferredComptimeArgument &argument,
                                llvm::SMLoc location) -> bool;
  auto bindComptimeUInt64Argument(uint64_t value,
                                  InferredComptimeArgument &argument) -> bool;
  auto computedArrayLeafParameterIndex(
      const TypeNode *pattern,
      const std::vector<ComptimeParam> &parameters) const
      -> std::optional<size_t>;
  auto matchComptimeArgument(
      const ComptimeArg &pattern, const ComptimeValue &actual,
      const std::vector<ComptimeParam> &parameters,
      std::vector<InferredComptimeArgument> &arguments,
      std::vector<const Type *> *arrayLeafConstraints = nullptr) -> bool;
  auto matchGenericType(const TypeNode *pattern, const Type *actualType,
                        const std::vector<ComptimeParam> &parameters,
                        std::vector<InferredComptimeArgument> &arguments,
                        std::vector<const Type *> *arrayLeafConstraints =
                            nullptr)
      -> bool;
  auto matchMethodReceiverType(
      const TypeNode *pattern, const Type *actualType,
      const std::vector<ComptimeParam> &parameters,
      std::vector<InferredComptimeArgument> &arguments) -> bool;
  auto instantiateGenericFunction(const Node *diagnosticNode,
                                  std::string_view name,
                                  const Type *argumentType,
                                  std::string &concreteName)
      -> llvm::LogicalResult;
  auto resolveSubstitutedType(
      const TypeNode *typeNode,
      const std::vector<TypeSubstitution> &substitutions) -> const Type *;

  SemaImpl &_sema;
};

} // namespace mulberry

#endif // MULBERRY_SEMA_EXPRESSIONS_H
