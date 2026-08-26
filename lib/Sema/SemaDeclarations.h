//===--- SemaDeclarations.h - Declaration semantic analysis --------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_SEMA_DECLARATIONS_H
#define MULBERRY_SEMA_DECLARATIONS_H

#include "mulberry/AST/AST.h"
#include "Symbols.h"
#include <string>
#include <string_view>

namespace mulberry {

class SemaImpl;

class DeclarationSema {
public:
  explicit DeclarationSema(SemaImpl &sema) : _sema(sema) {}

  auto sema(Prototype *node) -> llvm::LogicalResult;
  auto sema(FunctionDecl *node) -> llvm::LogicalResult;
  auto sema(StructDecl *node) -> llvm::LogicalResult;
  auto sema(DataDecl *node) -> llvm::LogicalResult;
  auto sema(ComptimeTypeAliasDecl *node) -> llvm::LogicalResult;
  auto semaFunctionSignature(
      Prototype *node, bool isExtern = false,
      std::string_view packageName = {},
      Visibility visibility = Visibility::Private)
      -> llvm::LogicalResult;

private:
  auto functionPackageName(std::string_view name) const -> std::string;
  auto checkFunctionPacks(Prototype *node) -> llvm::LogicalResult;
  auto semaFunctionParameters(
      Prototype *node, std::vector<const Type *> &parameterTypes,
      std::vector<bool> &parameterCanMutateObject) -> llvm::LogicalResult;
  auto checkPublicFunctionSignature(
      Prototype *node,
      const std::vector<ComptimeParam> &comptimeParameters,
      bool allowSelf = false) -> llvm::LogicalResult;
  auto bindFunctionParameters(Prototype *node,
                              const FunctionSymbol *signature)
      -> llvm::LogicalResult;
  auto declareStructMethods(
      std::string_view ownerName, const VectorUniquePtr<FunctionDecl> &methods,
      const std::vector<ComptimeParam> &typeParameters,
      std::string_view packageName, Visibility ownerVisibility)
      -> llvm::LogicalResult;

  SemaImpl &_sema;
};

} // namespace mulberry

#endif // MULBERRY_SEMA_DECLARATIONS_H
