//===--- SemaTraits.h - Trait semantic analysis ----------------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_SEMA_TRAITS_H
#define MULBERRY_SEMA_TRAITS_H

#include "SemaData.h"
#include "mulberry/AST/AST.h"
#include <string_view>
#include <vector>

namespace mulberry {

class SemaImpl;

class TraitSema {
public:
  explicit TraitSema(SemaImpl &sema) : _sema(sema) {}

  auto resolveConstraints(const Node *node,
                          std::vector<ComptimeParam> &parameters)
      -> llvm::LogicalResult;
  auto sema(TraitDecl *node) -> llvm::LogicalResult;
  auto sema(ImplDecl *node) -> llvm::LogicalResult;
  auto checkConstraints(
      const Node *node, const std::vector<ComptimeParam> &parameters,
      const std::vector<InferredComptimeArgument> &arguments)
      -> llvm::LogicalResult;
  auto materializeMethod(const Node *diagnosticNode, const Type *type,
                         std::string_view methodName,
                         std::string &functionName)
      -> llvm::LogicalResult;

private:
  auto lookupTrait(std::string_view name) -> const TraitDecl *;
  auto methodFunctionName(const TraitDecl *trait, const Type *targetType,
                          std::string_view methodName) const -> std::string;
  auto instantiateDefaultMethod(const TraitDecl *trait,
                                const TraitMethodDecl *method,
                                const Type *targetType,
                                std::string &functionName)
      -> llvm::LogicalResult;
  auto instantiateGenericMethod(const ImplDecl *impl,
                                const FunctionDecl *method,
                                const TraitMethodDecl *contract,
                                const Type *targetType,
                                std::string &functionName)
      -> llvm::LogicalResult;
  auto genericImplementationMatches(const ImplDecl *impl, const Type *type,
                                    bool &matches) -> llvm::LogicalResult;
  auto findMatchingGenericImplementations(
      const Type *type, const TraitDecl *trait,
      std::vector<const ImplDecl *> &implementations) -> llvm::LogicalResult;
  auto materializeImplementation(const Node *diagnosticNode, const Type *type,
                                 const TraitDecl *trait, bool &matched)
      -> llvm::LogicalResult;
  auto typeConforms(const Node *diagnosticNode, const Type *type,
                    const TraitDecl *trait, bool &conforms)
      -> llvm::LogicalResult;

  SemaImpl &_sema;
};

} // namespace mulberry

#endif // MULBERRY_SEMA_TRAITS_H
