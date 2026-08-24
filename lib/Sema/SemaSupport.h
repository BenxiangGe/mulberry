//===--- SemaSupport.h - Shared semantic-analysis helpers -*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_SEMA_SUPPORT_H
#define MULBERRY_SEMA_SUPPORT_H

#include "SemaData.h"
#include "Symbols.h"
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace mulberry {

auto formatNameDiagnostic(const char *diagnostic, std::string_view name)
    -> std::string;
auto formatNameSizeDiagnostic(const char *diagnostic, std::string_view name,
                              size_t size) -> std::string;
auto formatTypeTraitDiagnostic(const char *diagnostic, const Type *type,
                               std::string_view traitName) -> std::string;
auto formatMissingFromDiagnostic(const Type *sourceType,
                                 const Type *targetType) -> std::string;
auto declareName(NameSet &names, std::string_view name) -> bool;
auto packageNameOf(std::string_view name) -> std::string;
auto createMemberAccessChain(llvm::SMLoc location, std::string_view name)
    -> std::unique_ptr<Expr>;
auto containsReturnStat(const Expr *node) -> bool;
auto containsReturnStat(const Stat *node) -> bool;
auto containsReturnStat(const BlockExpr *node) -> bool;
auto methodFunctionName(std::string_view ownerName,
                        std::string_view methodName) -> std::string;
auto isSourceObjectType(const Type *type) -> bool;
auto isMutableSourceObjectType(const Type *type) -> bool;
auto isIntegerWidening(const Type *sourceType, const Type *targetType) -> bool;
auto mangleTypeName(std::string name) -> std::string;
auto genericTypeName(std::string_view declarationName,
                     const std::vector<ComptimeArgument> &arguments)
    -> std::string;
auto sameReturnType(const Type *returnType, const Type *actualType) -> bool;
auto formatStringificationType(const Type *type) -> std::string;

auto cloneTypeNode(const TypeNode *node) -> std::unique_ptr<TypeNode>;
auto typeToTypeNode(const Type *type, llvm::SMLoc location)
    -> std::unique_ptr<TypeNode>;
auto substituteTypeNode(const TypeNode *node,
                        const TypeSubstitution &substitution)
    -> std::unique_ptr<TypeNode>;
auto substituteTypeNode(
    const TypeNode *node,
    const std::vector<TypeSubstitution> &substitutions)
    -> std::unique_ptr<TypeNode>;
auto containsSelfType(const TypeNode *node) -> bool;
auto containsComptimeParameter(
    const TypeNode *node, const std::vector<ComptimeParam> &parameters) -> bool;
auto substituteBlockExpr(
    const BlockExpr *node,
    const std::vector<TypeSubstitution> &substitutions)
    -> std::unique_ptr<BlockExpr>;
auto substituteComptimeBlockExpr(
    const ComptimeBlockExpr *node,
    const std::vector<TypeSubstitution> &substitutions)
    -> std::unique_ptr<ComptimeBlockExpr>;
auto substituteExpr(const Expr *node,
                    const std::vector<TypeSubstitution> &substitutions)
    -> std::unique_ptr<Expr>;
auto instantiateFunctionDecl(
    const FunctionDecl *node, std::string_view concreteName,
    const std::vector<TypeSubstitution> &substitutions,
    const std::vector<ComptimeValue> &comptimeArguments = {},
    const std::vector<InferredComptimeArgument> *inferredArguments = nullptr)
    -> std::unique_ptr<FunctionDecl>;
auto toComptimeValues(const std::vector<ComptimeArgument> &arguments)
    -> std::vector<ComptimeValue>;

} // namespace mulberry

#endif // MULBERRY_SEMA_SUPPORT_H
