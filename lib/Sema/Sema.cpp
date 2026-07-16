//===--- Sema.cpp - Mulberry Semantic Analysis ------------------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "mulberry/Sema/Sema.h"
#include "Symbols.h"
#include "mulberry/AST/AST.h"
#include "mulberry/Basic/Builtins.h"
#include "mulberry/Sema/DiagnosticsSema.h"
#include "llvm/Support/Debug.h"
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using namespace mulberry;
using llvm::cast;
using llvm::dyn_cast;

#undef DEBUG_TYPE
#define DEBUG_TYPE "Sema"

using NameSet = std::set<std::string, std::less<>>;

struct TypeSubstitution {
  std::string parameterName;
  const TypeNode *argumentTypeNode = nullptr;
  std::optional<uint64_t> uint64Value;
};

struct ComptimeArgument {
  ComptimeArg::Kind kind = ComptimeArg::Kind::Type;
  const Type *type = nullptr;
  std::unique_ptr<TypeNode> typeNode;
  uint64_t uint64Value = 0;
};

struct InferredComptimeArgument {
  ComptimeParam::Kind kind = ComptimeParam::Kind::Type;
  const Type *type = nullptr;
  std::unique_ptr<TypeNode> typeNode;
  std::optional<uint64_t> uint64Value;

  auto isResolved() const -> bool {
    if (kind == ComptimeParam::Kind::Type)
      return type != nullptr;
    return uint64Value.has_value();
  }
};

struct ComptimeEvaluation {
  enum class Kind {
    Runtime,
    Value,
    Error,
  };

  Kind kind;
  std::optional<ComptimeValue> value;
  bool isComptimeOnly = false;
};

auto substituteExpr(const Expr *node,
                    const std::vector<TypeSubstitution> &substitutions)
    -> std::unique_ptr<Expr>;

auto substituteBlockExpr(const BlockExpr *node,
                         const std::vector<TypeSubstitution> &substitutions)
    -> std::unique_ptr<BlockExpr>;

auto containsReturnStat(const BlockExpr *node) -> bool;

auto toComptimeValues(const std::vector<ComptimeArgument> &arguments)
    -> std::vector<ComptimeValue> {
  std::vector<ComptimeValue> values;
  for (auto &argument : arguments) {
    if (argument.kind == ComptimeArg::Kind::Type)
      values.push_back(ComptimeValue(argument.type));
    else
      values.push_back(ComptimeValue(argument.uint64Value));
  }
  return values;
}

auto replacePlaceholder(std::string message, std::string_view placeholder,
                        std::string_view value) -> std::string {
  auto position = message.find(placeholder);
  if (position == std::string::npos)
    return message;
  message.replace(position, placeholder.size(), value);
  return message;
}

auto formatNameDiagnostic(const char *diagnostic, std::string_view name)
    -> std::string {
  return replacePlaceholder(diagnostic, "%s", name);
}

auto formatNameSizeDiagnostic(const char *diagnostic, std::string_view name,
                              size_t size) -> std::string {
  auto message = formatNameDiagnostic(diagnostic, name);
  return replacePlaceholder(message, "%d", std::to_string(size));
}

auto declareName(NameSet &names, std::string_view name) -> bool {
  return names.insert(std::string(name)).second;
}

auto packageNameOf(std::string_view name) -> std::string {
  auto dot = name.rfind('.');
  if (dot == std::string_view::npos)
    return {};
  return std::string(name.substr(0, dot));
}

auto createMemberAccessChain(llvm::SMLoc location, std::string_view name)
    -> std::unique_ptr<Expr> {
  auto dot = name.find('.');
  if (dot == std::string_view::npos)
    return std::make_unique<VariableExpr>(location, name);

  std::unique_ptr<Expr> expr =
      std::make_unique<VariableExpr>(location, name.substr(0, dot));
  while (dot != std::string_view::npos) {
    auto nextDot = name.find('.', dot + 1);
    auto fieldName = name.substr(dot + 1, nextDot - dot - 1);
    expr = std::make_unique<MemberExpr>(location, std::move(expr), fieldName);
    dot = nextDot;
  }
  return expr;
}

auto containsReturnStat(const Expr *node) -> bool {
  if (auto *block = dyn_cast<BlockExpr>(node))
    return containsReturnStat(block);
  return false;
}

auto containsReturnStat(const Stat *node) -> bool {
  if (dyn_cast<ReturnStat>(node))
    return true;
  if (auto *ifStat = dyn_cast<IfStat>(node)) {
    if (ifStat->comptimeValue()) {
      if (*ifStat->comptimeValue())
        return containsReturnStat(ifStat->thenBlock().get());
      return ifStat->hasElseBlock() &&
             containsReturnStat(ifStat->elseBlock().get());
    }
    if (containsReturnStat(ifStat->thenBlock().get()))
      return true;
    return ifStat->hasElseBlock() &&
           containsReturnStat(ifStat->elseBlock().get());
  }
  if (auto *whileStat = dyn_cast<WhileStat>(node))
    return containsReturnStat(whileStat->bodyBlock().get());
  if (auto *forStat = dyn_cast<ForStat>(node))
    return containsReturnStat(forStat->bodyBlock().get());
  if (auto *exprStat = dyn_cast<ExprStat>(node))
    return containsReturnStat(exprStat->expression().get());
  return false;
}

auto containsReturnStat(const BlockExpr *node) -> bool {
  for (auto &statement : node->statements())
    if (containsReturnStat(statement.get()))
      return true;
  return false;
}

auto methodFunctionName(std::string_view ownerName,
                        std::string_view methodName) -> std::string {
  std::string name(ownerName);
  name += ".";
  name += methodName;
  return name;
}

auto isSourceObjectType(const Type *type) -> bool {
  if (auto *ptrType = getPtrType(type))
    type = ptrType->pointeeType();
  return getStructType(type) || getArrayType(type);
}

auto unqualifiedTypeName(std::string_view name) -> std::string_view {
  auto dot = name.rfind('.');
  return dot == std::string_view::npos ? name : name.substr(dot + 1);
}

auto formatStringificationType(const Type *type) -> std::string {
  if (auto *builtinType = getBuiltinType(type))
    return std::string(builtinType->name());

  if (auto *arrayType = getArrayType(type)) {
    return "Array<" + formatStringificationType(arrayType->elementType()) +
           ", " + std::to_string(arrayType->size()) + ">";
  }

  if (auto *structType = getStructType(type)) {
    auto *origin = structType->origin();
    if (!origin)
      return std::string(unqualifiedTypeName(structType->name()));

    std::string result(unqualifiedTypeName(origin->aliasName()));
    result += "<";
    std::string separator;
    for (auto &argument : origin->arguments()) {
      result += separator;
      if (argument.kind() == ComptimeValue::Kind::Type)
        result += formatStringificationType(argument.type());
      else
        result += std::to_string(argument.uint64Value());
      separator = ", ";
    }
    result += ">";
    return result;
  }

  return formatType(type);
}

auto cloneTypeNode(const TypeNode *node) -> std::unique_ptr<TypeNode> {
  if (auto *unitType = dyn_cast<UnitTypeNode>(node))
    return std::make_unique<UnitTypeNode>(unitType->location());

  if (auto *namedType = dyn_cast<NamedTypeNode>(node))
    return std::make_unique<NamedTypeNode>(namedType->location(),
                                           namedType->name());

  if (auto *computedType = dyn_cast<ComputedTypeNode>(node)) {
    return std::make_unique<ComputedTypeNode>(
        computedType->location(),
        substituteExpr(computedType->expression().get(), {}));
  }

  if (auto *arrayType = dyn_cast<ArrayTypeNode>(node)) {
    return std::make_unique<ArrayTypeNode>(
        cloneTypeNode(arrayType->elementTypeNode()), arrayType->shape(),
        arrayType->location());
  }

  if (auto *ptrType = dyn_cast<PtrTypeNode>(node)) {
    return std::make_unique<PtrTypeNode>(
        cloneTypeNode(ptrType->pointeeTypeNode()), ptrType->location());
  }

  if (auto *genericType = dyn_cast<GenericTypeNode>(node)) {
    std::vector<ComptimeArg> arguments;
    for (auto &argument : genericType->arguments()) {
      if (argument.kind() == ComptimeArg::Kind::UInt64) {
        arguments.push_back(
            ComptimeArg(argument.location(), argument.uint64Value()));
        continue;
      }
      arguments.push_back(ComptimeArg(cloneTypeNode(argument.typeNode())));
    }
    return std::make_unique<GenericTypeNode>(
        genericType->location(), genericType->name(), std::move(arguments));
  }

  auto *structType = cast<StructTypeNode>(node);
  VectorUniquePtr<FieldDecl> fields;
  for (auto &field : structType->fields()) {
    auto variable = std::make_unique<VariableExpr>(
        field->variable()->location(), field->variable()->name());
    fields.push_back(std::make_unique<FieldDecl>(
        field->location(), std::move(variable),
        cloneTypeNode(field->typeNode())));
  }
  VectorUniquePtr<FunctionDecl> methods;
  for (auto &method : structType->methods()) {
    auto functionName = std::make_unique<FunctionName>(
        method->proto()->id()->location(), method->proto()->id()->name());
    VectorUniquePtr<ParameterDecl> parameters;
    for (auto &parameter : method->proto()->parameters()) {
      auto variable = std::make_unique<VariableExpr>(
          parameter->variable()->location(), parameter->variable()->name());
      parameters.push_back(std::make_unique<ParameterDecl>(
          parameter->location(), std::move(variable),
          cloneTypeNode(parameter->typeNode()), parameter->canMutateObject()));
    }
    auto prototype = std::make_unique<Prototype>(
        method->proto()->location(), std::move(functionName),
        std::move(parameters), cloneTypeNode(method->proto()->returnTypeNode()),
        std::vector<ComptimeParam>(method->proto()->comptimeParameters().begin(),
                                   method->proto()->comptimeParameters().end()));
    methods.push_back(std::make_unique<FunctionDecl>(
        method->location(), std::move(prototype),
        substituteBlockExpr(method->body().get(), {})));
  }
  return std::make_unique<StructTypeNode>(
      structType->location(), std::move(fields), std::move(methods));
}

auto typeToTypeNode(const Type *type, llvm::SMLoc location)
    -> std::unique_ptr<TypeNode> {
  if (auto *builtinType = getBuiltinType(type))
    return std::make_unique<NamedTypeNode>(location, builtinType->name());

  if (auto *structType = getStructType(type)) {
    if (auto *origin = structType->origin()) {
      // Keep generic alias identity when a resolved type is substituted into
      // another alias; the internal mangled struct name is not a source type.
      std::vector<ComptimeArg> arguments;
      for (auto &argument : origin->arguments()) {
        if (argument.kind() == ComptimeValue::Kind::UInt64) {
          arguments.push_back(
              ComptimeArg(location, argument.uint64Value()));
          continue;
        }
        arguments.push_back(ComptimeArg(typeToTypeNode(
            argument.type(), location)));
      }
      return std::make_unique<GenericTypeNode>(
          location, origin->aliasName(), std::move(arguments));
    }

    return std::make_unique<NamedTypeNode>(location, structType->name());
  }

  if (auto *arrayType = getArrayType(type)) {
    std::vector<ComptimeArg> arguments;
    arguments.push_back(ComptimeArg(
        typeToTypeNode(arrayType->elementType(), location)));
    arguments.push_back(ComptimeArg(location, arrayType->size()));
    return std::make_unique<GenericTypeNode>(
        location, "Array", std::move(arguments));
  }

  auto *ptrType = cast<PtrType>(type);
  return std::make_unique<PtrTypeNode>(
      typeToTypeNode(ptrType->pointeeType(), location), location);
}

auto substituteTypeNode(const TypeNode *node,
                        const TypeSubstitution &substitution)
    -> std::unique_ptr<TypeNode> {
  if (substitution.parameterName.empty())
    return cloneTypeNode(node);

  if (auto *unitType = dyn_cast<UnitTypeNode>(node))
    return std::make_unique<UnitTypeNode>(unitType->location());

  if (auto *namedType = dyn_cast<NamedTypeNode>(node)) {
    if (namedType->name() == substitution.parameterName &&
        substitution.argumentTypeNode)
      return cloneTypeNode(substitution.argumentTypeNode);
    return cloneTypeNode(namedType);
  }

  if (auto *computedType = dyn_cast<ComputedTypeNode>(node)) {
    return std::make_unique<ComputedTypeNode>(
        computedType->location(),
        substituteExpr(computedType->expression().get(), {substitution}));
  }

  if (auto *arrayType = dyn_cast<ArrayTypeNode>(node)) {
    return std::make_unique<ArrayTypeNode>(
        substituteTypeNode(arrayType->elementTypeNode(), substitution),
        arrayType->shape(), arrayType->location());
  }

  if (auto *ptrType = dyn_cast<PtrTypeNode>(node)) {
    return std::make_unique<PtrTypeNode>(
        substituteTypeNode(ptrType->pointeeTypeNode(), substitution),
        ptrType->location());
  }

  if (auto *genericType = dyn_cast<GenericTypeNode>(node)) {
    std::vector<ComptimeArg> arguments;
    for (auto &argument : genericType->arguments()) {
      if (argument.kind() == ComptimeArg::Kind::UInt64) {
        arguments.push_back(
            ComptimeArg(argument.location(), argument.uint64Value()));
        continue;
      }

      if (auto *namedType = dyn_cast<NamedTypeNode>(argument.typeNode())) {
        if (namedType->name() == substitution.parameterName &&
            substitution.uint64Value) {
          arguments.push_back(
              ComptimeArg(namedType->location(), *substitution.uint64Value));
          continue;
        }
      }

      arguments.push_back(
          ComptimeArg(substituteTypeNode(argument.typeNode(), substitution)));
    }
    return std::make_unique<GenericTypeNode>(
        genericType->location(), genericType->name(), std::move(arguments));
  }

  auto *structType = cast<StructTypeNode>(node);
  VectorUniquePtr<FieldDecl> fields;
  for (auto &field : structType->fields()) {
    auto variable = std::make_unique<VariableExpr>(
        field->variable()->location(), field->variable()->name());
    fields.push_back(std::make_unique<FieldDecl>(
        field->location(), std::move(variable),
        substituteTypeNode(field->typeNode(), substitution)));
  }
  VectorUniquePtr<FunctionDecl> methods;
  for (auto &method : structType->methods()) {
    auto functionName = std::make_unique<FunctionName>(
        method->proto()->id()->location(), method->proto()->id()->name());
    VectorUniquePtr<ParameterDecl> parameters;
    for (auto &parameter : method->proto()->parameters()) {
      auto variable = std::make_unique<VariableExpr>(
          parameter->variable()->location(), parameter->variable()->name());
      parameters.push_back(std::make_unique<ParameterDecl>(
          parameter->location(), std::move(variable),
          substituteTypeNode(parameter->typeNode(), substitution),
          parameter->canMutateObject()));
    }
    auto prototype = std::make_unique<Prototype>(
        method->proto()->location(), std::move(functionName),
        std::move(parameters),
        substituteTypeNode(method->proto()->returnTypeNode(), substitution),
        std::vector<ComptimeParam>(method->proto()->comptimeParameters().begin(),
                                   method->proto()->comptimeParameters().end()));
    methods.push_back(std::make_unique<FunctionDecl>(
        method->location(), std::move(prototype),
        substituteBlockExpr(
            method->body().get(),
            std::vector<TypeSubstitution>{substitution})));
  }
  return std::make_unique<StructTypeNode>(
      structType->location(), std::move(fields), std::move(methods));
}

auto substituteTypeNode(const TypeNode *node,
                        const std::vector<TypeSubstitution> &substitutions)
    -> std::unique_ptr<TypeNode> {
  auto result = cloneTypeNode(node);
  for (auto &substitution : substitutions)
    result = substituteTypeNode(result.get(), substitution);
  return result;
}

auto containsComptimeParameter(const TypeNode *node,
                               const std::vector<ComptimeParam> &parameters)
    -> bool {
  if (auto *namedType = dyn_cast<NamedTypeNode>(node)) {
    for (auto &parameter : parameters)
      if (namedType->name() == parameter.name)
        return true;
    return false;
  }

  if (llvm::isa<ComputedTypeNode>(node))
    return !parameters.empty();

  if (auto *arrayType = dyn_cast<ArrayTypeNode>(node))
    return containsComptimeParameter(arrayType->elementTypeNode(),
                                     parameters);

  if (auto *ptrType = dyn_cast<PtrTypeNode>(node))
    return containsComptimeParameter(ptrType->pointeeTypeNode(), parameters);

  if (auto *genericType = dyn_cast<GenericTypeNode>(node)) {
    for (auto &argument : genericType->arguments()) {
      if (argument.kind() == ComptimeArg::Kind::Type &&
          containsComptimeParameter(argument.typeNode(), parameters))
        return true;
    }
    return false;
  }

  if (auto *structType = dyn_cast<StructTypeNode>(node)) {
    for (auto &field : structType->fields())
      if (containsComptimeParameter(field->typeNode(), parameters))
        return true;
  }

  return false;
}

auto substituteBlockExpr(const BlockExpr *node,
                         const std::vector<TypeSubstitution> &substitutions)
    -> std::unique_ptr<BlockExpr> {
  VectorUniquePtr<Stat> statements;
  for (auto &statement : node->statements()) {
    if (auto *variable = dyn_cast<VariableStat>(statement.get())) {
      auto clonedVariable = std::make_unique<VariableExpr>(
          variable->variable()->location(), variable->variable()->name());
      auto clonedInit = variable->init()
                            ? substituteExpr(variable->init().get(),
                                             substitutions)
                            : nullptr;
      statements.push_back(std::make_unique<VariableStat>(
          variable->location(), std::move(clonedVariable),
          variable->hasExplicitType()
              ? substituteTypeNode(variable->typeNode(), substitutions)
              : nullptr,
          std::move(clonedInit), variable->isConstBinding(),
          variable->canMutateObject()));
      continue;
    }

    if (auto *ifStat = dyn_cast<IfStat>(statement.get())) {
      auto elseBlock = ifStat->hasElseBlock()
                           ? substituteBlockExpr(ifStat->elseBlock().get(),
                                                 substitutions)
                           : nullptr;
      statements.push_back(std::make_unique<IfStat>(
          ifStat->location(),
          substituteExpr(ifStat->conditionExpr().get(), substitutions),
          substituteBlockExpr(ifStat->thenBlock().get(), substitutions),
          std::move(elseBlock)));
      continue;
    }

    if (auto *whileStat = dyn_cast<WhileStat>(statement.get())) {
      statements.push_back(std::make_unique<WhileStat>(
          whileStat->location(),
          substituteExpr(whileStat->conditionExpr().get(), substitutions),
          substituteBlockExpr(whileStat->bodyBlock().get(), substitutions)));
      continue;
    }

    if (auto *forStat = dyn_cast<ForStat>(statement.get())) {
      statements.push_back(std::make_unique<ForStat>(
          forStat->location(), forStat->variableName(),
          substituteExpr(forStat->startExpr().get(), substitutions),
          substituteExpr(forStat->endExpr().get(), substitutions),
          substituteBlockExpr(forStat->bodyBlock().get(), substitutions)));
      continue;
    }

    if (auto *breakStat = dyn_cast<BreakStat>(statement.get())) {
      statements.push_back(std::make_unique<BreakStat>(
          breakStat->location()));
      continue;
    }

    if (auto *continueStat = dyn_cast<ContinueStat>(statement.get())) {
      statements.push_back(std::make_unique<ContinueStat>(
          continueStat->location()));
      continue;
    }

    if (auto *returnStat = dyn_cast<ReturnStat>(statement.get())) {
      auto expression = returnStat->hasExpression()
                            ? substituteExpr(returnStat->expression().get(),
                                             substitutions)
                            : nullptr;
      statements.push_back(std::make_unique<ReturnStat>(
          returnStat->location(), std::move(expression)));
      continue;
    }

    auto *exprStat = cast<ExprStat>(statement.get());
    statements.push_back(std::make_unique<ExprStat>(
        exprStat->location(),
        substituteExpr(exprStat->expression().get(), substitutions)));
  }

  return std::make_unique<BlockExpr>(node->location(), std::move(statements));
}

auto substituteExpr(const Expr *node,
                    const std::vector<TypeSubstitution> &substitutions)
    -> std::unique_ptr<Expr> {
  switch (node->getKind()) {
  case Expr::Expr_Unit:
    return std::make_unique<UnitExpr>(node->location());
  case Expr::Expr_DecimalLiteral: {
    auto *expr = cast<DecimalLiteralExpr>(node);
    return std::make_unique<DecimalLiteralExpr>(expr->location(),
                                                expr->value());
  }
  case Expr::Expr_FloatLiteral: {
    auto *expr = cast<FloatLiteralExpr>(node);
    return std::make_unique<FloatLiteralExpr>(expr->location(),
                                              expr->value());
  }
  case Expr::Expr_BoolLiteral: {
    auto *expr = cast<BoolLiteralExpr>(node);
    return std::make_unique<BoolLiteralExpr>(expr->location(), expr->value());
  }
  case Expr::Expr_StringLiteral: {
    auto *expr = cast<StringLiteralExpr>(node);
    return std::make_unique<StringLiteralExpr>(
        expr->location(), std::string(expr->value()));
  }
  case Expr::Expr_InterpolatedString: {
    auto *expr = cast<InterpolatedStringExpr>(node);
    VectorUniquePtr<Expr> segments;
    for (auto &segment : expr->segments())
      segments.push_back(substituteExpr(segment.get(), substitutions));
    return std::make_unique<InterpolatedStringExpr>(
        expr->location(), std::move(segments));
  }
  case Expr::Expr_ObjectIdentity: {
    auto *expr = cast<ObjectIdentityExpr>(node);
    auto result = std::make_unique<ObjectIdentityExpr>(
        expr->location(),
        substituteExpr(expr->value().get(), substitutions));
    result->setTypeName(expr->typeName());
    return result;
  }
  case Expr::Expr_CharLiteral: {
    auto *expr = cast<CharLiteralExpr>(node);
    return std::make_unique<CharLiteralExpr>(expr->location(), expr->value());
  }
  case Expr::Expr_ArrayLiteral: {
    auto *expr = cast<ArrayLiteralExpr>(node);
    std::vector<std::unique_ptr<Expr>> elements;
    for (auto &element : expr->getElements())
      elements.push_back(substituteExpr(element.get(), substitutions));
    return std::make_unique<ArrayLiteralExpr>(expr->location(),
                                              std::move(elements));
  }
  case Expr::Expr_Index: {
    auto *expr = cast<IndexExpr>(node);
    std::vector<std::unique_ptr<Expr>> indices;
    for (auto &index : expr->indices())
      indices.push_back(substituteExpr(index.get(), substitutions));
    return std::make_unique<IndexExpr>(
        expr->location(),
        substituteExpr(expr->base().get(), substitutions),
        std::move(indices));
  }
  case Expr::Expr_Member: {
    auto *expr = cast<MemberExpr>(node);
    return std::make_unique<MemberExpr>(
        expr->location(),
        substituteExpr(expr->base().get(), substitutions),
        expr->fieldName());
  }
  case Expr::Expr_Variable: {
    auto *expr = cast<VariableExpr>(node);
    for (auto &substitution : substitutions) {
      if (expr->name() != substitution.parameterName)
        continue;
      if (substitution.uint64Value)
        return std::make_unique<DecimalLiteralExpr>(
            expr->location(), *substitution.uint64Value);
    }
    return std::make_unique<VariableExpr>(expr->location(), expr->name());
  }
  case Expr::Expr_Assign: {
    auto *expr = cast<AssignExpr>(node);
    return std::make_unique<AssignExpr>(
        expr->location(),
        substituteExpr(expr->lhs().get(), substitutions),
        substituteExpr(expr->rhs().get(), substitutions));
  }
  case Expr::Expr_Binary: {
    auto *expr = cast<BinaryExpr>(node);
    return std::make_unique<BinaryExpr>(
        expr->location(), expr->opEnum(),
        substituteExpr(expr->lhs().get(), substitutions),
        substituteExpr(expr->rhs().get(), substitutions));
  }
  case Expr::Expr_Block:
    return substituteBlockExpr(cast<BlockExpr>(node), substitutions);
  case Expr::Expr_TypeInfo: {
    auto *expr = cast<TypeInfoExpr>(node);
    return std::make_unique<TypeInfoExpr>(
        expr->location(), substituteTypeNode(expr->typeNode(), substitutions));
  }
  case Expr::Expr_TypeLayout: {
    auto *expr = cast<TypeLayoutExpr>(node);
    return std::make_unique<TypeLayoutExpr>(
        expr->location(), expr->query(),
        substituteTypeNode(expr->typeNode(), substitutions));
  }
  case Expr::Expr_HeapAlloc: {
    auto *expr = cast<HeapAllocExpr>(node);
    auto count = expr->count()
                     ? substituteExpr(expr->count().get(), substitutions)
                     : nullptr;
    return std::make_unique<HeapAllocExpr>(
        expr->location(), substituteTypeNode(expr->typeNode(), substitutions),
        std::move(count));
  }
  case Expr::Expr_Call: {
    auto *expr = cast<CallExpr>(node);
    VectorUniquePtr<Expr> expressions;
    for (auto &argument : expr->expressions())
      expressions.push_back(substituteExpr(argument.get(), substitutions));
    if (expr->hasReceiver()) {
      return std::make_unique<CallExpr>(
          expr->location(),
          substituteExpr(expr->receiver().get(), substitutions),
          expr->name(), std::move(expressions));
    }
    return std::make_unique<CallExpr>(
        expr->location(), expr->name(), std::move(expressions));
  }
  case Expr::Expr_StructLiteral: {
    auto *expr = cast<StructLiteralExpr>(node);
    VectorUniquePtr<Expr> expressions;
    for (auto &argument : expr->expressions())
      expressions.push_back(substituteExpr(argument.get(), substitutions));
    return std::make_unique<StructLiteralExpr>(
        expr->location(), substituteTypeNode(expr->typeNode(), substitutions),
        std::move(expressions));
  }
  }

  llvm_unreachable("Unexpected expression");
}

auto instantiateFunctionDecl(const FunctionDecl *node,
                             std::string_view concreteName,
                             const std::vector<TypeSubstitution> &substitutions)
    -> std::unique_ptr<FunctionDecl> {
  VectorUniquePtr<ParameterDecl> parameters;
  for (auto &parameter : node->proto()->parameters()) {
    auto variable = std::make_unique<VariableExpr>(
        parameter->variable()->location(), parameter->variable()->name());
    parameters.push_back(std::make_unique<ParameterDecl>(
        parameter->location(), std::move(variable),
        substituteTypeNode(parameter->typeNode(), substitutions),
        parameter->canMutateObject()));
  }

  auto functionName =
      std::make_unique<FunctionName>(node->proto()->id()->location(),
                                     concreteName);
  auto prototype = std::make_unique<Prototype>(
      node->proto()->location(), std::move(functionName),
      std::move(parameters),
      substituteTypeNode(node->proto()->returnTypeNode(), substitutions));
  prototype->setIsMethod(node->proto()->isMethod());
  return std::make_unique<FunctionDecl>(
      node->location(), std::move(prototype),
      substituteBlockExpr(node->body().get(), substitutions));
}

class SemaImpl {
public:
  SemaImpl(const llvm::SourceMgr &sourceManager)
      : _sourceManager{sourceManager} {
    addBuiltins();
    registerBuiltinHandlers();
  }

  SemaImpl(const llvm::SourceMgr &sourceManager,
           const std::map<std::string, std::string> &importAliases)
      : _sourceManager{sourceManager}, _importAliases{importAliases} {
    addBuiltins();
    registerBuiltinHandlers();
  }

  auto sema(Module &node) -> MulberryResult {
    for (auto &decl : node)
      if (sema(decl.get()))
        return failure();

    for (size_t i = 0; i < _instantiatedFunctions.size(); ++i) {
      if (sema(_instantiatedFunctions[i].get()))
        return failure();
    }

    auto declarations = node.takeDeclarations();
    for (auto &function : _instantiatedFunctions)
      declarations.push_back(std::move(function));
    node.setDeclarations(std::move(declarations));

    std::string mainName = node.packageName().empty()
                               ? "main"
                               : std::string(node.packageName()) + ".main";
    auto *mainSignature = lookupFunction(mainName);
    if (!mainSignature ||
        !mainSignature->parameterTypes.empty() ||
        !isUInt64Type(mainSignature->returnType))
      return emitError(llvm::SMLoc{}, diag::undefined_main);
    return success();
  }

private:
  using BuiltinHandler =
      std::function<MulberryResult(Expr *, const Type *)>;

  const llvm::SourceMgr &_sourceManager;
  TypeContext _typeContext;
  Symbols _symbols;
  std::map<std::string, const StructType *> _genericStructTypes;
  std::map<std::string, const FunctionSymbol *> _instantiatedFunctionSymbols;
  std::map<std::string, std::string> _functionPackages;
  std::map<std::string, std::string> _genericFunctionPackages;
  std::map<std::string, std::string> _instantiatedFunctionPackages;
  std::map<std::string, BuiltinHandler, std::less<>> _builtinHandlers;
  VectorUniquePtr<FunctionDecl> _instantiatedFunctions;
  const std::map<std::string, std::string> &_importAliases =
      emptyImportAliases();
  std::string _currentPackageName;
  const Type *_currentFunctionReturnType = nullptr;
  int _whileDepth = 0;

  enum class UnitPolicy {
    Allow,
    Reject,
  };

  // Semantic Analysis

  // Declarations
  auto sema(Decl *node) -> MulberryResult;
  auto sema(Prototype *node) -> MulberryResult;
  auto semaFunctionParameters(Prototype *node,
                              std::vector<const Type *> &parameterTypes,
                              std::vector<bool> &parameterCanMutateObject)
      -> MulberryResult;
  auto bindFunctionParameters(Prototype *node,
                              const FunctionSymbol *signature)
      -> MulberryResult;
  auto semaFunctionSignature(Prototype *node) -> MulberryResult;
  auto sema(FunctionDecl *node) -> MulberryResult;
  auto sema(StructDecl *node) -> MulberryResult;
  auto sema(ComptimeTypeAliasDecl *node) -> MulberryResult;

  // Expressions
  auto sema(Expr *node) -> MulberryResult;
  auto sema(Expr *node, const Type *type) -> MulberryResult;
  auto sema(UnitExpr *node) -> MulberryResult;
  auto sema(BlockExpr *node) -> MulberryResult;
  auto sema(CallExpr *node) -> MulberryResult;
  auto sema(StructLiteralExpr *node) -> MulberryResult;
  auto sema(VariableExpr *node) -> MulberryResult;
  auto sema(MemberExpr *node) -> MulberryResult;
  auto sema(AssignExpr *node) -> MulberryResult;
  auto sema(DecimalLiteralExpr *node) -> MulberryResult;
  auto sema(FloatLiteralExpr *node) -> MulberryResult;
  auto sema(BoolLiteralExpr *node) -> MulberryResult;
  auto sema(StringLiteralExpr *node) -> MulberryResult;
  auto sema(InterpolatedStringExpr *node) -> MulberryResult;
  auto sema(ObjectIdentityExpr *node) -> MulberryResult;
  auto sema(CharLiteralExpr *node) -> MulberryResult;
  auto evaluateComptime(Expr *node) -> ComptimeEvaluation;
  auto evaluateComptimeCall(CallExpr *node) -> ComptimeEvaluation;
  auto evaluateComptimeBinary(BinaryExpr *node) -> ComptimeEvaluation;
  auto comptimeRuntimeType(const ComptimeValue &value) -> const Type *;
  auto setComptimeResultType(Expr *node, const ComptimeValue &value) -> void;
  auto sema(TypeLayoutExpr *node) -> MulberryResult;
  auto sema(HeapAllocExpr *node) -> MulberryResult;
  auto sema(BinaryExpr *node) -> MulberryResult;
  auto checkStringConcatFunction(Expr *node, const Type *stringType)
      -> MulberryResult;
  auto semaFormatValueCall(std::unique_ptr<Expr> &expression,
                           const Type *stringType) -> MulberryResult;
  auto hasMethod(const Type *type, std::string_view methodName) -> bool;
  auto checkAssignable(const Expr *expr) -> MulberryResult;
  auto checkConstObjectUseAsMutable(const Expr *expr) -> MulberryResult;
  auto canMutateObjectReference(const Expr *expr) -> bool;
  auto checkMutableObjectArgument(const FunctionSymbol *signature, size_t index,
                                  const Expr *arg) -> MulberryResult;
  auto sema(ArrayLiteralExpr *expr) -> MulberryResult;
  auto sema(ArrayLiteralExpr *expr, const ArrayType *type) -> MulberryResult;
  auto arrayLiteralTypeWithLeaf(const ArrayLiteralExpr *expr,
                                const Type *leafType) -> const ArrayType *;
  auto semaDefaultArrayLiteral(ArrayLiteralExpr *expr) -> MulberryResult;
  auto semaTensorDisposeCall(CallExpr *node) -> MulberryResult;
  auto semaTensorStorageAllocCall(CallExpr *node, const Type *expectedType)
      -> MulberryResult;
  auto sema(IndexExpr *expr) -> MulberryResult;
  auto semaArrayLiteralElement(Expr *expr, const Type *type)
      -> MulberryResult;
  auto semaGenericCall(CallExpr *node, const GenericFunctionSymbol *symbol,
                       const Type *expectedType = nullptr) -> MulberryResult;
  auto semaMethodCall(CallExpr *node,
                      const Type *expectedType = nullptr) -> MulberryResult;
  auto semaDottedMethodCall(CallExpr *node,
                            const Type *expectedType = nullptr)
      -> MulberryResult;
  auto declareStructMethods(std::string_view ownerName,
                            const VectorUniquePtr<FunctionDecl> &methods,
                            const std::vector<ComptimeParam> &typeParameters,
                            std::string_view packageName) -> MulberryResult;

  // Compiler builtins
  auto registerBuiltinHandlers() -> void;
  auto registerBuiltinHandler(std::string_view name, BuiltinHandler handler)
      -> void;
  auto lookupBuiltinHandler(std::string_view name) const
      -> const BuiltinHandler *;
  auto semaToUInt8(CallExpr *node, const Type *expectedType)
      -> MulberryResult;
  auto semaToUInt64(CallExpr *node, const Type *expectedType)
      -> MulberryResult;
  auto semaToFloat32(CallExpr *node, const Type *expectedType)
      -> MulberryResult;

  // Statements
  auto sema(Stat *node) -> MulberryResult;
  auto sema(VariableStat *node) -> MulberryResult;
  auto sema(ExprStat *node) -> MulberryResult;
  auto sema(IfStat *node) -> MulberryResult;
  auto sema(WhileStat *node) -> MulberryResult;
  auto sema(ForStat *node) -> MulberryResult;
  auto sema(BreakStat *node) -> MulberryResult;
  auto sema(ContinueStat *node) -> MulberryResult;
  auto sema(ReturnStat *node) -> MulberryResult;

  // Errors
  auto emitError(const Node *node, const llvm::Twine &msg) -> MulberryResult {
    _sourceManager.PrintMessage(node->location(),
                                llvm::SourceMgr::DiagKind::DK_Error, msg);
    return failure();
  }

  auto emitError(llvm::SMLoc loc, const llvm::Twine &msg) -> MulberryResult {
    _sourceManager.PrintMessage(loc, llvm::SourceMgr::DiagKind::DK_Error, msg);
    return failure();
  }

  auto isInternalSourceLocation(llvm::SMLoc location) const -> bool {
    if (!location.isValid())
      return false;

    // Imported stdlib declarations keep their own source buffer locations, so
    // this allows internal storage code while rejecting the user's file.
    auto bufferId = _sourceManager.FindBufferContainingLoc(location);
    if (bufferId == 0)
      return false;

    auto path = std::string(
        _sourceManager.getMemoryBuffer(bufferId)->getBufferIdentifier());
    return path.rfind("stdlib/", 0) == 0 ||
           path.find("/stdlib/") != std::string::npos;
  }

  auto checkInternalFeature(llvm::SMLoc location) -> MulberryResult {
    if (isInternalSourceLocation(location))
      return success();
    return emitError(location, diag::internal_only);
  }

  auto lookupVariable(std::string_view name) -> const VariableSymbol * {
    return _symbols.lookupVariable(name);
  }

  auto addBuiltins() -> void {
    declareBuiltinType(BuiltinTypeKind::Unit);
    declareBuiltinType(BuiltinTypeKind::Bool);
    declareBuiltinType(BuiltinTypeKind::UInt8);
    declareBuiltinType(BuiltinTypeKind::UInt64);
    declareBuiltinType(BuiltinTypeKind::Float32);
  }

  auto lookupFunction(std::string_view name) -> const FunctionSymbol * {
    if (auto *signature = _symbols.lookupFunction(name))
      return signature;

    auto importedName = canonicalizeImportedName(name);
    if (auto *signature = _symbols.lookupFunction(importedName))
      return signature;

    return _symbols.lookupFunction(qualifyCurrentPackageName(name));
  }

  auto resolveFunctionName(std::string_view name) -> std::string {
    if (_symbols.lookupFunction(name))
      return std::string(name);

    auto importedName = canonicalizeImportedName(name);
    if (_symbols.lookupFunction(importedName))
      return importedName;

    auto qualifiedName = qualifyCurrentPackageName(name);
    if (_symbols.lookupFunction(qualifiedName))
      return qualifiedName;

    return {};
  }

  auto lookupGenericFunction(std::string_view name)
      -> const GenericFunctionSymbol * {
    if (auto *genericFunction = _symbols.lookupGenericFunction(name))
      return genericFunction;

    auto importedName = canonicalizeImportedName(name);
    if (auto *genericFunction = _symbols.lookupGenericFunction(importedName))
      return genericFunction;

    return _symbols.lookupGenericFunction(qualifyCurrentPackageName(name));
  }

  static auto emptyImportAliases()
      -> const std::map<std::string, std::string> & {
    static const std::map<std::string, std::string> aliases;
    return aliases;
  }

  auto canonicalizeImportedName(std::string_view name) const -> std::string {
    auto importedName = _importAliases.find(std::string(name));
    if (importedName != _importAliases.end())
      return importedName->second;

    auto dot = name.find('.');
    if (dot == std::string_view::npos)
      return std::string(name);

    auto alias = _importAliases.find(std::string(name.substr(0, dot)));
    if (alias == _importAliases.end())
      return std::string(name);

    std::string fullName = alias->second;
    fullName += ".";
    fullName += name.substr(dot + 1);
    return fullName;
  }

  auto qualifyCurrentPackageName(std::string_view name) const -> std::string {
    if (_currentPackageName.empty() ||
        name.find('.') != std::string_view::npos)
      return std::string(name);

    std::string fullName = _currentPackageName;
    fullName += ".";
    fullName += name;
    return fullName;
  }

  auto lookupType(std::string_view name) -> const Type * {
    if (auto *type = _symbols.lookupType(name))
      return type;

    auto importedName = canonicalizeImportedName(name);
    if (auto *type = _symbols.lookupType(importedName))
      return type;

    return _symbols.lookupType(qualifyCurrentPackageName(name));
  }

  auto lookupStructType(std::string_view name) -> const StructType * {
    return mulberry::getStructType(lookupType(name));
  }

  auto comptimeTypeAliasName(std::string_view name) -> std::string {
    if (_symbols.lookupComptimeTypeAlias(name))
      return std::string(name);

    auto importedName = canonicalizeImportedName(name);
    if (_symbols.lookupComptimeTypeAlias(importedName))
      return importedName;

    auto packageName = qualifyCurrentPackageName(name);
    if (_symbols.lookupComptimeTypeAlias(packageName))
      return packageName;

    return {};
  }

  auto lookupComptimeTypeAlias(std::string_view name)
      -> const ComptimeTypeAliasSymbol * {
    auto aliasName = comptimeTypeAliasName(name);
    if (aliasName.empty())
      return nullptr;
    return _symbols.lookupComptimeTypeAlias(aliasName);
  }

  auto resolveType(const NamedTypeNode *typeNode) -> const Type * {
    auto *type = lookupType(typeNode->name());
    if (!type) {
      emitError(typeNode, diag::undefined_type);
      return nullptr;
    }

    return type;
  }

  auto resolveType(const UnitTypeNode *typeNode) -> const Type * {
    return _typeContext.getBuiltinType(BuiltinTypeKind::Unit);
  }

  auto declareVariable(std::string_view name, const Type *type,
                       bool isConstBinding = false,
                       bool canMutateObject = true,
                       std::optional<ComptimeValue> comptimeValue = std::nullopt,
                       bool isComptimeOnly = false)
      -> MulberryResult {
    return _symbols.declareVariable(name, type, isConstBinding,
                                    canMutateObject, std::move(comptimeValue),
                                    isComptimeOnly);
  }

  class VariableScope {
  public:
    explicit VariableScope(Symbols &symbols) : _symbols(symbols) {
      _symbols.enterVariableScope();
    }

    ~VariableScope() { _symbols.leaveVariableScope(); }

  private:
    Symbols &_symbols;
  };

  class PackageScope {
  public:
    PackageScope(std::string &currentPackageName, std::string_view packageName)
        : _currentPackageName(currentPackageName),
          _oldPackageName(currentPackageName) {
      _currentPackageName = packageName;
    }

    ~PackageScope() { _currentPackageName = _oldPackageName; }

  private:
    std::string &_currentPackageName;
    std::string _oldPackageName;
  };

  class WhileScope {
  public:
    explicit WhileScope(int &whileDepth) : _whileDepth(whileDepth) {
      ++_whileDepth;
    }

    ~WhileScope() { --_whileDepth; }

  private:
    int &_whileDepth;
  };

  class FunctionReturnTypeScope {
  public:
    FunctionReturnTypeScope(const Type *&currentFunctionReturnType,
                            const Type *returnType)
        : _currentFunctionReturnType(currentFunctionReturnType),
          _oldFunctionReturnType(currentFunctionReturnType) {
      _currentFunctionReturnType = returnType;
    }

    ~FunctionReturnTypeScope() {
      _currentFunctionReturnType = _oldFunctionReturnType;
    }

  private:
    const Type *&_currentFunctionReturnType;
    const Type *_oldFunctionReturnType;
  };

  auto declareFunction(std::string_view name,
                       std::vector<const Type *> parameterTypes,
                       std::vector<bool> parameterCanMutateObject,
                       const Type *returnType,
                       std::string_view packageName = {})
      -> MulberryResult {
    if (packageName.empty())
      packageName = _currentPackageName;
    _functionPackages[std::string(name)] = std::string(packageName);
    return _symbols.declareFunction(name, std::move(parameterTypes),
                                    std::move(parameterCanMutateObject),
                                    returnType);
  }

  auto declareGenericFunction(std::string_view name,
                              const FunctionDecl *decl,
                              std::string_view packageName = {})
      -> MulberryResult {
    if (packageName.empty())
      packageName = _currentPackageName;
    _genericFunctionPackages[std::string(name)] = std::string(packageName);
    return _symbols.declareGenericFunction(name, decl);
  }

  auto declareType(std::string_view name, const Type *type) -> MulberryResult {
    return _symbols.declareType(name, type);
  }

  auto declareBuiltinType(BuiltinTypeKind kind) -> const BuiltinType * {
    auto *type = _typeContext.getBuiltinType(kind);
    declareType(type->name(), type);
    return type;
  }

  auto declareStructType(const StructType *type) -> MulberryResult {
    return declareType(type->name(), type);
  }

  static auto mangleTypeName(std::string name) -> std::string {
    for (auto &character : name) {
      if ((character >= 'a' && character <= 'z') ||
          (character >= 'A' && character <= 'Z') ||
          (character >= '0' && character <= '9'))
        continue;
      character = '_';
    }
    return name;
  }

  auto genericStructName(std::string_view aliasName,
                         const std::vector<ComptimeArgument> &arguments) const
      -> std::string {
    std::string name = mangleTypeName(std::string(aliasName));
    for (auto &argument : arguments) {
      name += "__";
      if (argument.kind == ComptimeArg::Kind::Type)
        name += mangleTypeName(formatType(argument.type));
      else
        name += std::to_string(argument.uint64Value);
    }
    return name;
  }

  auto genericFunctionName(std::string_view name,
                           const Type *argumentType) const -> std::string {
    std::string result = mangleTypeName(std::string(name));
    result += "__";
    result += mangleTypeName(formatType(argumentType));
    return result;
  }

  auto genericFunctionName(
      std::string_view name,
      const std::vector<InferredComptimeArgument> &arguments) const
      -> std::string {
    std::string result = mangleTypeName(std::string(name));
    for (auto &argument : arguments) {
      result += "__";
      if (argument.kind == ComptimeParam::Kind::Type)
        result += mangleTypeName(formatType(argument.type));
      else
        result += std::to_string(*argument.uint64Value);
    }
    return result;
  }

  auto makeInferredComptimeArguments(
      const std::vector<ComptimeParam> &parameters) const
      -> std::vector<InferredComptimeArgument> {
    std::vector<InferredComptimeArgument> arguments;
    for (auto &parameter : parameters) {
      InferredComptimeArgument argument;
      argument.kind = parameter.kind;
      arguments.push_back(std::move(argument));
    }
    return arguments;
  }

  auto comptimeSubstitutions(
      const std::vector<ComptimeParam> &parameters,
      const std::vector<InferredComptimeArgument> &arguments) const
      -> std::vector<TypeSubstitution> {
    std::vector<TypeSubstitution> substitutions;
    for (size_t i = 0; i < parameters.size(); ++i) {
      auto &parameter = parameters[i];
      auto &argument = arguments[i];
      if (parameter.kind == ComptimeParam::Kind::Type) {
        substitutions.push_back(TypeSubstitution{
            parameter.name, argument.typeNode.get(), std::nullopt});
        continue;
      }
      substitutions.push_back(TypeSubstitution{
          parameter.name, nullptr, *argument.uint64Value});
    }
    return substitutions;
  }

  auto comptimeParameterIndex(const std::vector<ComptimeParam> &parameters,
                              std::string_view name) const
      -> std::optional<size_t> {
    for (size_t i = 0; i < parameters.size(); ++i)
      if (parameters[i].name == name)
        return i;
    return std::nullopt;
  }

  auto hasComputedType(const TypeNode *typeNode,
                       NameSet &visitingAliases) -> bool {
    if (llvm::isa<ComputedTypeNode>(typeNode))
      return true;

    if (auto *arrayType = dyn_cast<ArrayTypeNode>(typeNode))
      return hasComputedType(arrayType->elementTypeNode(), visitingAliases);

    if (auto *ptrType = dyn_cast<PtrTypeNode>(typeNode))
      return hasComputedType(ptrType->pointeeTypeNode(), visitingAliases);

    if (auto *genericType = dyn_cast<GenericTypeNode>(typeNode)) {
      for (auto &argument : genericType->arguments()) {
        if (argument.kind() == ComptimeArg::Kind::Type &&
            hasComputedType(argument.typeNode(), visitingAliases))
          return true;
      }

      auto aliasName = comptimeTypeAliasName(genericType->name());
      if (aliasName.empty())
        return false;
      auto [iter, inserted] = visitingAliases.insert(aliasName);
      if (!inserted)
        return false;

      auto *symbol = _symbols.lookupComptimeTypeAlias(aliasName);
      PackageScope packageScope(_currentPackageName, symbol->packageName);
      auto result = hasComputedType(symbol->bodyTypeNode, visitingAliases);
      visitingAliases.erase(iter);
      return result;
    }

    if (auto *structType = dyn_cast<StructTypeNode>(typeNode)) {
      for (auto &field : structType->fields())
        if (hasComputedType(field->typeNode(), visitingAliases))
          return true;
    }

    return false;
  }

  auto hasComputedType(const TypeNode *typeNode) -> bool {
    NameSet visitingAliases;
    return hasComputedType(typeNode, visitingAliases);
  }

  auto bindComptimeTypeArgument(
      const Type *type, InferredComptimeArgument &argument,
      llvm::SMLoc location) -> bool {
    if (argument.kind != ComptimeParam::Kind::Type)
      return false;
    if (!argument.type) {
      argument.type = type;
      argument.typeNode = typeToTypeNode(type, location);
      return true;
    }
    return sameType(argument.type, type);
  }

  auto bindComptimeUInt64Argument(uint64_t value,
                                  InferredComptimeArgument &argument) -> bool {
    if (argument.kind != ComptimeParam::Kind::UInt64)
      return false;
    if (!argument.uint64Value) {
      argument.uint64Value = value;
      return true;
    }
    return *argument.uint64Value == value;
  }

  auto computedArrayLeafParameterIndex(
      const TypeNode *pattern,
      const std::vector<ComptimeParam> &parameters) const
      -> std::optional<size_t> {
    auto *computedType = dyn_cast<ComputedTypeNode>(pattern);
    auto *call = computedType
                     ? dyn_cast<CallExpr>(computedType->expression().get())
                     : nullptr;
    if (!call || call->name() != "arrayLeafElementType" ||
        !call->hasReceiver() || !call->expressions().empty())
      return std::nullopt;

    auto *typeInfo = dyn_cast<TypeInfoExpr>(call->receiver().get());
    auto *namedType = typeInfo
                          ? dyn_cast<NamedTypeNode>(typeInfo->typeNode())
                          : nullptr;
    if (!namedType)
      return std::nullopt;
    auto index = comptimeParameterIndex(parameters, namedType->name());
    if (!index || parameters[*index].kind != ComptimeParam::Kind::Type)
      return std::nullopt;
    return index;
  }

  auto matchComptimeArgument(
      const ComptimeArg &pattern, const ComptimeValue &actual,
      const std::vector<ComptimeParam> &parameters,
      std::vector<InferredComptimeArgument> &arguments,
      std::vector<const Type *> *arrayLeafConstraints = nullptr) -> bool {
    if (pattern.kind() == ComptimeArg::Kind::UInt64)
      return actual.kind() == ComptimeValue::Kind::UInt64 &&
             pattern.uint64Value() == actual.uint64Value();

    if (auto *namedType = dyn_cast<NamedTypeNode>(pattern.typeNode())) {
      if (auto index = comptimeParameterIndex(parameters, namedType->name())) {
        if (actual.kind() == ComptimeValue::Kind::Type)
          return bindComptimeTypeArgument(
              actual.type(), arguments[*index], namedType->location());
        return bindComptimeUInt64Argument(actual.uint64Value(),
                                          arguments[*index]);
      }
    }

    if (actual.kind() != ComptimeValue::Kind::Type)
      return false;
    return matchGenericType(pattern.typeNode(), actual.type(), parameters,
                            arguments, arrayLeafConstraints);
  }

  auto matchGenericType(const TypeNode *pattern, const Type *actualType,
                        const std::vector<ComptimeParam> &parameters,
                        std::vector<InferredComptimeArgument> &arguments,
                        std::vector<const Type *> *arrayLeafConstraints =
                            nullptr)
      -> bool {
    if (auto index = computedArrayLeafParameterIndex(pattern, parameters)) {
      if (!arrayLeafConstraints)
        return false;
      auto *&constraint = (*arrayLeafConstraints)[*index];
      if (!constraint) {
        constraint = actualType;
        return true;
      }
      return sameType(constraint, actualType);
    }

    if (llvm::isa<ComputedTypeNode>(pattern))
      return false;

    if (llvm::isa<UnitTypeNode>(pattern))
      return isUnitType(actualType);

    if (auto *namedType = dyn_cast<NamedTypeNode>(pattern)) {
      if (auto index = comptimeParameterIndex(parameters, namedType->name()))
        return bindComptimeTypeArgument(actualType, arguments[*index],
                                        namedType->location());

      auto *patternType = resolveType(namedType);
      return sameType(patternType, actualType);
    }

    if (auto *arrayPattern = dyn_cast<ArrayTypeNode>(pattern)) {
      auto *arrayType = getArrayType(actualType);
      return arrayType && arrayPattern->shape().size() == 1 &&
             arrayPattern->shape().front() >= 0 &&
             static_cast<uint64_t>(arrayPattern->shape().front()) ==
                 arrayType->size() &&
             matchGenericType(arrayPattern->elementTypeNode(),
                              arrayType->elementType(), parameters,
                              arguments, arrayLeafConstraints);
    }

    if (auto *ptrPattern = dyn_cast<PtrTypeNode>(pattern)) {
      auto *ptrType = getPtrType(actualType);
      return ptrType &&
             matchGenericType(ptrPattern->pointeeTypeNode(),
                              ptrType->pointeeType(), parameters, arguments,
                              arrayLeafConstraints);
    }

    if (auto *genericPattern = dyn_cast<GenericTypeNode>(pattern)) {
      if (genericPattern->name() == "Array") {
        auto *arrayType = getArrayType(actualType);
        auto &patternArguments = genericPattern->arguments();
        if (!arrayType || patternArguments.size() != 2 ||
            patternArguments[0].kind() != ComptimeArg::Kind::Type)
          return false;
        if (!matchGenericType(patternArguments[0].typeNode(),
                              arrayType->elementType(), parameters,
                              arguments, arrayLeafConstraints))
          return false;
        auto sizeValue = ComptimeValue(arrayType->size());
        return matchComptimeArgument(patternArguments[1], sizeValue,
                                     parameters, arguments,
                                     arrayLeafConstraints);
      }

      auto aliasName = comptimeTypeAliasName(genericPattern->name());
      auto *structType = getStructType(actualType);
      auto *origin = structType ? structType->origin() : nullptr;
      auto &patternArguments = genericPattern->arguments();
      if (origin && origin->aliasName() == aliasName) {
        auto &actualArguments = origin->arguments();
        if (patternArguments.size() != actualArguments.size())
          return false;
        for (size_t i = 0; i < patternArguments.size(); ++i)
          if (!matchComptimeArgument(patternArguments[i], actualArguments[i],
                                     parameters, arguments,
                                     arrayLeafConstraints))
            return false;
        return true;
      }

      auto *alias = lookupComptimeTypeAlias(genericPattern->name());
      if (!alias || patternArguments.size() != alias->parameters.size())
        return false;

      std::vector<TypeSubstitution> substitutions;
      for (size_t i = 0; i < patternArguments.size(); ++i) {
        auto &patternArgument = patternArguments[i];
        auto &aliasParameter = alias->parameters[i];
        if (patternArgument.kind() == ComptimeArg::Kind::UInt64) {
          if (aliasParameter.kind != ComptimeParam::Kind::UInt64)
            return false;
          substitutions.push_back(TypeSubstitution{
              aliasParameter.name, nullptr, patternArgument.uint64Value()});
          continue;
        }

        if (aliasParameter.kind != ComptimeParam::Kind::Type)
          return false;

        auto *argumentTypeNode = patternArgument.typeNode();
        if (containsComptimeParameter(argumentTypeNode, parameters)) {
          substitutions.push_back(TypeSubstitution{
              aliasParameter.name, argumentTypeNode, std::nullopt});
          continue;
        }

        auto *argumentType = resolveType(argumentTypeNode);
        if (!argumentType)
          return false;
        auto resolvedArgumentTypeNode =
            typeToTypeNode(argumentType, argumentTypeNode->location());
        substitutions.push_back(TypeSubstitution{
            aliasParameter.name, resolvedArgumentTypeNode.get(),
            std::nullopt});
      }

      // Expand the alias in its defining package before matching its body.
      // This lets generic methods infer T from aliases such as `List<T>`.
      auto aliasBody = substituteTypeNode(alias->bodyTypeNode, substitutions);
      PackageScope packageScope(_currentPackageName, alias->packageName);
      return matchGenericType(aliasBody.get(), actualType, parameters,
                              arguments, arrayLeafConstraints);
    }

    if (auto *structPattern = dyn_cast<StructTypeNode>(pattern)) {
      auto *structType = getStructType(actualType);
      if (!structType)
        return false;
      auto &patternFields = structPattern->fields();
      auto &actualFields = structType->fields();
      if (patternFields.size() != actualFields.size())
        return false;

      for (size_t i = 0; i < patternFields.size(); ++i) {
        if (patternFields[i]->variable()->name() != actualFields[i].name())
          return false;
        if (!matchGenericType(patternFields[i]->typeNode(),
                              actualFields[i].type(), parameters, arguments,
                              arrayLeafConstraints))
          return false;
      }
      return true;
    }

    auto *patternType = resolveType(pattern);
    return sameType(patternType, actualType);
  }

  auto matchMethodReceiverType(
      const TypeNode *pattern, const Type *actualType,
      const std::vector<ComptimeParam> &parameters,
      std::vector<InferredComptimeArgument> &arguments) -> bool {
    if (auto *ptrPattern = dyn_cast<PtrTypeNode>(pattern)) {
      if (matchGenericType(pattern, actualType, parameters, arguments))
        return true;
      return matchGenericType(ptrPattern->pointeeTypeNode(), actualType,
                              parameters, arguments);
    }
    if (auto *ptrType = getPtrType(actualType))
      if (matchGenericType(pattern, ptrType->pointeeType(), parameters,
                           arguments))
        return true;
    return matchGenericType(pattern, actualType, parameters, arguments);
  }

  auto sameCallArgumentType(const Type *parameterType, const Type *actualType,
                            bool allowAddressOf) -> bool {
    if (sameType(parameterType, actualType))
      return true;
    if (!allowAddressOf)
      return false;

    auto *ptrType = getPtrType(parameterType);
    return ptrType && sameType(ptrType->pointeeType(), actualType);
  }

  auto sameReturnType(const Type *returnType, const Type *actualType) -> bool {
    if (sameType(returnType, actualType))
      return true;

    // Pointer reinterpretation stays explicit in stdlib/internal source
    // through helpers such as std.internal.ptr.asUInt8<T>(). The helper's body
    // returns Ptr<T>; MLIRGen materializes the declared Ptr<UInt8> return with
    // mulberry_core.ptr.cast.
    return getPtrType(returnType) && getPtrType(actualType);
  }

  auto instantiateGenericFunction(const Node *diagnosticNode,
                                  std::string_view name,
                                  const Type *argumentType,
                                  std::string &concreteName)
      -> MulberryResult {
    auto *symbol = lookupGenericFunction(name);
    if (!symbol) {
      auto diagnostic = formatNameDiagnostic(diag::undefined_func, name);
      return emitError(diagnosticNode, diagnostic);
    }

    auto *genericFunction = symbol->decl;
    auto *genericProto = genericFunction->proto().get();
    auto &genericParameters = genericProto->comptimeParameters();
    if (genericParameters.size() != 1 ||
        genericParameters.front().kind != ComptimeParam::Kind::Type)
      return emitError(diagnosticNode, diag::mismatch_type);

    concreteName = genericFunctionName(name, argumentType);

    auto cached = _instantiatedFunctionSymbols.find(concreteName);
    if (cached != _instantiatedFunctionSymbols.end())
      return success();

    auto argumentTypeNode =
        typeToTypeNode(argumentType, genericProto->location());
    auto concreteFunction = instantiateFunctionDecl(
        genericFunction, concreteName,
        std::vector<TypeSubstitution>{
            TypeSubstitution{genericParameters.front().name,
                             argumentTypeNode.get(), std::nullopt}});
    _instantiatedFunctionPackages[concreteName] =
        genericFunctionPackageName(name);

    VariableScope signatureScope(_symbols);
    PackageScope packageScope(_currentPackageName,
                              genericFunctionPackageName(name));
    if (semaFunctionSignature(concreteFunction->proto().get()))
      return failure();
    auto *signature = lookupFunction(concreteName);
    if (!signature)
      return failure();

    _instantiatedFunctionSymbols.insert({concreteName, signature});
    _instantiatedFunctions.push_back(std::move(concreteFunction));
    return success();
  }

  auto resolveSubstitutedType(const TypeNode *typeNode,
                              const std::vector<TypeSubstitution> &substitutions)
      -> const Type * {
    auto substitutedTypeNode = substituteTypeNode(typeNode, substitutions);
    return resolveType(substitutedTypeNode.get());
  }

  auto functionPackageName(std::string_view name) const -> std::string {
    auto package = _instantiatedFunctionPackages.find(std::string(name));
    if (package != _instantiatedFunctionPackages.end())
      return package->second;
    package = _functionPackages.find(std::string(name));
    if (package != _functionPackages.end())
      return package->second;
    package = _genericFunctionPackages.find(std::string(name));
    if (package != _genericFunctionPackages.end())
      return package->second;
    return packageNameOf(name);
  }

  auto genericFunctionPackageName(std::string_view name) const -> std::string {
    auto package = _genericFunctionPackages.find(std::string(name));
    if (package != _genericFunctionPackages.end())
      return package->second;
    return packageNameOf(name);
  }

  auto rejectUnitType(const TypeNode *typeNode, const Type *type)
      -> MulberryResult {
    if (!isUnitType(type))
      return success();

    return emitError(typeNode, diag::unexpected_unit_type);
  }

  auto rejectUnitElementType(const TypeNode *typeNode, const Type *type)
      -> MulberryResult {
    if (!hasUnitElementType(type))
      return success();

    return emitError(typeNode, diag::unexpected_unit_type);
  }

  auto resolveType(const ArrayTypeNode *typeNode) -> const Type * {
    auto *elementType = resolveType(typeNode->elementTypeNode());
    if (!elementType)
      return nullptr;

    auto &shape = typeNode->shape();
    if (shape.size() == 1 && shape.front() >= 0) {
      if (!isArrayElementType(elementType)) {
        emitError(typeNode->elementTypeNode(), diag::mismatch_type);
        return nullptr;
      }
      return _typeContext.createArrayType(elementType, shape.front());
    }

    emitError(typeNode, diag::mismatch_type);
    return nullptr;
  }

  auto resolveType(const PtrTypeNode *typeNode) -> const Type * {
    if (checkInternalFeature(typeNode->location()))
      return nullptr;

    auto *pointeeType = resolveType(typeNode->pointeeTypeNode());
    if (!pointeeType)
      return nullptr;

    return _typeContext.createPtrType(pointeeType);
  }

  auto resolveStructFields(const StructTypeNode *typeNode)
      -> std::optional<std::vector<StructField>> {
    std::vector<StructField> fields;
    NameSet fieldNames;
    unsigned fieldIndex = 0;
    for (auto &fieldDecl : typeNode->fields()) {
      auto variable = fieldDecl->variable().get();
      auto *fieldType = checkType(fieldDecl->typeNode(), UnitPolicy::Reject);
      if (!fieldType)
        return std::nullopt;
      fieldDecl->setType(fieldType);
      auto fieldName = variable->name();
      if (!declareName(fieldNames, fieldName)) {
        emitError(variable, diag::redefinition_var);
        return std::nullopt;
      }
      fields.push_back(StructField{fieldName, fieldType, fieldIndex++});
    }
    return fields;
  }

  auto resolveType(const StructTypeNode *typeNode, std::string_view name)
      -> const StructType * {
    auto fields = resolveStructFields(typeNode);
    if (!fields)
      return nullptr;
    return _typeContext.createStructType(name, std::move(*fields));
  }

  auto resolveType(const StructTypeNode *typeNode, std::string_view name,
                   ComptimeAliasOrigin origin) -> const StructType * {
    auto fields = resolveStructFields(typeNode);
    if (!fields)
      return nullptr;
    return _typeContext.createStructType(name, std::move(*fields),
                                         std::move(origin));
  }

  auto resolveType(const ComputedTypeNode *typeNode) -> const Type * {
    auto result = evaluateComptime(typeNode->expression().get());
    if (result.kind == ComptimeEvaluation::Kind::Error)
      return nullptr;
    if (result.kind != ComptimeEvaluation::Kind::Value ||
        result.value->kind() != ComptimeValue::Kind::Type) {
      emitError(typeNode, diag::expected_comptime_type);
      return nullptr;
    }

    auto *type = result.value->type();
    LLVM_DEBUG(llvm::dbgs() << "resolved computed type `"
                            << formatType(type) << "`\n");
    return type;
  }

  auto resolveType(const GenericTypeNode *typeNode) -> const Type * {
    if (typeNode->name() == "Array") {
      auto &arguments = typeNode->arguments();
      if (arguments.size() != 2 ||
          arguments[0].kind() != ComptimeArg::Kind::Type ||
          arguments[1].kind() != ComptimeArg::Kind::UInt64) {
        emitError(typeNode, diag::mismatch_type);
        return nullptr;
      }

      auto *elementType = resolveType(arguments[0].typeNode());
      if (!elementType)
        return nullptr;
      if (!isArrayElementType(elementType)) {
        emitError(arguments[0].typeNode(), diag::mismatch_type);
        return nullptr;
      }
      return _typeContext.createArrayType(elementType,
                                          arguments[1].uint64Value());
    }

    auto aliasName = comptimeTypeAliasName(typeNode->name());
    auto *alias = aliasName.empty()
                      ? nullptr
                      : _symbols.lookupComptimeTypeAlias(aliasName);
    if (!alias) {
      emitError(typeNode, diag::undefined_type);
      return nullptr;
    }
    if (typeNode->arguments().size() != alias->parameters.size()) {
      emitError(typeNode, diag::mismatch_type);
      return nullptr;
    }

    std::vector<ComptimeArgument> arguments;
    std::vector<TypeSubstitution> substitutions;
    for (size_t i = 0; i < typeNode->arguments().size(); ++i) {
      auto &argument = typeNode->arguments()[i];
      auto &parameter = alias->parameters[i];
      if (argument.kind() == ComptimeArg::Kind::Type &&
          parameter.kind == ComptimeParam::Kind::Type) {
        auto *argumentType = resolveType(argument.typeNode());
        if (!argumentType)
          return nullptr;

        auto argumentTypeNode =
            typeToTypeNode(argumentType, argument.typeNode()->location());
        substitutions.push_back(
            TypeSubstitution{parameter.name, argumentTypeNode.get(),
                             std::nullopt});
        ComptimeArgument resolvedArgument;
        resolvedArgument.kind = argument.kind();
        resolvedArgument.type = argumentType;
        resolvedArgument.typeNode = std::move(argumentTypeNode);
        arguments.push_back(std::move(resolvedArgument));
        continue;
      }

      if (argument.kind() == ComptimeArg::Kind::UInt64 &&
          parameter.kind == ComptimeParam::Kind::UInt64) {
        substitutions.push_back(
            TypeSubstitution{parameter.name, nullptr,
                             argument.uint64Value()});
        ComptimeArgument resolvedArgument;
        resolvedArgument.kind = argument.kind();
        resolvedArgument.uint64Value = argument.uint64Value();
        arguments.push_back(std::move(resolvedArgument));
        continue;
      }

      emitError(typeNode, diag::mismatch_type);
      return nullptr;
    }

    // Generic arguments belong to the use site, while the alias body belongs to
    // the definition package. Resolve arguments first, then switch package
    // scope for the substituted alias body.
    auto instantiatedTypeNode =
        substituteTypeNode(alias->bodyTypeNode, substitutions);
    PackageScope packageScope(_currentPackageName, alias->packageName);
    if (auto *structTypeNode =
            dyn_cast<StructTypeNode>(instantiatedTypeNode.get())) {
      auto structName = genericStructName(aliasName, arguments);
      auto cached = _genericStructTypes.find(structName);
      if (cached != _genericStructTypes.end())
        return cached->second;

      auto origin = ComptimeAliasOrigin(
          aliasName, toComptimeValues(arguments));
      auto *structType = resolveType(structTypeNode, structName,
                                     std::move(origin));
      if (!structType)
        return nullptr;
      _genericStructTypes[structName] = structType;
      return structType;
    }

    return resolveType(instantiatedTypeNode.get());
  }

  auto resolveType(const TypeNode *typeNode) -> const Type * {
    if (auto *unitType = dyn_cast<UnitTypeNode>(typeNode))
      return resolveType(unitType);

    if (auto *arrayType = dyn_cast<ArrayTypeNode>(typeNode))
      return resolveType(arrayType);

    if (auto *computedType = dyn_cast<ComputedTypeNode>(typeNode))
      return resolveType(computedType);

    if (auto *ptrType = dyn_cast<PtrTypeNode>(typeNode))
      return resolveType(ptrType);

    if (auto *genericType = dyn_cast<GenericTypeNode>(typeNode))
      return resolveType(genericType);

    if (auto *structType = dyn_cast<StructTypeNode>(typeNode))
      return resolveType(structType, "$anonymous");

    return resolveType(cast<NamedTypeNode>(typeNode));
  }

  auto checkType(const TypeNode *typeNode, UnitPolicy unitPolicy)
      -> const Type * {
    auto *type = resolveType(typeNode);
    if (!type)
      return nullptr;
    if (unitPolicy == UnitPolicy::Reject && rejectUnitType(typeNode, type))
      return nullptr;
    if (rejectUnitElementType(typeNode, type))
      return nullptr;
    return type;
  }

  auto setBuiltinType(Expr *expr, BuiltinTypeKind kind) -> void {
    expr->setType(_typeContext.getBuiltinType(kind));
  }

  static auto stdlibListElementType(const Type *type) -> const Type * {
    auto *structType = mulberry::getStructType(type);
    if (!structType)
      return nullptr;

    auto *origin = structType->origin();
    if (!origin || origin->aliasName() != "std.list.List")
      return nullptr;

    auto &fields = structType->fields();
    if (fields.size() != 3)
      return nullptr;
    if (fields[0].name() != "length" || !isUInt64Type(fields[0].type()))
      return nullptr;
    if (fields[1].name() != "capacity" || !isUInt64Type(fields[1].type()))
      return nullptr;
    if (fields[2].name() != "data")
      return nullptr;

    auto *dataPtrType = mulberry::getPtrType(fields[2].type());
    if (!dataPtrType)
      return nullptr;
    return dataPtrType->pointeeType();
  }

  auto tensorElementType(const Type *type) -> const Type * {
    auto *structType = mulberry::getStructType(type);
    if (!structType)
      return nullptr;

    auto *origin = structType->origin();
    if (!origin || origin->aliasName() != "std.tensor.Tensor")
      return nullptr;

    auto &arguments = origin->arguments();
    if (arguments.size() != 1)
      return nullptr;
    if (arguments[0].kind() != ComptimeValue::Kind::Type)
      return nullptr;
    return arguments[0].type();
  }

  auto tensorStorageElementType(const Type *type) -> const Type * {
    auto *structType = mulberry::getStructType(type);
    if (!structType)
      return nullptr;

    auto *origin = structType->origin();
    if (!origin || origin->aliasName() != "std.tensor.TensorStorage")
      return nullptr;

    auto &arguments = origin->arguments();
    if (arguments.size() != 1 ||
        arguments[0].kind() != ComptimeValue::Kind::Type)
      return nullptr;
    return arguments[0].type();
  }
};

} // end namespace

auto SemaImpl::registerBuiltinHandlers() -> void {
  registerBuiltinHandler(
      "std.tensor.__allocate",
      [this](Expr *node, const Type *expectedType) {
        return semaTensorStorageAllocCall(cast<CallExpr>(node), expectedType);
      });
  registerBuiltinHandler(
      "std.tensor.__dispose",
      [this](Expr *node, const Type *) {
        return semaTensorDisposeCall(cast<CallExpr>(node));
      });
  registerBuiltinHandler(
      "std.core.toUInt8",
      [this](Expr *node, const Type *expectedType) {
        return semaToUInt8(cast<CallExpr>(node), expectedType);
      });
  registerBuiltinHandler(
      "std.core.toUInt64",
      [this](Expr *node, const Type *expectedType) {
        return semaToUInt64(cast<CallExpr>(node), expectedType);
      });
  registerBuiltinHandler(
      "std.core.toFloat32",
      [this](Expr *node, const Type *expectedType) {
        return semaToFloat32(cast<CallExpr>(node), expectedType);
      });
}

auto SemaImpl::registerBuiltinHandler(std::string_view name,
                                      BuiltinHandler handler) -> void {
  auto [iter, inserted] =
      _builtinHandlers.try_emplace(std::string(name), std::move(handler));
  if (!inserted)
    llvm_unreachable("duplicate builtin Sema handler");
  LLVM_DEBUG(llvm::dbgs() << "register builtin Sema handler `" << iter->first
                          << "`\n");
}

auto SemaImpl::lookupBuiltinHandler(std::string_view name) const
    -> const BuiltinHandler * {
  auto iter = _builtinHandlers.find(name);
  if (iter == _builtinHandlers.end())
    return nullptr;
  return &iter->second;
}

auto SemaImpl::semaToUInt8(CallExpr *node, const Type *expectedType)
    -> MulberryResult {
  auto &arguments = node->expressions();
  if (arguments.size() != 1) {
    auto diagnostic =
        formatNameSizeDiagnostic(diag::func_param, node->name(), 1);
    return emitError(node, diagnostic);
  }

  auto *parameterType =
      _typeContext.getBuiltinType(BuiltinTypeKind::UInt64);
  if (sema(arguments.front().get(), parameterType))
    return failure();
  if (!sameType(arguments.front()->type(), parameterType))
    return emitError(arguments.front().get(), diag::mismatch_type);

  auto *resultType = _typeContext.getBuiltinType(BuiltinTypeKind::UInt8);
  if (expectedType && !sameType(expectedType, resultType))
    return emitError(node, diag::mismatch_type);
  node->setType(resultType);
  return success();
}

auto SemaImpl::semaToUInt64(CallExpr *node, const Type *expectedType)
    -> MulberryResult {
  auto &arguments = node->expressions();
  if (arguments.size() != 1) {
    auto diagnostic =
        formatNameSizeDiagnostic(diag::func_param, node->name(), 1);
    return emitError(node, diagnostic);
  }

  auto *parameterType =
      _typeContext.getBuiltinType(BuiltinTypeKind::UInt8);
  if (sema(arguments.front().get(), parameterType))
    return failure();
  if (!sameType(arguments.front()->type(), parameterType))
    return emitError(arguments.front().get(), diag::mismatch_type);

  auto *resultType = _typeContext.getBuiltinType(BuiltinTypeKind::UInt64);
  if (expectedType && !sameType(expectedType, resultType))
    return emitError(node, diag::mismatch_type);
  node->setType(resultType);
  return success();
}

auto SemaImpl::semaToFloat32(CallExpr *node, const Type *expectedType)
    -> MulberryResult {
  auto &arguments = node->expressions();
  if (arguments.size() != 1) {
    auto diagnostic =
        formatNameSizeDiagnostic(diag::func_param, node->name(), 1);
    return emitError(node, diagnostic);
  }

  auto *parameterType =
      _typeContext.getBuiltinType(BuiltinTypeKind::UInt64);
  if (sema(arguments.front().get(), parameterType))
    return failure();
  if (!sameType(arguments.front()->type(), parameterType))
    return emitError(arguments.front().get(), diag::mismatch_type);

  auto *resultType = _typeContext.getBuiltinType(BuiltinTypeKind::Float32);
  if (expectedType && !sameType(expectedType, resultType))
    return emitError(node, diag::mismatch_type);
  node->setType(resultType);
  return success();
}

auto SemaImpl::sema(Decl *node) -> MulberryResult {
  switch (node->getKind()) {
  case Decl::Decl_Import:
    return success();
  case Decl::Decl_Function:
    return sema(cast<FunctionDecl>(node));
  case Decl::Decl_Struct:
    return sema(cast<StructDecl>(node));
  case Decl::Decl_ComptimeTypeAlias:
    return sema(cast<ComptimeTypeAliasDecl>(node));
  }
}

auto SemaImpl::sema(Prototype *node) -> MulberryResult {
  if (node->isGeneric())
    return success();

  return semaFunctionSignature(node);
}

auto SemaImpl::semaFunctionParameters(
    Prototype *node, std::vector<const Type *> &parameterTypes,
    std::vector<bool> &parameterCanMutateObject)
    -> MulberryResult {
  for (const auto &indexedParameter : llvm::enumerate(node->parameters())) {
    auto &par = indexedParameter.value();
    auto *parameterType = checkType(par->typeNode(), UnitPolicy::Reject);
    if (!parameterType)
      return failure();
    par->setType(parameterType);
    auto canMutateObject = par->canMutateObject();
    if (declareVariable(par->variable()->name(), parameterType,
                        !canMutateObject, canMutateObject))
      return emitError(par->variable().get(), diag::redefinition_var);
    parameterTypes.push_back(parameterType);
    parameterCanMutateObject.push_back(canMutateObject);
  }
  return success();
}

auto SemaImpl::bindFunctionParameters(Prototype *node,
                                      const FunctionSymbol *signature)
    -> MulberryResult {
  auto &parameters = node->parameters();
  for (size_t i = 0; i < parameters.size(); ++i) {
    auto &parameter = parameters[i];
    auto *parameterType = signature->parameterTypes[i];
    auto canMutateObject = signature->parameterCanMutateObject[i];
    parameter->setType(parameterType);
    if (declareVariable(parameter->variable()->name(), parameterType,
                        !canMutateObject, canMutateObject))
      return emitError(parameter->variable().get(), diag::redefinition_var);
  }
  node->setType(signature->returnType);
  return success();
}

auto SemaImpl::semaFunctionSignature(Prototype *node) -> MulberryResult {
  std::vector<const Type *> parameterTypes;
  std::vector<bool> parameterCanMutateObject;
  if (semaFunctionParameters(node, parameterTypes, parameterCanMutateObject))
    return failure();

  auto *returnType = resolveType(node->returnTypeNode());
  if (!returnType)
    return failure();
  node->setType(returnType);

  auto name = node->id()->name();
  if (declareFunction(name, std::move(parameterTypes),
                      std::move(parameterCanMutateObject), returnType)) {
    auto diagnostic = formatNameDiagnostic(diag::redefinition_func, name);
    return emitError(node->id().get(), diagnostic);
  }
  return success();
}

auto SemaImpl::sema(FunctionDecl *node) -> MulberryResult {
  auto functionPackage = node->isExtern()
                             ? _currentPackageName
                             : functionPackageName(node->proto()->id()->name());
  PackageScope packageScope(_currentPackageName, functionPackage);
  if (node->isExtern()) {
    if (node->proto()->isGeneric())
      return emitError(node->proto()->id().get(), diag::mismatch_type);
    _symbols.resetVariables();
    auto result = sema(node->proto().get());
    _symbols.resetVariables();
    return result;
  }

  if (node->proto()->isGeneric()) {
    auto name = node->proto()->id()->name();
    if (lookupFunction(name) || lookupGenericFunction(name)) {
      auto diagnostic = formatNameDiagnostic(diag::redefinition_func, name);
      return emitError(node->proto()->id().get(), diagnostic);
    }
    if (declareGenericFunction(name, node)) {
      auto diagnostic = formatNameDiagnostic(diag::redefinition_func, name);
      return emitError(node->proto()->id().get(), diagnostic);
    }
    return success();
  }

  _symbols.resetVariables();
  auto *signature = lookupFunction(node->proto()->id()->name());
  if (signature) {
    if (bindFunctionParameters(node->proto().get(), signature))
      return failure();
  } else {
    if (sema(node->proto().get()))
      return failure();
    signature = lookupFunction(node->proto()->id()->name());
    if (!signature)
      return failure();
  }
  FunctionReturnTypeScope returnTypeScope(_currentFunctionReturnType,
                                          signature->returnType);
  if (sema(node->body().get()))
    return failure();

  auto hasReturn = containsReturnStat(node->body().get());
  if (!isUnitType(signature->returnType) && !hasReturn)
    return emitError(node->proto()->id().get(), diag::wrong_return_type);

  return success();
}

auto SemaImpl::declareStructMethods(
    std::string_view ownerName, const VectorUniquePtr<FunctionDecl> &methods,
    const std::vector<ComptimeParam> &typeParameters,
    std::string_view packageName)
    -> MulberryResult {
  NameSet methodNames;
  for (auto &method : methods) {
    auto *prototype = method->proto().get();
    auto methodName = prototype->id()->name();
    if (!declareName(methodNames, methodName)) {
      auto diagnostic = formatNameDiagnostic(diag::redefinition_func,
                                             methodName);
      return emitError(prototype->id().get(), diagnostic);
    }

    auto fullName = methodFunctionName(ownerName, methodName);
    prototype->id()->setName(fullName);
    prototype->setIsMethod(true);
    if (!prototype->isGeneric() && !typeParameters.empty())
      prototype->setComptimeParameters(
          std::vector<ComptimeParam>(typeParameters.begin(),
                                     typeParameters.end()));

    if (prototype->isGeneric()) {
      if (declareGenericFunction(fullName, method.get(), packageName)) {
        auto diagnostic = formatNameDiagnostic(diag::redefinition_func,
                                               fullName);
        return emitError(prototype->id().get(), diagnostic);
      }
      continue;
    }

    _functionPackages[fullName] = std::string(packageName);
    if (sema(method.get()))
      return failure();
  }
  return success();
}

auto SemaImpl::sema(StructDecl *node) -> MulberryResult {
  PackageScope packageScope(_currentPackageName,
                            packageNameOf(node->id()->name()));
  std::vector<StructField> fields;
  NameSet fieldNames;
  unsigned fieldIndex = 0;
  for (auto &varDecl : *node) {
    auto var = varDecl->variable().get();
    auto *fieldType = checkType(varDecl->typeNode(), UnitPolicy::Reject);
    if (!fieldType)
      return failure();
    varDecl->setType(fieldType);
    auto fieldName = var->name();
    if (!declareName(fieldNames, fieldName))
      return emitError(var, diag::redefinition_var);
    fields.push_back(StructField{fieldName, fieldType, fieldIndex++});
  }
  auto id = node->id().get();
  if (lookupType(id->name()))
    return emitError(id, diag::redefinition_type);

  auto *structType =
      _typeContext.createStructType(id->name(), std::move(fields));
  id->setType(structType);
  if (declareStructType(structType))
    return emitError(id, diag::redefinition_type);
  if (declareStructMethods(id->name(), node->methods(), {},
                           packageNameOf(id->name())))
    return failure();
  return success();
}

auto SemaImpl::sema(ComptimeTypeAliasDecl *node) -> MulberryResult {
  auto packageName = packageNameOf(node->name());
  PackageScope packageScope(_currentPackageName, packageName);
  if (_symbols.lookupType(node->name()) ||
      _symbols.lookupComptimeTypeAlias(node->name()))
    return emitError(node, diag::redefinition_type);

  if (!node->isGeneric()) {
    auto *bodyType = checkType(node->bodyTypeNode(), UnitPolicy::Reject);
    if (!bodyType)
      return failure();
    if (_symbols.declareType(node->name(), bodyType))
      return emitError(node, diag::redefinition_type);
    if (auto *structTypeNode = dyn_cast<StructTypeNode>(node->bodyTypeNode()))
      return declareStructMethods(node->name(), structTypeNode->methods(), {},
                                  packageName);
    return success();
  }

  if (_symbols.declareComptimeTypeAlias(
          node->name(), packageName,
          std::vector<ComptimeParam>(node->parameters().begin(),
                                     node->parameters().end()),
          node->bodyTypeNode()))
    return emitError(node, diag::redefinition_type);
  if (auto *structTypeNode = dyn_cast<StructTypeNode>(node->bodyTypeNode())) {
    if (declareStructMethods(node->name(), structTypeNode->methods(),
                             node->parameters(), packageName))
      return failure();
  }
  return success();
}

auto SemaImpl::sema(Expr *node) -> MulberryResult {
  switch (node->getKind()) {
  case Expr::Expr_Unit:
    return sema(cast<UnitExpr>(node));
  case Expr::Expr_DecimalLiteral:
    return sema(cast<DecimalLiteralExpr>(node));
  case Expr::Expr_FloatLiteral:
    return sema(cast<FloatLiteralExpr>(node));
  case Expr::Expr_BoolLiteral:
    return sema(cast<BoolLiteralExpr>(node));
  case Expr::Expr_StringLiteral:
    return sema(cast<StringLiteralExpr>(node));
  case Expr::Expr_InterpolatedString:
    return sema(cast<InterpolatedStringExpr>(node));
  case Expr::Expr_ObjectIdentity:
    return sema(cast<ObjectIdentityExpr>(node));
  case Expr::Expr_CharLiteral:
    return sema(cast<CharLiteralExpr>(node));
  case Expr::Expr_TypeInfo:
    return emitError(node, diag::expected_comptime_value);
  case Expr::Expr_TypeLayout:
    return sema(cast<TypeLayoutExpr>(node));
  case Expr::Expr_HeapAlloc:
    return sema(cast<HeapAllocExpr>(node));
  case Expr::Expr_Call:
    return sema(cast<CallExpr>(node));
  case Expr::Expr_StructLiteral:
    return sema(cast<StructLiteralExpr>(node));
  case Expr::Expr_Variable:
    return sema(cast<VariableExpr>(node));
  case Expr::Expr_Member:
    return sema(cast<MemberExpr>(node));
  case Expr::Expr_Assign:
    return sema(cast<AssignExpr>(node));
  case Expr::Expr_ArrayLiteral:
    return sema(cast<ArrayLiteralExpr>(node));
  case Expr::Expr_Index:
    return sema(cast<IndexExpr>(node));
  case Expr::Expr_Binary:
    return sema(cast<BinaryExpr>(node));
  default:
    llvm_unreachable("Unexpected expression");
  }
}

auto SemaImpl::sema(Expr *node, const Type *type) -> MulberryResult {
  auto *arrayLiteral = dyn_cast<ArrayLiteralExpr>(node);
  if (arrayLiteral) {
    // Source `[...]` defaults to Array. Explicit Array annotations keep
    // target-typed literal semantics.
    if (auto *arrayType = mulberry::getArrayType(type))
      return sema(arrayLiteral, arrayType);
  }

  auto *call = dyn_cast<CallExpr>(node);
  if (call && !call->hasReceiver()) {
    auto name = canonicalizeImportedName(call->name());
    call->setName(name);
    if (auto *handler = lookupBuiltinHandler(name)) {
      LLVM_DEBUG(llvm::dbgs() << "dispatch builtin Sema handler `" << name
                              << "`\n");
      return (*handler)(call, type);
    }
  }

  if (call && call->hasReceiver())
    return semaMethodCall(call, type);

  if (call) {
    auto name = canonicalizeImportedName(call->name());
    if (auto *genericFunction = lookupGenericFunction(name)) {
      call->setName(name);
      return semaGenericCall(call, genericFunction, type);
    }
  }

  return sema(node);
}

auto SemaImpl::semaGenericCall(CallExpr *node,
                               const GenericFunctionSymbol *symbol,
                               const Type *expectedType)
    -> MulberryResult {
  auto *genericFunction = symbol->decl;
  auto *genericProto = genericFunction->proto().get();
  auto name = genericProto->id()->name();
  PackageScope packageScope(_currentPackageName,
                            genericFunctionPackageName(name));
  auto &expressions = node->expressions();
  auto &parameters = genericProto->parameters();
  if (expressions.size() != parameters.size()) {
    auto diagnostic =
        formatNameSizeDiagnostic(diag::func_param, name, parameters.size());
    return emitError(node, diagnostic);
  }

  auto &comptimeParameters = genericProto->comptimeParameters();
  auto inferredArguments =
      makeInferredComptimeArguments(comptimeParameters);
  std::vector<const Type *> arrayLeafConstraints(comptimeParameters.size());
  auto returnHasComputedType =
      hasComputedType(genericProto->returnTypeNode());
  if (expectedType) {
    if (!matchGenericType(genericProto->returnTypeNode(), expectedType,
                          comptimeParameters, inferredArguments,
                          &arrayLeafConstraints)) {
      inferredArguments =
          makeInferredComptimeArguments(comptimeParameters);
      arrayLeafConstraints.assign(comptimeParameters.size(), nullptr);
      if (!returnHasComputedType)
        return emitError(node, diag::mismatch_type);
      LLVM_DEBUG(llvm::dbgs()
                 << "defer computed return type of `" << name << "`\n");
    }
  }

  auto semaArgument = [&](Expr *argument, const TypeNode *parameterTypeNode)
      -> MulberryResult {
    auto *literal = dyn_cast<ArrayLiteralExpr>(argument);
    auto *namedType = dyn_cast<NamedTypeNode>(parameterTypeNode);
    auto parameterIndex = namedType
                              ? comptimeParameterIndex(
                                    comptimeParameters, namedType->name())
                              : std::nullopt;
    if (literal && parameterIndex &&
        !inferredArguments[*parameterIndex].isResolved() &&
        arrayLeafConstraints[*parameterIndex]) {
      auto *arrayType = arrayLiteralTypeWithLeaf(
          literal, arrayLeafConstraints[*parameterIndex]);
      if (!arrayType)
        return emitError(literal, diag::expected_expr);
      LLVM_DEBUG(llvm::dbgs()
                 << "target Array literal leaf from computed return of `"
                 << name << "`: " << formatType(arrayType) << "\n");
      return sema(literal, arrayType);
    }

    auto knownArguments = true;
    for (auto &inferredArgument : inferredArguments)
      knownArguments = knownArguments && inferredArgument.isResolved();
    if (!knownArguments)
      return sema(argument);

    auto *parameterType = resolveSubstitutedType(
        parameterTypeNode,
        comptimeSubstitutions(comptimeParameters, inferredArguments));
    if (!parameterType)
      return failure();
    return sema(argument, parameterType);
  };

  std::vector<size_t> deferredParameters;
  for (size_t i = 0; i < expressions.size(); ++i) {
    auto *parameterTypeNode = parameters[i]->typeNode();
    if (hasComputedType(parameterTypeNode)) {
      deferredParameters.push_back(i);
      LLVM_DEBUG(llvm::dbgs()
                 << "defer computed parameter " << i << " of `" << name
                 << "`\n");
      continue;
    }

    if (semaArgument(expressions[i].get(), parameterTypeNode))
      return failure();
    auto matched =
        node->isLoweredMethodCall() && i == 0
            ? matchMethodReceiverType(parameterTypeNode, expressions[i]->type(),
                                      comptimeParameters, inferredArguments)
            : matchGenericType(parameterTypeNode, expressions[i]->type(),
                               comptimeParameters, inferredArguments);
    if (!matched)
      return emitError(expressions[i].get(), diag::mismatch_type);
  }

  for (auto &argument : inferredArguments)
    if (!argument.isResolved())
      return emitError(node, diag::mismatch_type);

  auto substitutions =
      comptimeSubstitutions(comptimeParameters, inferredArguments);
  for (auto index : deferredParameters) {
    auto *parameterType = resolveSubstitutedType(
        parameters[index]->typeNode(), substitutions);
    if (!parameterType)
      return failure();

    auto &argument = expressions[index];
    if (node->isLoweredMethodCall() && index == 0) {
      if (sema(argument.get()))
        return failure();
    } else if (sema(argument.get(), parameterType)) {
      return failure();
    }
  }

  auto concreteName = genericFunctionName(name, inferredArguments);
  auto cached = _instantiatedFunctionSymbols.find(concreteName);
  if (cached == _instantiatedFunctionSymbols.end()) {
    auto concreteFunction = instantiateFunctionDecl(
        genericFunction, concreteName, substitutions);
    _instantiatedFunctionPackages[concreteName] =
        genericFunctionPackageName(name);

    VariableScope signatureScope(_symbols);
    if (semaFunctionSignature(concreteFunction->proto().get()))
      return failure();
    auto *signature = lookupFunction(concreteName);
    if (!signature)
      return failure();

    cached = _instantiatedFunctionSymbols
                 .insert({concreteName, signature})
                 .first;
    _instantiatedFunctions.push_back(std::move(concreteFunction));
  }

  auto *signature = cached->second;
  if (expectedType && !sameType(expectedType, signature->returnType))
    return emitError(node, diag::mismatch_type);

  for (size_t i = 0; i < expressions.size(); ++i) {
    auto &arg = expressions[i];
    auto *parameterType = signature->parameterTypes[i];
    if (!sameCallArgumentType(parameterType, arg->type(),
                              node->isLoweredMethodCall() && i == 0))
      return emitError(arg.get(), diag::mismatch_type);
    if (checkMutableObjectArgument(signature, i, arg.get()))
      return failure();
  }

  node->setName(concreteName);
  if (signature->returnType)
    node->setType(signature->returnType);
  return success();
}

auto SemaImpl::sema(UnitExpr *node) -> MulberryResult {
  setBuiltinType(node, BuiltinTypeKind::Unit);
  return success();
}

auto SemaImpl::sema(BlockExpr *node) -> MulberryResult {
  VariableScope blockScope(_symbols);
  for (auto &expr : *node)
    if (sema(expr.get()))
      return failure();

  node->setType(_typeContext.getBuiltinType(BuiltinTypeKind::Unit));
  return success();
}

auto SemaImpl::sema(CallExpr *node) -> MulberryResult {
  if (node->hasReceiver())
    return semaMethodCall(node);

  node->setName(canonicalizeImportedName(node->name()));
  if (node->name().rfind("std.internal.", 0) == 0 &&
      checkInternalFeature(node->location()))
    return failure();

  if (auto *handler = lookupBuiltinHandler(node->name())) {
    LLVM_DEBUG(llvm::dbgs() << "dispatch builtin Sema handler `"
                            << node->name() << "`\n");
    return (*handler)(node, nullptr);
  }

  auto name = node->name();

  auto *signature = lookupFunction(name);
  if (!signature) {
    if (auto *genericFunction = lookupGenericFunction(name))
      return semaGenericCall(node, genericFunction);

    if (name.find('.') != std::string_view::npos)
      return semaDottedMethodCall(node);

    auto diagnostic = formatNameDiagnostic(diag::undefined_func, name);
    return emitError(node, diagnostic);
  }

  auto &expressions = node->expressions();
  if (expressions.size() != signature->parameterTypes.size()) {
    auto diagnostic = formatNameSizeDiagnostic(
        diag::func_param, name, signature->parameterTypes.size());
    return emitError(node, diagnostic);
  }

  for (size_t i = 0; i < expressions.size(); ++i) {
    auto &arg = expressions[i];
    auto *parameterType = signature->parameterTypes[i];
    if (node->isLoweredMethodCall() && i == 0) {
      if (sema(arg.get()))
        return failure();
    } else if (sema(arg.get(), parameterType)) {
      return failure();
    }
    if (!sameCallArgumentType(parameterType, arg->type(),
                              node->isLoweredMethodCall() && i == 0))
      return emitError(arg.get(), diag::mismatch_type);
    if (checkMutableObjectArgument(signature, i, arg.get()))
      return failure();
  }

  auto resolvedName = resolveFunctionName(name);
  if (!resolvedName.empty())
    node->setName(resolvedName);

  if (signature->returnType)
    node->setType(signature->returnType);
  return success();
}

auto SemaImpl::semaMethodCall(CallExpr *node, const Type *expectedType)
    -> MulberryResult {
  if (!node->hasReceiver())
    return semaDottedMethodCall(node, expectedType);

  if (sema(node->receiver().get()))
    return failure();

  auto *receiverType = node->receiver()->type();
  auto *ptrType = mulberry::getPtrType(receiverType);
  auto *structType = ptrType ? mulberry::getStructType(ptrType->pointeeType())
                             : mulberry::getStructType(receiverType);
  if (!structType)
    return emitError(node->receiver().get(), diag::mismatch_type);

  std::vector<std::string> owners;
  if (auto *origin = structType->origin())
    owners.push_back(std::string(origin->aliasName()));
  owners.push_back(std::string(structType->name()));

  auto methodName = std::string(node->name());
  for (auto &owner : owners) {
    auto fullName = methodFunctionName(owner, methodName);
    if (auto *genericFunction = lookupGenericFunction(fullName)) {
      node->lowerMethodCall(fullName);
      return semaGenericCall(node, genericFunction, expectedType);
    }

    if (lookupFunction(fullName)) {
      node->lowerMethodCall(fullName);
      return sema(node);
    }
  }

  auto diagnostic = formatNameDiagnostic(diag::undefined_func, methodName);
  return emitError(node, diagnostic);
}

auto SemaImpl::semaDottedMethodCall(CallExpr *node, const Type *expectedType)
    -> MulberryResult {
  auto name = std::string(node->name());
  auto dot = name.rfind('.');
  if (dot == std::string::npos) {
    auto diagnostic = formatNameDiagnostic(diag::undefined_func, name);
    return emitError(node, diagnostic);
  }

  auto receiverName = name.substr(0, dot);
  auto methodName = name.substr(dot + 1);
  node->setReceiver(createMemberAccessChain(node->location(), receiverName),
                    methodName);
  return semaMethodCall(node, expectedType);
}

auto SemaImpl::sema(StructLiteralExpr *node) -> MulberryResult {
  auto *type = resolveType(node->typeNode());
  auto *structType = mulberry::getStructType(type);
  if (!structType)
    return emitError(node, diag::undefined_type);
  node->setStructType(structType);

  auto &expressions = node->expressions();
  auto &fields = structType->fields();
  if (expressions.size() != fields.size())
    return emitError(node, diag::wrong_num_arg);

  for (size_t i = 0; i < expressions.size(); ++i) {
    auto &expr = expressions[i];
    auto &field = fields[i];
    if (sema(expr.get(), field.type()))
      return failure();
    if (!sameType(field.type(), expr->type()))
      return emitError(expr.get(), diag::mismatch_type);
  }

  node->setType(structType);
  return success();
}

auto SemaImpl::sema(VariableExpr *node) -> MulberryResult {
  auto *symbol = lookupVariable(node->name());
  if (!symbol)
    return emitError(node, diag::undefined_var);

  if (symbol->isComptimeOnly) {
    assert(symbol->comptimeValue && "comptime variable has no value");
    auto *type = comptimeRuntimeType(*symbol->comptimeValue);
    if (!type)
      return emitError(node, diag::comptime_type_runtime);
    node->setType(type);
    node->setComptimeValue(*symbol->comptimeValue);
    return success();
  }

  node->setType(symbol->type);
  return success();
}

auto SemaImpl::sema(MemberExpr *node) -> MulberryResult {
  if (sema(node->base().get()))
    return failure();

  auto *baseType = node->base()->type();
  auto *ptrType = mulberry::getPtrType(baseType);
  auto *structType = ptrType ? mulberry::getStructType(ptrType->pointeeType())
                             : mulberry::getStructType(baseType);
  if (!structType)
    return emitError(node->base().get(), diag::mismatch_type);

  auto *field = structType->field(node->fieldName());
  if (!field)
    return emitError(node, diag::undefined_field);
  if (!field->type())
    return emitError(node, diag::mismatch_type);
  if (mulberry::isPtrType(field->type()) &&
      checkInternalFeature(node->location()))
    return failure();

  node->setType(field->type());
  node->setFieldIndex(field->index());
  node->setLvalue(ptrType || node->base()->isLvalue());
  return success();
}

auto SemaImpl::sema(DecimalLiteralExpr *node) -> MulberryResult {
  setBuiltinType(node, BuiltinTypeKind::UInt64);
  return success();
}

auto SemaImpl::sema(FloatLiteralExpr *node) -> MulberryResult {
  setBuiltinType(node, BuiltinTypeKind::Float32);
  return success();
}

auto SemaImpl::sema(BoolLiteralExpr *node) -> MulberryResult {
  setBuiltinType(node, BuiltinTypeKind::Bool);
  return success();
}

auto SemaImpl::sema(StringLiteralExpr *node) -> MulberryResult {
  auto *type = lookupType("String");
  if (!type)
    return emitError(node, diag::undefined_type);
  node->setType(type);
  return success();
}

auto SemaImpl::sema(InterpolatedStringExpr *node) -> MulberryResult {
  auto *stringType = lookupType("String");
  if (!stringType)
    return emitError(node, diag::undefined_type);

  for (auto &segment : node->segments()) {
    if (semaFormatValueCall(segment, stringType))
      return failure();
  }

  if (node->segments().size() > 1 &&
      checkStringConcatFunction(node, stringType))
    return failure();
  node->setType(stringType);
  return success();
}

auto SemaImpl::sema(ObjectIdentityExpr *node) -> MulberryResult {
  if (checkInternalFeature(node->location()))
    return failure();

  if (sema(node->value().get()))
    return failure();

  auto *valueType = node->value()->type();
  if (!getStructType(valueType) && !getArrayType(valueType))
    return emitError(node->value().get(), diag::mismatch_type);

  auto *stringType = lookupType("String");
  if (!stringType)
    return emitError(node, diag::undefined_type);

  constexpr std::string_view functionName =
      "mulberry_string_object_identity";
  auto resolvedName = resolveFunctionName(functionName);
  auto *signature = lookupFunction(resolvedName);
  if (!signature || signature->parameterTypes.size() != 2 ||
      !sameType(signature->parameterTypes[0], stringType) ||
      !sameType(signature->returnType, stringType)) {
    auto diagnostic = formatNameDiagnostic(diag::undefined_func, functionName);
    return emitError(node, diagnostic);
  }

  auto *objectPtr = getPtrType(signature->parameterTypes[1]);
  if (!objectPtr || !isUInt8Type(objectPtr->pointeeType())) {
    auto diagnostic = formatNameDiagnostic(diag::undefined_func, functionName);
    return emitError(node, diagnostic);
  }

  node->setTypeName(formatStringificationType(valueType));
  node->setType(stringType);
  return success();
}

auto SemaImpl::sema(CharLiteralExpr *node) -> MulberryResult {
  setBuiltinType(node, BuiltinTypeKind::UInt8);
  return success();
}

auto SemaImpl::sema(TypeLayoutExpr *node) -> MulberryResult {
  auto *queriedType = resolveType(node->typeNode());
  if (!queriedType)
    return failure();

  auto value = node->query() == TypeLayoutExpr::Query::SizeOf
                   ? sizeOfType(queriedType)
                   : alignOfType(queriedType);
  if (!value)
    return emitError(node, diag::unsupported_type_layout);

  node->setQueriedType(queriedType);
  node->setValue(*value);
  setBuiltinType(node, BuiltinTypeKind::UInt64);
  return success();
}

auto SemaImpl::sema(HeapAllocExpr *node) -> MulberryResult {
  if (checkInternalFeature(node->location()))
    return failure();

  auto *allocatedType = checkType(node->typeNode(), UnitPolicy::Reject);
  if (!allocatedType)
    return failure();

  if (node->count()) {
    if (sema(node->count().get()))
      return failure();
    if (!isUInt64Type(node->count()->type()))
      return emitError(node->count().get(), diag::mismatch_type);
  }

  node->setAllocatedType(allocatedType);
  node->setType(_typeContext.createPtrType(allocatedType));
  return success();
}

auto SemaImpl::sema(AssignExpr *node) -> MulberryResult {
  if (sema(node->lhs().get()) ||
      sema(node->rhs().get(), node->lhs()->type()))
    return failure();
  if (!sameType(node->lhs()->type(), node->rhs()->type()))
    return emitError(node->lhs().get(), diag::mismatch_type);
  if (!node->lhs()->isLvalue())
    return emitError(node->lhs().get(), diag::expected_lvalue);
  if (checkAssignable(node->lhs().get()))
    return failure();
  setBuiltinType(node, BuiltinTypeKind::Unit);
  return success();
}

auto SemaImpl::sema(BinaryExpr *node) -> MulberryResult {
  using Operator = BinaryExpr::Operator;
  if (sema(node->lhs().get()) || sema(node->rhs().get()))
    return failure();

  auto *lhsType = node->lhs()->type();
  auto *rhsType = node->rhs()->type();
  auto *stringType = lookupType("String");
  if (node->opEnum() == Operator::Add && stringType &&
      sameType(lhsType, stringType)) {
    if (semaFormatValueCall(node->rhs(), stringType))
      return failure();
    if (checkStringConcatFunction(node, stringType))
      return failure();
    node->setType(stringType);
    return success();
  }

  if (!sameType(lhsType, rhsType))
    return emitError(node->lhs().get(), diag::mismatch_type);

  switch (node->opEnum()) {
  case Operator::Add:
  case Operator::Mul:
  case Operator::Diff:
  case Operator::Div: {
    if (!isNumericType(lhsType))
      return emitError(node->lhs().get(), diag::mismatch_type);
    if (lhsType)
      node->setType(lhsType);
    return success();
  }
  case Operator::Rem: {
    if (!isUInt64Type(lhsType))
      return emitError(node->lhs().get(), diag::mismatch_type);
    setBuiltinType(node, BuiltinTypeKind::UInt64);
    return success();
  }
  case Operator::And:
  case Operator::Or: {
    if (!isBoolType(lhsType))
      return emitError(node->lhs().get(), diag::mismatch_type);
    setBuiltinType(node, BuiltinTypeKind::Bool);
    return success();
  }
  case Operator::EQ:
  case Operator::NEQ: {
    if (!isEquatableType(lhsType))
      return emitError(node->lhs().get(), diag::mismatch_type);
    setBuiltinType(node, BuiltinTypeKind::Bool);
    return success();
  }
  case Operator::LT:
  case Operator::LE:
  case Operator::GT:
  case Operator::GE: {
    if (!isNumericType(lhsType))
      return emitError(node->lhs().get(), diag::mismatch_type);
    setBuiltinType(node, BuiltinTypeKind::Bool);
    return success();
  }
  }

  llvm_unreachable("Unexpected BinaryExpr operator");
}

auto SemaImpl::checkStringConcatFunction(Expr *node,
                                         const Type *stringType)
    -> MulberryResult {
  auto functionName = resolveFunctionName("std.string.concat");
  auto *signature = lookupFunction(functionName);
  if (signature && signature->parameterTypes.size() == 2 &&
      sameType(signature->parameterTypes[0], stringType) &&
      sameType(signature->parameterTypes[1], stringType) &&
      sameType(signature->returnType, stringType))
    return success();

  auto diagnostic =
      formatNameDiagnostic(diag::undefined_func, "std.string.concat");
  return emitError(node, diagnostic);
}

auto SemaImpl::hasMethod(const Type *type, std::string_view methodName)
    -> bool {
  auto *structType = getStructType(type);
  if (!structType)
    return false;

  std::vector<std::string> owners;
  if (auto *origin = structType->origin())
    owners.push_back(std::string(origin->aliasName()));
  owners.push_back(std::string(structType->name()));

  for (auto &owner : owners) {
    auto fullName = methodFunctionName(owner, methodName);
    if (lookupFunction(fullName) || lookupGenericFunction(fullName))
      return true;
  }
  return false;
}

auto SemaImpl::comptimeRuntimeType(const ComptimeValue &value) -> const Type * {
  switch (value.kind()) {
  case ComptimeValue::Kind::Type:
    return nullptr;
  case ComptimeValue::Kind::Bool:
    return _typeContext.getBuiltinType(BuiltinTypeKind::Bool);
  case ComptimeValue::Kind::UInt64:
    return _typeContext.getBuiltinType(BuiltinTypeKind::UInt64);
  case ComptimeValue::Kind::String:
    return lookupType("String");
  }
  llvm_unreachable("unexpected comptime value");
}

auto SemaImpl::setComptimeResultType(Expr *node,
                                      const ComptimeValue &value) -> void {
  if (auto *type = comptimeRuntimeType(value))
    node->setType(type);
}

auto SemaImpl::evaluateComptime(Expr *node) -> ComptimeEvaluation {
  if (auto *typeExpr = dyn_cast<TypeInfoExpr>(node)) {
    auto *type = resolveType(typeExpr->typeNode());
    if (!type)
      return {ComptimeEvaluation::Kind::Error, std::nullopt};
    return {ComptimeEvaluation::Kind::Value, ComptimeValue(type), true};
  }

  if (auto *value = dyn_cast<BoolLiteralExpr>(node)) {
    setBuiltinType(value, BuiltinTypeKind::Bool);
    return {ComptimeEvaluation::Kind::Value,
            ComptimeValue(value->value())};
  }

  if (auto *value = dyn_cast<DecimalLiteralExpr>(node)) {
    setBuiltinType(value, BuiltinTypeKind::UInt64);
    return {ComptimeEvaluation::Kind::Value,
            ComptimeValue(value->value())};
  }

  if (auto *value = dyn_cast<StringLiteralExpr>(node))
    return {ComptimeEvaluation::Kind::Value,
            ComptimeValue(value->value())};

  if (auto *variable = dyn_cast<VariableExpr>(node)) {
    if (auto *symbol = lookupVariable(variable->name())) {
      if (symbol->comptimeValue) {
        if (symbol->isComptimeOnly)
          variable->setComptimeValue(*symbol->comptimeValue);
        return {ComptimeEvaluation::Kind::Value, *symbol->comptimeValue,
                symbol->isComptimeOnly};
      }
      return {ComptimeEvaluation::Kind::Runtime, std::nullopt};
    }
    if (auto *type = lookupType(variable->name()))
      return {ComptimeEvaluation::Kind::Value, ComptimeValue(type), true};
    return {ComptimeEvaluation::Kind::Runtime, std::nullopt};
  }

  if (auto *call = dyn_cast<CallExpr>(node)) {
    auto result = evaluateComptimeCall(call);
    if (result.kind == ComptimeEvaluation::Kind::Value)
      setComptimeResultType(call, *result.value);
    return result;
  }

  if (auto *binary = dyn_cast<BinaryExpr>(node)) {
    auto result = evaluateComptimeBinary(binary);
    if (result.kind == ComptimeEvaluation::Kind::Value)
      setComptimeResultType(binary, *result.value);
    return result;
  }

  return {ComptimeEvaluation::Kind::Runtime, std::nullopt};
}

auto SemaImpl::evaluateComptimeCall(CallExpr *node) -> ComptimeEvaluation {
  auto &arguments = node->expressions();
  if (!node->hasReceiver()) {
    if (node->name() == builtins::typeOf) {
      if (arguments.size() != 1) {
        auto diagnostic = formatNameSizeDiagnostic(
            diag::func_param, node->name(), 1);
        emitError(node, diagnostic);
        return {ComptimeEvaluation::Kind::Error, std::nullopt};
      }
      auto *expression = arguments.front().get();
      if (sema(expression))
        return {ComptimeEvaluation::Kind::Error, std::nullopt};
      return {ComptimeEvaluation::Kind::Value,
              ComptimeValue(expression->type()), true};
    }

    // The parser keeps dotted calls package-like until Sema can prove that the
    // left side is a receiver. Reflection needs the same delayed decision.
    auto name = std::string(node->name());
    auto dot = name.rfind('.');
    if (dot != std::string::npos) {
      auto receiver = createMemberAccessChain(
          node->location(), std::string_view(name).substr(0, dot));
      auto receiverValue = evaluateComptime(receiver.get());
      if (receiverValue.kind == ComptimeEvaluation::Kind::Error)
        return receiverValue;
      if (receiverValue.kind == ComptimeEvaluation::Kind::Value &&
          receiverValue.value->kind() == ComptimeValue::Kind::Type) {
        LLVM_DEBUG(llvm::dbgs()
                   << "resolve comptime method call `" << name << "`\n");
        node->setReceiver(std::move(receiver),
                          std::string_view(name).substr(dot + 1));
        return evaluateComptimeCall(node);
      }
    }

    return {ComptimeEvaluation::Kind::Runtime, std::nullopt};
  }

  auto receiver = evaluateComptime(node->receiver().get());
  if (receiver.kind != ComptimeEvaluation::Kind::Value)
    return receiver;
  if (receiver.value->kind() != ComptimeValue::Kind::Type)
    return {ComptimeEvaluation::Kind::Runtime, std::nullopt};

  auto *type = receiver.value->type();
  auto name = node->name();
  auto requireNoArguments = [&]() -> bool {
    if (arguments.empty())
      return true;
    auto diagnostic = formatNameSizeDiagnostic(diag::func_param, name, 0);
    emitError(node, diagnostic);
    return false;
  };

  if (name == "isBool" || name == "isUInt8" || name == "isUInt64" ||
      name == "isFloat32" || name == "isArray" || name == "isStruct") {
    if (!requireNoArguments())
      return {ComptimeEvaluation::Kind::Error, std::nullopt};
    if (name == "isBool")
      return {ComptimeEvaluation::Kind::Value,
              ComptimeValue(isBoolType(type)), true};
    if (name == "isUInt8")
      return {ComptimeEvaluation::Kind::Value,
              ComptimeValue(isUInt8Type(type)), true};
    if (name == "isUInt64")
      return {ComptimeEvaluation::Kind::Value,
              ComptimeValue(isUInt64Type(type)), true};
    if (name == "isFloat32")
      return {ComptimeEvaluation::Kind::Value,
              ComptimeValue(isFloat32Type(type)), true};
    if (name == "isArray")
      return {ComptimeEvaluation::Kind::Value,
              ComptimeValue(isArrayType(type)), true};
    return {ComptimeEvaluation::Kind::Value,
            ComptimeValue(getStructType(type) != nullptr), true};
  }

  if (name == "hasMethod") {
    if (arguments.size() != 1) {
      auto diagnostic = formatNameSizeDiagnostic(diag::func_param, name, 1);
      emitError(node, diagnostic);
      return {ComptimeEvaluation::Kind::Error, std::nullopt};
    }
    auto methodName = evaluateComptime(arguments.front().get());
    if (methodName.kind == ComptimeEvaluation::Kind::Error)
      return methodName;
    if (methodName.kind != ComptimeEvaluation::Kind::Value ||
        methodName.value->kind() != ComptimeValue::Kind::String) {
      emitError(arguments.front().get(), diag::expected_comptime_value);
      return {ComptimeEvaluation::Kind::Error, std::nullopt};
    }
    return {ComptimeEvaluation::Kind::Value,
            ComptimeValue(hasMethod(type,
                                    methodName.value->stringValue())), true};
  }

  if (name == "arrayElementType" || name == "arrayLeafElementType" ||
      name == "arrayLength" || name == "arrayRank") {
    if (!requireNoArguments())
      return {ComptimeEvaluation::Kind::Error, std::nullopt};
    auto *arrayType = getArrayType(type);
    if (!arrayType) {
      emitError(node, diag::invalid_reflection_query);
      return {ComptimeEvaluation::Kind::Error, std::nullopt};
    }
    if (name == "arrayElementType")
      return {ComptimeEvaluation::Kind::Value,
              ComptimeValue(arrayType->elementType()), true};
    if (name == "arrayLeafElementType")
      return {ComptimeEvaluation::Kind::Value,
              ComptimeValue(getArrayLeafElementType(type)), true};
    if (name == "arrayLength")
      return {ComptimeEvaluation::Kind::Value,
              ComptimeValue(arrayType->size()), true};
    return {ComptimeEvaluation::Kind::Value,
            ComptimeValue(static_cast<uint64_t>(getArrayShape(type).size())),
            true};
  }

  auto diagnostic = formatNameDiagnostic(diag::undefined_func, name);
  emitError(node, diagnostic);
  return {ComptimeEvaluation::Kind::Error, std::nullopt};
}

auto SemaImpl::evaluateComptimeBinary(BinaryExpr *node)
    -> ComptimeEvaluation {
  using EvaluationKind = ComptimeEvaluation::Kind;
  using Operator = BinaryExpr::Operator;

  auto lhs = evaluateComptime(node->lhs().get());
  if (lhs.kind != EvaluationKind::Value)
    return lhs;

  auto op = node->opEnum();
  if (lhs.value->kind() == ComptimeValue::Kind::Bool) {
    if (op == Operator::And && !lhs.value->boolValue())
      return {EvaluationKind::Value, ComptimeValue(false),
              lhs.isComptimeOnly};
    if (op == Operator::Or && lhs.value->boolValue())
      return {EvaluationKind::Value, ComptimeValue(true),
              lhs.isComptimeOnly};
  }

  auto rhs = evaluateComptime(node->rhs().get());
  if (rhs.kind != EvaluationKind::Value)
    return rhs;
  auto isComptimeOnly = lhs.isComptimeOnly || rhs.isComptimeOnly;
  auto cannotEvaluate = [&]() -> ComptimeEvaluation {
    if (!isComptimeOnly)
      return {EvaluationKind::Runtime, std::nullopt};
    emitError(node, diag::expected_comptime_value);
    return {EvaluationKind::Error, std::nullopt};
  };

  if (op == Operator::EQ || op == Operator::NEQ) {
    if (lhs.value->kind() != rhs.value->kind())
      return cannotEvaluate();

    bool equal = false;
    switch (lhs.value->kind()) {
    case ComptimeValue::Kind::Type:
      equal = sameType(lhs.value->type(), rhs.value->type());
      break;
    case ComptimeValue::Kind::Bool:
      equal = lhs.value->boolValue() == rhs.value->boolValue();
      break;
    case ComptimeValue::Kind::UInt64:
      equal = lhs.value->uint64Value() == rhs.value->uint64Value();
      break;
    case ComptimeValue::Kind::String:
      equal = lhs.value->stringValue() == rhs.value->stringValue();
      break;
    }
    return {EvaluationKind::Value,
            ComptimeValue(op == Operator::EQ ? equal : !equal),
            isComptimeOnly};
  }

  if (lhs.value->kind() == ComptimeValue::Kind::Bool &&
      rhs.value->kind() == ComptimeValue::Kind::Bool) {
    if (op == Operator::And)
      return {EvaluationKind::Value,
              ComptimeValue(lhs.value->boolValue() &&
                            rhs.value->boolValue()),
              isComptimeOnly};
    if (op == Operator::Or)
      return {EvaluationKind::Value,
              ComptimeValue(lhs.value->boolValue() ||
                            rhs.value->boolValue()),
              isComptimeOnly};
    return cannotEvaluate();
  }

  if (lhs.value->kind() != ComptimeValue::Kind::UInt64 ||
      rhs.value->kind() != ComptimeValue::Kind::UInt64)
    return cannotEvaluate();

  auto left = lhs.value->uint64Value();
  auto right = rhs.value->uint64Value();
  switch (op) {
  case Operator::Add:
    return {EvaluationKind::Value, ComptimeValue(left + right),
            isComptimeOnly};
  case Operator::Mul:
    return {EvaluationKind::Value, ComptimeValue(left * right),
            isComptimeOnly};
  case Operator::Diff:
    return {EvaluationKind::Value, ComptimeValue(left - right),
            isComptimeOnly};
  case Operator::Div:
    if (right != 0)
      return {EvaluationKind::Value, ComptimeValue(left / right),
              isComptimeOnly};
    break;
  case Operator::Rem:
    if (right != 0)
      return {EvaluationKind::Value, ComptimeValue(left % right),
              isComptimeOnly};
    break;
  case Operator::LT:
    return {EvaluationKind::Value, ComptimeValue(left < right),
            isComptimeOnly};
  case Operator::LE:
    return {EvaluationKind::Value, ComptimeValue(left <= right),
            isComptimeOnly};
  case Operator::GT:
    return {EvaluationKind::Value, ComptimeValue(left > right),
            isComptimeOnly};
  case Operator::GE:
    return {EvaluationKind::Value, ComptimeValue(left >= right),
            isComptimeOnly};
  case Operator::And:
  case Operator::Or:
  case Operator::EQ:
  case Operator::NEQ:
    break;
  }

  return cannotEvaluate();
}

auto SemaImpl::semaFormatValueCall(std::unique_ptr<Expr> &expression,
                                   const Type *stringType)
    -> MulberryResult {
  if (sema(expression.get()))
    return failure();
  if (isUnitType(expression->type()))
    return emitError(expression.get(), diag::mismatch_type);

  auto location = expression->location();
  VectorUniquePtr<Expr> arguments;
  arguments.push_back(std::move(expression));
  auto call = std::make_unique<CallExpr>(
      location, "std.string.formatValue", std::move(arguments));
  auto *callExpr = call.get();
  expression = std::move(call);
  if (sema(callExpr, stringType))
    return failure();
  if (!sameType(callExpr->type(), stringType))
    return emitError(callExpr, diag::mismatch_type);
  return success();
}

auto SemaImpl::checkAssignable(const Expr *expr) -> MulberryResult {
  if (auto *var = llvm::dyn_cast<VariableExpr>(expr)) {
    auto *symbol = lookupVariable(var->name());
    if (!symbol)
      return emitError(var->location(), diag::undefined_var);
    if (symbol->isConstBinding)
      return emitError(var->location(), diag::assign_const);
    return success();
  }

  if (auto *index = llvm::dyn_cast<IndexExpr>(expr))
    return checkConstObjectUseAsMutable(index->base().get());

  if (auto *memberAccess = llvm::dyn_cast<MemberExpr>(expr)) {
    if (!memberAccess->isLvalue())
      return emitError(memberAccess, diag::expected_lvalue);
    return checkConstObjectUseAsMutable(memberAccess->base().get());
  }

  return success();
}

auto SemaImpl::canMutateObjectReference(const Expr *expr) -> bool {
  if (auto *index = llvm::dyn_cast<IndexExpr>(expr))
    return canMutateObjectReference(index->base().get());

  if (auto *memberAccess = llvm::dyn_cast<MemberExpr>(expr))
    return canMutateObjectReference(memberAccess->base().get());

  auto *var = llvm::dyn_cast<VariableExpr>(expr);
  if (!var)
    return true;

  auto *symbol = lookupVariable(var->name());
  return !symbol || symbol->canMutateObject;
}

auto SemaImpl::checkConstObjectUseAsMutable(const Expr *expr)
    -> MulberryResult {
  if (auto *index = llvm::dyn_cast<IndexExpr>(expr))
    return checkConstObjectUseAsMutable(index->base().get());

  if (auto *memberAccess = llvm::dyn_cast<MemberExpr>(expr))
    return checkConstObjectUseAsMutable(memberAccess->base().get());

  auto *var = llvm::dyn_cast<VariableExpr>(expr);
  if (!var)
    return success();

  auto *symbol = lookupVariable(var->name());
  if (!symbol)
    return emitError(var->location(), diag::undefined_var);
  if (!symbol->canMutateObject)
    return emitError(var->location(), diag::readonly_to_mutable_reference);
  return success();
}

auto SemaImpl::checkMutableObjectArgument(const FunctionSymbol *signature,
                                          size_t index, const Expr *arg)
    -> MulberryResult {
  if (!signature->parameterCanMutateObject[index])
    return success();
  if (!isSourceObjectType(signature->parameterTypes[index]))
    return success();
  return checkConstObjectUseAsMutable(arg);
}

auto SemaImpl::arrayLiteralTypeWithLeaf(const ArrayLiteralExpr *expr,
                                        const Type *leafType)
    -> const ArrayType * {
  auto &elements = expr->getElements();
  if (elements.empty())
    return nullptr;

  const Type *elementType = leafType;
  if (auto *nested = dyn_cast<ArrayLiteralExpr>(elements.front().get())) {
    elementType = arrayLiteralTypeWithLeaf(nested, leafType);
    if (!elementType)
      return nullptr;
  }
  return _typeContext.createArrayType(elementType, elements.size());
}

auto SemaImpl::semaDefaultArrayLiteral(ArrayLiteralExpr *expr)
    -> MulberryResult {
  auto &elements = expr->getElements();
  if (elements.empty())
    return emitError(expr, diag::expected_expr);

  auto semaElement = [&](Expr *element) -> MulberryResult {
    if (auto *nestedLiteral = dyn_cast<ArrayLiteralExpr>(element))
      return semaDefaultArrayLiteral(nestedLiteral);
    return sema(element);
  };

  if (semaElement(elements.front().get()))
    return failure();

  auto *elementType = elements.front()->type();
  if (!isArrayElementType(elementType))
    return emitError(elements.front().get(), diag::mismatch_type);

  for (size_t index = 1; index < elements.size(); ++index) {
    auto &element = elements[index];
    if (semaElement(element.get()))
      return failure();
    if (!sameType(elementType, element->type()))
      return emitError(element.get(), diag::mismatch_type);
  }

  auto *arrayType =
      _typeContext.createArrayType(elementType, elements.size());
  expr->setType(arrayType);
  return success();
}

auto SemaImpl::semaTensorDisposeCall(CallExpr *node) -> MulberryResult {
  auto &expressions = node->expressions();
  if (expressions.size() != 1) {
    auto diagnostic =
        formatNameSizeDiagnostic(diag::func_param, node->name(), 1);
    return emitError(node, diagnostic);
  }

  auto *tensor = expressions.front().get();
  if (sema(tensor) || !tensorElementType(tensor->type()))
    return emitError(tensor, diag::mismatch_type);
  if (checkConstObjectUseAsMutable(tensor))
    return failure();

  setBuiltinType(node, BuiltinTypeKind::Unit);
  return success();
}

auto SemaImpl::semaTensorStorageAllocCall(CallExpr *node,
                                          const Type *expectedType)
    -> MulberryResult {
  if (checkInternalFeature(node->location()))
    return failure();

  auto &expressions = node->expressions();
  if (expressions.size() != 1) {
    auto diagnostic =
        formatNameSizeDiagnostic(diag::func_param, node->name(), 1);
    return emitError(node, diagnostic);
  }

  if (!expectedType || !tensorStorageElementType(expectedType))
    return emitError(node, diag::mismatch_type);
  if (sema(expressions.front().get()) ||
      !isUInt64Type(expressions.front()->type()))
    return emitError(expressions.front().get(), diag::mismatch_type);

  node->setType(expectedType);
  return success();
}

auto SemaImpl::sema(ArrayLiteralExpr *expr) -> MulberryResult {
  return semaDefaultArrayLiteral(expr);
}

auto SemaImpl::sema(ArrayLiteralExpr *expr, const ArrayType *type)
    -> MulberryResult {
  auto &elements = expr->getElements();
  if (elements.size() != type->size())
    return emitError(expr, diag::mismatch_type);

  for (auto &element : elements) {
    if (semaArrayLiteralElement(element.get(), type->elementType()))
      return failure();
  }

  expr->setType(type);
  return success();
}

auto SemaImpl::semaArrayLiteralElement(Expr *expr, const Type *type)
    -> MulberryResult {
  if (auto *arrayLiteral = llvm::dyn_cast<ArrayLiteralExpr>(expr)) {
    auto *arrayType = mulberry::getArrayType(type);
    if (!arrayType)
      return emitError(expr, diag::mismatch_type);
    return sema(arrayLiteral, arrayType);
  }

  if (mulberry::getArrayType(type))
    return emitError(expr, diag::mismatch_type);

  if (auto *decimal = dyn_cast<DecimalLiteralExpr>(expr)) {
    if (isUInt8Type(type)) {
      if (decimal->value() > 255)
        return emitError(expr, diag::mismatch_type);
      expr->setType(type);
      return success();
    }

    if (isUInt64Type(type)) {
      expr->setType(type);
      return success();
    }
  }

  if (sema(expr, type))
    return failure();
  if (!sameType(type, expr->type()))
    return emitError(expr, diag::mismatch_type);
  return success();
}

auto SemaImpl::sema(IndexExpr *expr) -> MulberryResult {
  if (sema(expr->base().get()))
    return failure();

  if (auto *elementType = tensorElementType(expr->base()->type())) {
    if (expr->indices().empty())
      return emitError(expr, diag::mismatch_type);

    for (auto &index : expr->indices()) {
      if (sema(index.get()))
        return failure();
      if (!isUInt64Type(index->type()))
        return emitError(index.get(), diag::mismatch_type);
    }

    expr->setType(elementType);
    expr->setLvalue(true);
    expr->setStdlibTensorIndex();
    return success();
  }

  if (auto *elementType = stdlibListElementType(expr->base()->type())) {
    if (expr->indices().size() != 1)
      return emitError(expr, diag::mismatch_type);
    auto &index = expr->indices().front();
    if (sema(index.get()))
      return failure();
    if (!isUInt64Type(index->type()))
      return emitError(index.get(), diag::mismatch_type);

    std::string getFunctionName;
    std::string setFunctionName;
    if (instantiateGenericFunction(expr, "std.list.List.get",
                                   elementType, getFunctionName) ||
        instantiateGenericFunction(expr, "std.list.List.set",
                                   elementType, setFunctionName))
      return failure();

    expr->setType(elementType);
    expr->setLvalue(true);
    expr->setStdlibListIndex(getFunctionName, setFunctionName);
    return success();
  }

  if (auto *arrayType = mulberry::getArrayType(expr->base()->type())) {
    if (expr->indices().size() != 1)
      return emitError(expr, diag::mismatch_type);
    auto &index = expr->indices().front();
    if (sema(index.get()))
      return failure();
    if (!isUInt64Type(index->type()))
      return emitError(index.get(), diag::mismatch_type);

    expr->setType(arrayType->elementType());
    expr->setLvalue(true);
    expr->setArrayIndex();
    return success();
  }

  auto *ptrType = mulberry::getPtrType(expr->base()->type());
  if (ptrType) {
    if (expr->indices().size() != 1)
      return emitError(expr, diag::mismatch_type);
    auto &index = expr->indices().front();
    if (sema(index.get()))
      return failure();
    if (!isUInt64Type(index->type()))
      return emitError(index.get(), diag::mismatch_type);

    expr->setType(ptrType->pointeeType());
    expr->setLvalue(true);
    expr->setPtrIndex();
    return success();
  }

  return emitError(expr, diag::mismatch_type);
}

auto SemaImpl::sema(IfStat *node) -> MulberryResult {
  auto condition = evaluateComptime(node->conditionExpr().get());
  if (condition.kind == ComptimeEvaluation::Kind::Error)
    return failure();

  if (condition.kind == ComptimeEvaluation::Kind::Value) {
    if (condition.value->kind() != ComptimeValue::Kind::Bool)
      return emitError(node->conditionExpr().get(), diag::expected_bool);

    auto value = condition.value->boolValue();
    node->setComptimeValue(value);
    LLVM_DEBUG(llvm::dbgs() << "evaluate if condition at comptime as `"
                            << (value ? "true" : "false") << "`\n");
    if (value)
      return sema(node->thenBlock().get());
    if (node->hasElseBlock())
      return sema(node->elseBlock().get());
    return success();
  }

  auto conditionExpr = node->conditionExpr().get();
  if (sema(conditionExpr))
    return failure();
  if (!isBoolType(conditionExpr->type()))
    return emitError(conditionExpr, diag::expected_bool);

  auto thenBlock = node->thenBlock().get();
  if (sema(thenBlock))
    return failure();

  if (!node->hasElseBlock()) {
    return success();
  }

  auto elseBlock = node->elseBlock().get();
  if (sema(elseBlock))
    return failure();

  return success();
}

auto SemaImpl::sema(WhileStat *node) -> MulberryResult {
  auto conditionExpr = node->conditionExpr().get();
  if (sema(conditionExpr))
    return failure();
  if (!isBoolType(conditionExpr->type()))
    return emitError(conditionExpr, diag::expected_bool);

  auto bodyBlock = node->bodyBlock().get();
  WhileScope whileScope(_whileDepth);
  if (sema(bodyBlock))
    return failure();

  return success();
}

auto SemaImpl::sema(BreakStat *node) -> MulberryResult {
  if (_whileDepth == 0)
    return emitError(node, diag::loop_control_outside_loop);
  return success();
}

auto SemaImpl::sema(ContinueStat *node) -> MulberryResult {
  if (_whileDepth == 0)
    return emitError(node, diag::loop_control_outside_loop);
  return success();
}

auto SemaImpl::sema(ForStat *node) -> MulberryResult {
  if (sema(node->startExpr().get()) || sema(node->endExpr().get()))
    return failure();

  if (!isUInt64Type(node->startExpr()->type()))
    return emitError(node->startExpr().get(), diag::mismatch_type);
  if (!isUInt64Type(node->endExpr()->type()))
    return emitError(node->endExpr().get(), diag::mismatch_type);

  VariableScope loopScope(_symbols);
  auto *uint64Type = _typeContext.getBuiltinType(BuiltinTypeKind::UInt64);
  if (declareVariable(node->variableName(), uint64Type,
                      /*isConstBinding=*/true,
                      /*canMutateObject=*/false))
    return emitError(node, diag::redefinition_var);

  auto bodyBlock = node->bodyBlock().get();
  if (sema(bodyBlock))
    return failure();

  return success();
}

auto SemaImpl::sema(Stat *node) -> MulberryResult {
  switch (node->getKind()) {
  case Stat::Stat_VariableDecl:
    return sema(cast<VariableStat>(node));
  case Stat::Stat_Expression:
    return sema(cast<ExprStat>(node));
  case Stat::Stat_If:
    return sema(cast<IfStat>(node));
  case Stat::Stat_While:
    return sema(cast<WhileStat>(node));
  case Stat::Stat_For:
    return sema(cast<ForStat>(node));
  case Stat::Stat_Break:
    return sema(cast<BreakStat>(node));
  case Stat::Stat_Continue:
    return sema(cast<ContinueStat>(node));
  case Stat::Stat_Return:
    return sema(cast<ReturnStat>(node));
  }
}

auto SemaImpl::sema(VariableStat *node) -> MulberryResult {
  auto var = node->variable().get();
  auto initExpr = node->init().get();

  std::optional<ComptimeValue> comptimeValue;
  if (node->isConstBinding()) {
    auto evaluation = evaluateComptime(initExpr);
    if (evaluation.kind == ComptimeEvaluation::Kind::Error)
      return failure();
    if (evaluation.kind == ComptimeEvaluation::Kind::Value)
      comptimeValue = evaluation.value;

    // Ordinary consts keep their runtime binding and only cache the value.
    // Reflection-derived initializers cannot enter MLIR, so their declaration
    // is comptime-only.
    if (evaluation.kind == ComptimeEvaluation::Kind::Value &&
        evaluation.isComptimeOnly) {
      auto *valueType = comptimeRuntimeType(*evaluation.value);
      if (node->hasExplicitType()) {
        auto *declaredType = checkType(node->typeNode(), UnitPolicy::Allow);
        if (!declaredType)
          return failure();
        if (!valueType || !sameType(declaredType, valueType))
          return emitError(initExpr, diag::mismatch_type);
        valueType = declaredType;
      }

      node->setType(valueType);
      node->setComptimeValue(*evaluation.value);
      if (declareVariable(var->name(), valueType,
                          /*isConstBinding=*/true,
                          /*canMutateObject=*/false, evaluation.value,
                          /*isComptimeOnly=*/true))
        return emitError(var, diag::redefinition_var);
      return success();
    }
  }

  const Type *varType = nullptr;
  if (node->hasExplicitType()) {
    varType = checkType(node->typeNode(), UnitPolicy::Allow);
    if (!varType || sema(initExpr, varType))
      return failure();
    if (!sameType(varType, initExpr->type()))
      return emitError(initExpr, diag::mismatch_type);
  } else {
    if (sema(initExpr))
      return failure();
    varType = initExpr->type();
    if (!varType)
      return emitError(initExpr, diag::mismatch_type);
  }
  node->setType(varType);

  auto canMutateObject = node->canMutateObject();
  if (canMutateObject && isSourceObjectType(varType) &&
      !canMutateObjectReference(initExpr))
    return emitError(initExpr, diag::readonly_to_mutable_binding);
  if (declareVariable(var->name(), varType, node->isConstBinding(),
                      canMutateObject, std::move(comptimeValue)))
    return emitError(var, diag::redefinition_var);
  return success();
}

auto SemaImpl::sema(ExprStat *node) -> MulberryResult {
  return sema(node->expression().get());
}

auto SemaImpl::sema(ReturnStat *node) -> MulberryResult {
  if (!_currentFunctionReturnType)
    return emitError(node, diag::return_outside_function);

  if (!node->hasExpression()) {
    if (!isUnitType(_currentFunctionReturnType))
      return emitError(node, diag::wrong_return_type);
    return success();
  }

  auto expression = node->expression().get();
  if (sema(expression, _currentFunctionReturnType))
    return failure();
  if (!sameReturnType(_currentFunctionReturnType, expression->type()))
    return emitError(expression, diag::wrong_return_type);
  return success();
}

namespace mulberry {

auto sema(const llvm::SourceMgr &sourceManager, Module &moduleAST)
    -> MulberryResult {
  return SemaImpl(sourceManager).sema(moduleAST);
}

auto sema(const llvm::SourceMgr &sourceManager, Module &moduleAST,
          const std::map<std::string, std::string> &importAliases)
    -> MulberryResult {
  return SemaImpl(sourceManager, importAliases).sema(moduleAST);
}

} // end namespace mulberry
