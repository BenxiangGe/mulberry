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

struct ResultTypeArguments {
  const Type *valueType;
  const Type *errorType;
};

auto getResultTypeArguments(const Type *type)
    -> std::optional<ResultTypeArguments> {
  auto *dataType = getDataType(type);
  if (!dataType || dataType->declarationName() != "std.result.Result")
    return std::nullopt;

  auto &arguments = dataType->arguments();
  auto &constructors = dataType->constructors();
  if (arguments.size() != 2 ||
      arguments[0].kind() != ComptimeValue::Kind::Type ||
      arguments[1].kind() != ComptimeValue::Kind::Type ||
      constructors.size() != 2 ||
      constructors[0].name() != "std.result.Ok" ||
      constructors[1].name() != "std.result.Err" ||
      constructors[0].payloadTypes().size() != 1 ||
      constructors[1].payloadTypes().size() != 1 ||
      !sameType(constructors[0].payloadTypes()[0], arguments[0].type()) ||
      !sameType(constructors[1].payloadTypes()[0], arguments[1].type()))
    return std::nullopt;

  return ResultTypeArguments{arguments[0].type(), arguments[1].type()};
}

auto substituteExpr(const Expr *node,
                    const std::vector<TypeSubstitution> &substitutions)
    -> std::unique_ptr<Expr>;

auto substituteBlockExpr(const BlockExpr *node,
                         const std::vector<TypeSubstitution> &substitutions)
    -> std::unique_ptr<BlockExpr>;

auto cloneDataPattern(const DataPattern *pattern)
    -> std::unique_ptr<DataPattern> {
  VectorUniquePtr<VariableExpr> bindings;
  for (auto &binding : pattern->bindings())
    bindings.push_back(std::make_unique<VariableExpr>(
        binding->location(), binding->name()));
  return std::make_unique<DataPattern>(
      pattern->location(), pattern->constructorName(), std::move(bindings));
}

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

auto formatTypeTraitDiagnostic(const char *diagnostic, const Type *type,
                               std::string_view traitName) -> std::string {
  auto message = replacePlaceholder(diagnostic, "%t", formatType(type));
  return replacePlaceholder(message, "%s", traitName);
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
  if (auto *matchExpr = dyn_cast<MatchExpr>(node)) {
    for (auto &arm : matchExpr->arms())
      if (containsReturnStat(arm->bodyBlock().get()) ||
          containsReturnStat(arm->resultExpr().get()))
        return true;
  }
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
  if (auto *matchStat = dyn_cast<MatchStat>(node)) {
    for (auto &arm : matchStat->arms())
      if (containsReturnStat(arm->bodyBlock().get()))
        return true;
    return false;
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
  return isIntegerType(type) || getStructType(type) || getDataType(type) ||
         getArrayType(type);
}

auto isMutableSourceObjectType(const Type *type) -> bool {
  if (auto *ptrType = getPtrType(type))
    type = ptrType->pointeeType();
  return !isIntegerType(type) && isSourceObjectType(type);
}

auto isIntegerWidening(const Type *sourceType, const Type *targetType)
    -> bool {
  return isIntegerType(targetType) &&
         (isUInt8Type(sourceType) || isUInt64Type(sourceType));
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

  if (auto *functionType = dyn_cast<FunctionTypeNode>(node)) {
    VectorUniquePtr<TypeNode> parameterTypes;
    for (auto &parameterType : functionType->parameterTypes())
      parameterTypes.push_back(cloneTypeNode(parameterType.get()));
    return std::make_unique<FunctionTypeNode>(
        functionType->location(), std::move(parameterTypes),
        functionType->parameterCanMutateObject(),
        cloneTypeNode(functionType->returnTypeNode()));
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

  if (auto *dataType = getDataType(type)) {
    if (dataType->arguments().empty())
      return std::make_unique<NamedTypeNode>(
          location, dataType->declarationName());

    std::vector<ComptimeArg> arguments;
    for (auto &argument : dataType->arguments()) {
      if (argument.kind() == ComptimeValue::Kind::UInt64) {
        arguments.push_back(ComptimeArg(location, argument.uint64Value()));
        continue;
      }
      arguments.push_back(
          ComptimeArg(typeToTypeNode(argument.type(), location)));
    }
    return std::make_unique<GenericTypeNode>(
        location, dataType->declarationName(), std::move(arguments));
  }

  if (auto *arrayType = getArrayType(type)) {
    std::vector<ComptimeArg> arguments;
    arguments.push_back(ComptimeArg(
        typeToTypeNode(arrayType->elementType(), location)));
    arguments.push_back(ComptimeArg(location, arrayType->size()));
    return std::make_unique<GenericTypeNode>(
        location, "Array", std::move(arguments));
  }

  if (auto *functionType = getFunctionType(type)) {
    VectorUniquePtr<TypeNode> parameterTypes;
    for (auto *parameterType : functionType->parameterTypes())
      parameterTypes.push_back(typeToTypeNode(parameterType, location));
    return std::make_unique<FunctionTypeNode>(
        location, std::move(parameterTypes),
        functionType->parameterCanMutateObject(),
        typeToTypeNode(functionType->returnType(), location));
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

  if (auto *functionType = dyn_cast<FunctionTypeNode>(node)) {
    VectorUniquePtr<TypeNode> parameterTypes;
    for (auto &parameterType : functionType->parameterTypes()) {
      parameterTypes.push_back(
          substituteTypeNode(parameterType.get(), substitution));
    }
    return std::make_unique<FunctionTypeNode>(
        functionType->location(), std::move(parameterTypes),
        functionType->parameterCanMutateObject(),
        substituteTypeNode(functionType->returnTypeNode(), substitution));
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

  if (auto *functionType = dyn_cast<FunctionTypeNode>(node)) {
    for (auto &parameterType : functionType->parameterTypes())
      if (containsComptimeParameter(parameterType.get(), parameters))
        return true;
    return containsComptimeParameter(functionType->returnTypeNode(),
                                     parameters);
  }

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

    if (auto *matchStat = dyn_cast<MatchStat>(statement.get())) {
      VectorUniquePtr<MatchArm> arms;
      for (auto &arm : matchStat->arms()) {
        arms.push_back(std::make_unique<MatchArm>(
            arm->location(), cloneDataPattern(arm->pattern().get()),
            substituteBlockExpr(arm->bodyBlock().get(), substitutions)));
      }
      statements.push_back(std::make_unique<MatchStat>(
          matchStat->location(),
          substituteExpr(matchStat->value().get(), substitutions),
          std::move(arms)));
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
  case Expr::Expr_Lambda: {
    auto *expr = cast<LambdaExpr>(node);
    std::vector<LambdaExpr::Parameter> parameters;
    for (auto &parameter : expr->parameters())
      parameters.push_back({parameter.location, parameter.name});
    return std::make_unique<LambdaExpr>(
        expr->location(), std::move(parameters),
        substituteExpr(expr->body().get(), substitutions));
  }
  case Expr::Expr_Match: {
    auto *expr = cast<MatchExpr>(node);
    VectorUniquePtr<MatchExprArm> arms;
    for (auto &arm : expr->arms())
      arms.push_back(std::make_unique<MatchExprArm>(
          arm->location(), cloneDataPattern(arm->pattern().get()),
          substituteBlockExpr(arm->bodyBlock().get(), substitutions),
          substituteExpr(arm->resultExpr().get(), substitutions)));
    return std::make_unique<MatchExpr>(
        expr->location(),
        substituteExpr(expr->value().get(), substitutions), std::move(arms));
  }
  case Expr::Expr_Try: {
    auto *expr = cast<TryExpr>(node);
    return std::make_unique<TryExpr>(
        expr->location(),
        substituteExpr(expr->value().get(), substitutions));
  }
  case Expr::Expr_IntegerLiteral: {
    auto *expr = cast<IntegerLiteralExpr>(node);
    return std::make_unique<IntegerLiteralExpr>(
        expr->location(), std::string(expr->spelling()));
  }
  case Expr::Expr_IntegerWiden: {
    auto *expr = cast<IntegerWidenExpr>(node);
    return std::make_unique<IntegerWidenExpr>(
        expr->location(), substituteExpr(expr->value().get(), substitutions));
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
        return std::make_unique<IntegerLiteralExpr>(
            expr->location(), std::to_string(*substitution.uint64Value));
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
  case Expr::Expr_DataConstructor: {
    auto *expr = cast<DataConstructorExpr>(node);
    VectorUniquePtr<Expr> expressions;
    for (auto &argument : expr->expressions())
      expressions.push_back(substituteExpr(argument.get(), substitutions));
    return std::make_unique<DataConstructorExpr>(
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

auto traitMethodSignatureMatches(const TraitMethodDecl *method,
                                 const FunctionSymbol *signature,
                                 const Type *targetType) -> bool {
  auto &parameterTypes = signature->type->parameterTypes();
  auto &parameterMutability = signature->type->parameterCanMutateObject();
  auto valid =
      parameterTypes.size() == method->parameters().size() + 1 &&
      sameType(parameterTypes.front(), targetType) &&
      parameterMutability.front() == method->receiverCanMutateObject() &&
      sameType(signature->type->returnType(), method->returnType());
  for (size_t index = 0; valid && index < method->parameters().size();
       ++index) {
    auto &parameter = method->parameters()[index];
    valid = sameType(parameterTypes[index + 1], parameter->type()) &&
            parameterMutability[index + 1] == parameter->canMutateObject();
  }
  return valid;
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
    for (auto &function : _lambdaFunctions)
      declarations.push_back(std::move(function));
    node.setDeclarations(std::move(declarations));

    std::string mainName = node.packageName().empty()
                               ? "main"
                               : std::string(node.packageName()) + ".main";
    auto *mainSignature = lookupFunction(mainName);
    if (!mainSignature ||
        !mainSignature->type->parameterTypes().empty() ||
        !isUInt64Type(mainSignature->type->returnType()))
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
  std::map<std::string, DataType *> _dataTypes;
  std::map<std::string, const FunctionSymbol *> _instantiatedFunctionSymbols;
  std::map<std::string, std::string> _functionPackages;
  std::map<std::string, std::string> _genericFunctionPackages;
  std::map<std::string, std::string> _instantiatedFunctionPackages;
  std::map<std::string, BuiltinHandler, std::less<>> _builtinHandlers;
  VectorUniquePtr<FunctionDecl> _instantiatedFunctions;
  VectorUniquePtr<FunctionDecl> _lambdaFunctions;
  const std::map<std::string, std::string> &_importAliases =
      emptyImportAliases();
  std::string _currentPackageName;
  const Type *_currentFunctionReturnType = nullptr;
  int _whileDepth = 0;
  int _noncapturingLambdaDepth = 0;
  uint64_t _lambdaCounter = 0;

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
  auto semaFunctionSignature(Prototype *node, bool isExtern = false)
      -> MulberryResult;
  auto sema(FunctionDecl *node) -> MulberryResult;
  auto sema(StructDecl *node) -> MulberryResult;
  auto sema(DataDecl *node) -> MulberryResult;
  auto sema(ComptimeTypeAliasDecl *node) -> MulberryResult;
  auto sema(TraitDecl *node) -> MulberryResult;
  auto sema(ImplDecl *node) -> MulberryResult;
  auto resolveTraitConstraints(const Node *node,
                               std::vector<ComptimeParam> &parameters)
      -> MulberryResult;
  auto traitMethodFunctionName(const TraitDecl *trait, const Type *targetType,
                               std::string_view methodName) const
      -> std::string;
  auto instantiateTraitDefaultMethod(const TraitDecl *trait,
                                     const TraitMethodDecl *method,
                                     const Type *targetType,
                                     std::string &functionName)
      -> MulberryResult;
  auto instantiateGenericTraitMethod(const ImplDecl *impl,
                                     const FunctionDecl *method,
                                     const TraitMethodDecl *contract,
                                     const Type *targetType,
                                     std::string &functionName)
      -> MulberryResult;
  auto genericTraitImplementationMatches(const ImplDecl *impl,
                                         const Type *type, bool &matches)
      -> MulberryResult;
  auto findMatchingGenericTraitImplementations(
      const Type *type, const TraitDecl *trait,
      std::vector<const ImplDecl *> &implementations) -> MulberryResult;
  auto materializeGenericTraitImplementation(const Node *diagnosticNode,
                                             const Type *type,
                                             const TraitDecl *trait,
                                             bool &matched) -> MulberryResult;
  auto materializeGenericTraitMethod(const Node *diagnosticNode,
                                     const Type *type,
                                     std::string_view methodName,
                                     std::string &functionName)
      -> MulberryResult;
  auto checkTraitConstraints(
      const Node *node, const std::vector<ComptimeParam> &parameters,
      const std::vector<InferredComptimeArgument> &arguments)
      -> MulberryResult;
  auto typeConformsToTrait(const Node *diagnosticNode, const Type *type,
                           const TraitDecl *trait, bool &conforms)
      -> MulberryResult;

  // Expressions
  auto sema(Expr *node) -> MulberryResult;
  auto sema(Expr *node, const Type *type) -> MulberryResult;
  auto semaExpected(std::unique_ptr<Expr> &node, const Type *type)
      -> MulberryResult;
  auto sema(UnitExpr *node) -> MulberryResult;
  auto sema(BlockExpr *node) -> MulberryResult;
  auto sema(LambdaExpr *node) -> MulberryResult;
  auto sema(LambdaExpr *node, const FunctionType *functionType)
      -> MulberryResult;
  auto sema(MatchExpr *node, const Type *expectedType = nullptr)
      -> MulberryResult;
  auto sema(TryExpr *node) -> MulberryResult;
  auto semaLambda(LambdaExpr *node,
                  const std::vector<const Type *> &parameterTypes,
                  const std::vector<bool> &parameterCanMutateObject,
                  const Type *expectedReturnType,
                  std::string_view packageName) -> MulberryResult;
  auto sema(CallExpr *node) -> MulberryResult;
  auto sema(DataConstructorExpr *node) -> MulberryResult;
  auto sema(DataConstructorExpr *node, const DataType *dataType)
      -> MulberryResult;
  auto semaIndirectCall(CallExpr *node, const FunctionType *functionType,
                        const Type *expectedType = nullptr) -> MulberryResult;
  auto sema(StructLiteralExpr *node) -> MulberryResult;
  auto sema(VariableExpr *node) -> MulberryResult;
  auto sema(MemberExpr *node) -> MulberryResult;
  auto sema(AssignExpr *node) -> MulberryResult;
  auto sema(IntegerLiteralExpr *node) -> MulberryResult;
  auto sema(IntegerLiteralExpr *node, const Type *type) -> MulberryResult;
  auto sema(IntegerWidenExpr *node) -> MulberryResult;
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
  auto sema(BinaryExpr *node, const Type *expectedType) -> MulberryResult;
  auto checkStringConcatFunction(Expr *node, const Type *stringType)
      -> MulberryResult;
  auto semaFormatValueCall(std::unique_ptr<Expr> &expression,
                           const Type *stringType) -> MulberryResult;
  auto checkAssignable(const Expr *expr) -> MulberryResult;
  auto checkConstObjectUseAsMutable(const Expr *expr) -> MulberryResult;
  auto canMutateObjectReference(const Expr *expr) -> bool;
  auto checkMutableObjectArgument(const FunctionType *functionType, size_t index,
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
  auto semaArrayLiteralElement(std::unique_ptr<Expr> &expr, const Type *type)
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
  auto sema(MatchStat *node) -> MulberryResult;
  auto checkMatchPattern(DataPattern *pattern, const DataType *dataType,
                         std::vector<bool> &covered,
                         const DataConstructor *&constructor)
      -> MulberryResult;
  auto declareMatchPatternBindings(DataPattern *pattern,
                                   const DataConstructor *constructor)
      -> MulberryResult;
  auto checkExhaustiveMatch(const Node *node,
                            const std::vector<bool> &covered)
      -> MulberryResult;
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
    declareBuiltinType(BuiltinTypeKind::Integer);
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

  auto lookupTrait(std::string_view name) -> const TraitDecl * {
    if (auto *trait = _symbols.lookupTrait(name))
      return trait->decl;

    auto importedName = canonicalizeImportedName(name);
    if (auto *trait = _symbols.lookupTrait(importedName))
      return trait->decl;

    auto qualifiedName = qualifyCurrentPackageName(name);
    if (auto *trait = _symbols.lookupTrait(qualifiedName))
      return trait->decl;

    return nullptr;
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

  auto dataDeclName(std::string_view name) -> std::string {
    if (_symbols.lookupDataDecl(name))
      return std::string(name);

    auto importedName = canonicalizeImportedName(name);
    if (_symbols.lookupDataDecl(importedName))
      return importedName;

    auto packageName = qualifyCurrentPackageName(name);
    if (_symbols.lookupDataDecl(packageName))
      return packageName;

    return {};
  }

  auto lookupDataConstructor(std::string_view name,
                             std::string &resolvedName)
      -> const DataConstructorSymbol * {
    if (auto *constructor = _symbols.lookupDataConstructor(name)) {
      resolvedName = std::string(name);
      return constructor;
    }

    auto importedName = canonicalizeImportedName(name);
    if (auto *constructor =
            _symbols.lookupDataConstructor(importedName)) {
      resolvedName = importedName;
      return constructor;
    }

    auto packageName = qualifyCurrentPackageName(name);
    if (auto *constructor =
            _symbols.lookupDataConstructor(packageName)) {
      resolvedName = packageName;
      return constructor;
    }

    return nullptr;
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

  class IsolatedWhileScope {
  public:
    explicit IsolatedWhileScope(int &whileDepth)
        : _whileDepth(whileDepth), _oldWhileDepth(whileDepth) {
      _whileDepth = 0;
    }

    ~IsolatedWhileScope() { _whileDepth = _oldWhileDepth; }

  private:
    int &_whileDepth;
    int _oldWhileDepth;
  };

  class NoncapturingLambdaScope {
  public:
    explicit NoncapturingLambdaScope(int &depth) : _depth(depth) { ++_depth; }

    ~NoncapturingLambdaScope() { --_depth; }

  private:
    int &_depth;
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

  auto declareFunction(std::string_view name, const FunctionType *type,
                       bool isExtern,
                       std::string_view packageName = {})
      -> MulberryResult {
    if (packageName.empty())
      packageName = _currentPackageName;
    _functionPackages[std::string(name)] = std::string(packageName);
    return _symbols.declareFunction(name, type, isExtern);
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

  auto genericTypeName(std::string_view declarationName,
                       const std::vector<ComptimeArgument> &arguments) const
      -> std::string {
    std::string name = mangleTypeName(std::string(declarationName));
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

    if (auto *functionType = dyn_cast<FunctionTypeNode>(typeNode)) {
      for (auto &parameterType : functionType->parameterTypes())
        if (hasComputedType(parameterType.get(), visitingAliases))
          return true;
      return hasComputedType(functionType->returnTypeNode(), visitingAliases);
    }

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

    if (auto *functionPattern = dyn_cast<FunctionTypeNode>(pattern)) {
      auto *functionType = getFunctionType(actualType);
      if (!functionType ||
          functionPattern->parameterTypes().size() !=
              functionType->parameterTypes().size() ||
          functionPattern->parameterCanMutateObject() !=
              functionType->parameterCanMutateObject())
        return false;

      for (size_t i = 0; i < functionPattern->parameterTypes().size(); ++i) {
        if (!matchGenericType(functionPattern->parameterTypes()[i].get(),
                              functionType->parameterTypes()[i], parameters,
                              arguments, arrayLeafConstraints))
          return false;
      }
      return matchGenericType(functionPattern->returnTypeNode(),
                              functionType->returnType(), parameters,
                              arguments, arrayLeafConstraints);
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

      auto dataName = dataDeclName(genericPattern->name());
      auto *dataType = getDataType(actualType);
      auto &patternArguments = genericPattern->arguments();
      if (dataType && dataType->declarationName() == dataName) {
        auto &actualArguments = dataType->arguments();
        if (patternArguments.size() != actualArguments.size())
          return false;
        for (size_t i = 0; i < patternArguments.size(); ++i)
          if (!matchComptimeArgument(patternArguments[i], actualArguments[i],
                                     parameters, arguments,
                                     arrayLeafConstraints))
            return false;
        return true;
      }

      auto aliasName = comptimeTypeAliasName(genericPattern->name());
      auto *structType = getStructType(actualType);
      auto *origin = structType ? structType->origin() : nullptr;
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

  auto resolveType(const FunctionTypeNode *typeNode) -> const Type * {
    std::vector<const Type *> parameterTypes;
    for (auto &parameterTypeNode : typeNode->parameterTypes()) {
      auto *parameterType =
          checkType(parameterTypeNode.get(), UnitPolicy::Reject);
      if (!parameterType)
        return nullptr;
      parameterTypes.push_back(parameterType);
    }

    auto *returnType = checkType(typeNode->returnTypeNode(), UnitPolicy::Allow);
    if (!returnType)
      return nullptr;
    return _typeContext.createFunctionType(
        std::move(parameterTypes), typeNode->parameterCanMutateObject(),
        returnType);
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

  auto completeDataType(const DataDecl *decl, DataType *dataType,
                        const std::vector<TypeSubstitution> &substitutions)
      -> MulberryResult {
    std::vector<DataConstructor> constructors;
    for (auto &constructorDecl : decl->constructors()) {
      std::vector<const Type *> payloadTypes;
      for (auto &payloadTypeNode : constructorDecl->payloadTypes()) {
        auto concreteTypeNode =
            substituteTypeNode(payloadTypeNode.get(), substitutions);
        auto *payloadType =
            checkType(concreteTypeNode.get(), UnitPolicy::Allow);
        if (!payloadType)
          return failure();
        payloadTypes.push_back(payloadType);
      }
      constructors.push_back(DataConstructor(
          constructorDecl->name(), std::move(payloadTypes)));
    }
    dataType->setConstructors(std::move(constructors));
    return success();
  }

  auto instantiateDataType(
      const DataDecl *decl, const std::vector<ComptimeArgument> &arguments,
      const std::vector<TypeSubstitution> &substitutions) -> DataType * {
    auto concreteName = genericTypeName(decl->name(), arguments);
    auto cached = _dataTypes.find(concreteName);
    if (cached != _dataTypes.end())
      return cached->second;

    // Cache the shell before resolving payloads so a recursive constructor
    // such as Node(T, Tree<T>, Tree<T>) resolves to this canonical DataType.
    auto *dataType = _typeContext.createDataType(
        concreteName, decl->name(), toComptimeValues(arguments));
    _dataTypes[concreteName] = dataType;
    LLVM_DEBUG(llvm::dbgs() << "instantiate data type `"
                            << formatType(dataType) << "`\n");

    PackageScope packageScope(_currentPackageName,
                              packageNameOf(decl->name()));
    if (completeDataType(decl, dataType, substitutions))
      return nullptr;
    return dataType;
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

    auto declarationName = dataDeclName(typeNode->name());
    if (!declarationName.empty()) {
      auto *decl = _symbols.lookupDataDecl(declarationName)->decl;
      if (typeNode->arguments().size() != decl->parameters().size()) {
        emitError(typeNode, diag::mismatch_type);
        return nullptr;
      }

      std::vector<ComptimeArgument> arguments;
      std::vector<TypeSubstitution> substitutions;
      for (size_t i = 0; i < typeNode->arguments().size(); ++i) {
        auto &argument = typeNode->arguments()[i];
        auto &parameter = decl->parameters()[i];
        if (argument.kind() == ComptimeArg::Kind::Type &&
            parameter.kind == ComptimeParam::Kind::Type) {
          auto *argumentType = resolveType(argument.typeNode());
          if (!argumentType)
            return nullptr;
          auto argumentTypeNode =
              typeToTypeNode(argumentType, argument.typeNode()->location());
          substitutions.push_back(TypeSubstitution{
              parameter.name, argumentTypeNode.get(), std::nullopt});
          ComptimeArgument resolvedArgument;
          resolvedArgument.kind = argument.kind();
          resolvedArgument.type = argumentType;
          resolvedArgument.typeNode = std::move(argumentTypeNode);
          arguments.push_back(std::move(resolvedArgument));
          continue;
        }

        if (argument.kind() == ComptimeArg::Kind::UInt64 &&
            parameter.kind == ComptimeParam::Kind::UInt64) {
          substitutions.push_back(TypeSubstitution{
              parameter.name, nullptr, argument.uint64Value()});
          ComptimeArgument resolvedArgument;
          resolvedArgument.kind = argument.kind();
          resolvedArgument.uint64Value = argument.uint64Value();
          arguments.push_back(std::move(resolvedArgument));
          continue;
        }

        emitError(typeNode, diag::mismatch_type);
        return nullptr;
      }
      return instantiateDataType(decl, arguments, substitutions);
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
      auto structName = genericTypeName(aliasName, arguments);
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

    if (auto *functionType = dyn_cast<FunctionTypeNode>(typeNode))
      return resolveType(functionType);

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
  case Decl::Decl_Data:
    return sema(cast<DataDecl>(node));
  case Decl::Decl_ComptimeTypeAlias:
    return sema(cast<ComptimeTypeAliasDecl>(node));
  case Decl::Decl_Trait:
    return sema(cast<TraitDecl>(node));
  case Decl::Decl_Impl:
    return sema(cast<ImplDecl>(node));
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
    auto *parameterType = signature->type->parameterTypes()[i];
    auto canMutateObject =
        signature->type->parameterCanMutateObject()[i];
    parameter->setType(parameterType);
    if (declareVariable(parameter->variable()->name(), parameterType,
                        !canMutateObject, canMutateObject))
      return emitError(parameter->variable().get(), diag::redefinition_var);
  }
  node->setType(signature->type->returnType());
  return success();
}

auto SemaImpl::semaFunctionSignature(Prototype *node, bool isExtern)
    -> MulberryResult {
  std::vector<const Type *> parameterTypes;
  std::vector<bool> parameterCanMutateObject;
  if (semaFunctionParameters(node, parameterTypes, parameterCanMutateObject))
    return failure();

  auto *returnType = resolveType(node->returnTypeNode());
  if (!returnType)
    return failure();
  node->setType(returnType);

  auto name = node->id()->name();
  auto *functionType = _typeContext.createFunctionType(
      std::move(parameterTypes), std::move(parameterCanMutateObject),
      returnType);
  if (declareFunction(name, functionType, isExtern)) {
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
    auto result = semaFunctionSignature(node->proto().get(), true);
    _symbols.resetVariables();
    return result;
  }

  if (node->proto()->isGeneric()) {
    if (resolveTraitConstraints(node->proto().get(),
                                node->proto()->comptimeParameters()))
      return failure();
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
                                          signature->type->returnType());
  if (sema(node->body().get()))
    return failure();

  auto hasReturn = containsReturnStat(node->body().get());
  if (!isUnitType(signature->type->returnType()) && !hasReturn)
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
    if (!typeParameters.empty()) {
      std::vector<ComptimeParam> parameters(typeParameters.begin(),
                                            typeParameters.end());
      parameters.insert(parameters.end(),
                        prototype->comptimeParameters().begin(),
                        prototype->comptimeParameters().end());
      prototype->setComptimeParameters(std::move(parameters));
    }

    if (prototype->isGeneric()) {
      if (resolveTraitConstraints(prototype,
                                  prototype->comptimeParameters()))
        return failure();
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

auto SemaImpl::sema(DataDecl *node) -> MulberryResult {
  if (_symbols.lookupType(node->name()) ||
      _symbols.lookupComptimeTypeAlias(node->name()) ||
      _symbols.lookupDataDecl(node->name()))
    return emitError(node, diag::redefinition_type);

  if (_symbols.declareDataDecl(node->name(), node))
    return emitError(node, diag::redefinition_type);

  for (const auto &indexedConstructor :
       llvm::enumerate(node->constructors())) {
    auto &constructor = indexedConstructor.value();
    if (_symbols.declareDataConstructor(
            constructor->name(), node, indexedConstructor.index()))
      return emitError(constructor.get(),
                       diag::redefinition_data_constructor);
  }

  if (node->isGeneric())
    return success();

  std::vector<ComptimeArgument> arguments;
  auto concreteName = genericTypeName(node->name(), arguments);
  auto *dataType = _typeContext.createDataType(
      concreteName, node->name(), {});
  _dataTypes[concreteName] = dataType;

  // Publish a non-generic shell before resolving its constructors so direct
  // self-reference can resolve through the ordinary type symbol table.
  if (declareType(node->name(), dataType))
    return emitError(node, diag::redefinition_type);
  PackageScope packageScope(_currentPackageName,
                            packageNameOf(node->name()));
  return completeDataType(node, dataType, {});
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

auto SemaImpl::resolveTraitConstraints(
    const Node *node, std::vector<ComptimeParam> &parameters)
    -> MulberryResult {
  for (auto &parameter : parameters) {
    if (!parameter.hasTraitConstraint())
      continue;
    if (parameter.kind != ComptimeParam::Kind::Type)
      return emitError(node, diag::mismatch_type);

    parameter.trait = lookupTrait(parameter.traitName);
    if (!parameter.trait) {
      auto diagnostic =
          formatNameDiagnostic(diag::undefined_trait, parameter.traitName);
      return emitError(node, diagnostic);
    }
  }
  return success();
}

auto SemaImpl::sema(TraitDecl *node) -> MulberryResult {
  if (_symbols.lookupTrait(node->name())) {
    auto diagnostic = formatNameDiagnostic(diag::redefinition_trait,
                                           node->name());
    return emitError(node, diagnostic);
  }
  if (_symbols.declareTrait(node->name(), node)) {
    auto diagnostic = formatNameDiagnostic(diag::redefinition_trait,
                                           node->name());
    return emitError(node, diagnostic);
  }

  PackageScope packageScope(_currentPackageName,
                            packageNameOf(node->name()));
  NameSet methodNames;
  for (auto &method : node->methods()) {
    if (!declareName(methodNames, method->name())) {
      auto diagnostic = formatNameDiagnostic(diag::redefinition_func,
                                             method->name());
      return emitError(method.get(), diagnostic);
    }

    NameSet parameterNames;
    for (auto &parameter : method->parameters()) {
      auto *type = checkType(parameter->typeNode(), UnitPolicy::Reject);
      if (!type)
        return failure();
      parameter->setType(type);
      if (!declareName(parameterNames, parameter->variable()->name()))
        return emitError(parameter->variable().get(), diag::redefinition_var);
    }

    auto *returnType = resolveType(method->returnTypeNode());
    if (!returnType)
      return failure();
    method->setReturnType(returnType);
  }
  return success();
}

auto SemaImpl::traitMethodFunctionName(const TraitDecl *trait,
                                       const Type *targetType,
                                       std::string_view methodName) const
    -> std::string {
  std::string functionName(trait->name());
  functionName += "__";
  functionName += mangleTypeName(formatType(targetType));
  functionName += ".";
  functionName += methodName;
  return functionName;
}

auto SemaImpl::instantiateTraitDefaultMethod(
    const TraitDecl *trait, const TraitMethodDecl *method,
    const Type *targetType, std::string &functionName) -> MulberryResult {
  functionName = traitMethodFunctionName(trait, targetType, method->name());
  if (lookupFunction(functionName))
    return success();

  VectorUniquePtr<ParameterDecl> parameters;
  auto receiver = std::make_unique<VariableExpr>(method->location(), "self");
  parameters.push_back(std::make_unique<ParameterDecl>(
      method->location(), std::move(receiver),
      typeToTypeNode(targetType, method->location()),
      method->receiverCanMutateObject()));
  for (auto &parameter : method->parameters()) {
    auto variable = std::make_unique<VariableExpr>(
        parameter->variable()->location(), parameter->variable()->name());
    parameters.push_back(std::make_unique<ParameterDecl>(
        parameter->location(), std::move(variable),
        cloneTypeNode(parameter->typeNode()), parameter->canMutateObject()));
  }

  auto name = std::make_unique<FunctionName>(method->location(), functionName);
  auto prototype = std::make_unique<Prototype>(
      method->location(), std::move(name), std::move(parameters),
      cloneTypeNode(method->returnTypeNode()));
  prototype->setIsMethod(true);
  auto function = std::make_unique<FunctionDecl>(
      method->location(), std::move(prototype),
      substituteBlockExpr(method->body().get(), {}));

  // The body belongs to the trait's package, not to the concrete impl site.
  VariableScope signatureScope(_symbols);
  PackageScope packageScope(_currentPackageName, packageNameOf(trait->name()));
  if (semaFunctionSignature(function->proto().get()))
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "instantiate default trait method `"
                          << functionName << "`\n");
  _instantiatedFunctions.push_back(std::move(function));
  return success();
}

auto SemaImpl::instantiateGenericTraitMethod(
    const ImplDecl *impl, const FunctionDecl *method,
    const TraitMethodDecl *contract, const Type *targetType,
    std::string &functionName) -> MulberryResult {
  functionName = traitMethodFunctionName(impl->trait(), targetType,
                                         method->proto()->id()->name());
  if (lookupFunction(functionName))
    return success();

  auto argumentTypeNode = typeToTypeNode(targetType, method->location());
  auto function = instantiateFunctionDecl(
      method, functionName,
      std::vector<TypeSubstitution>{TypeSubstitution{
          impl->comptimeParameters().front().name, argumentTypeNode.get(),
          std::nullopt}});
  function->proto()->setIsMethod(true);
  _instantiatedFunctionPackages[functionName] = impl->packageName();

  VariableScope signatureScope(_symbols);
  PackageScope packageScope(_currentPackageName, impl->packageName());
  if (semaFunctionSignature(function->proto().get()))
    return failure();
  auto *signature = lookupFunction(functionName);
  if (!signature)
    return failure();
  if (!traitMethodSignatureMatches(contract, signature, targetType)) {
    auto diagnostic = formatNameDiagnostic(diag::trait_method_signature,
                                           method->proto()->id()->name());
    return emitError(method->proto()->id().get(), diagnostic);
  }

  LLVM_DEBUG(llvm::dbgs() << "instantiate conditional trait method `"
                          << functionName << "`\n");
  _instantiatedFunctions.push_back(std::move(function));
  return success();
}

auto SemaImpl::sema(ImplDecl *node) -> MulberryResult {
  PackageScope packageScope(_currentPackageName, node->packageName());
  auto *trait = lookupTrait(node->traitName());
  if (!trait) {
    auto diagnostic = formatNameDiagnostic(diag::undefined_trait,
                                           node->traitName());
    return emitError(node, diagnostic);
  }

  node->setTrait(trait);
  std::map<std::string, const TraitMethodDecl *, std::less<>> contracts;
  for (auto &method : trait->methods())
    contracts.insert({std::string(method->name()), method.get()});

  if (node->isGeneric()) {
    auto &parameters = node->comptimeParameters();
    auto *targetType = dyn_cast<NamedTypeNode>(node->targetTypeNode());
    if (parameters.size() != 1 ||
        parameters.front().kind != ComptimeParam::Kind::Type ||
        parameters.front().hasTraitConstraint() || !targetType ||
        targetType->name() != parameters.front().name ||
        !node->whereCondition())
      return emitError(node, diag::mismatch_type);

    NameSet implementedMethods;
    for (auto &method : node->methods()) {
      auto *prototype = method->proto().get();
      auto methodName = std::string(prototype->id()->name());
      if (contracts.find(methodName) == contracts.end() ||
          !declareName(implementedMethods, methodName) ||
          prototype->isGeneric()) {
        auto diagnostic =
            formatNameDiagnostic(diag::trait_method_signature, methodName);
        return emitError(prototype->id().get(), diagnostic);
      }
    }

    for (auto &contract : contracts) {
      if (implementedMethods.find(contract.first) != implementedMethods.end() ||
          contract.second->hasDefaultBody())
        continue;
      auto diagnostic =
          formatNameDiagnostic(diag::missing_trait_method, contract.first);
      return emitError(node, diagnostic);
    }

    if (_symbols.declareGenericTraitImplementation(node))
      return emitError(node, diag::mismatch_type);
    LLVM_DEBUG(llvm::dbgs() << "register conditional trait implementation `"
                            << trait->name() << "` for `"
                            << parameters.front().name << "`\n");
    return success();
  }

  auto *targetType = checkType(node->targetTypeNode(), UnitPolicy::Reject);
  if (!targetType)
    return failure();
  node->setTargetType(targetType);

  if (_symbols.lookupTraitImplementation(trait, targetType)) {
    auto diagnostic = formatNameDiagnostic(diag::redefinition_trait_impl,
                                           trait->name());
    return emitError(node, diagnostic);
  }

  NameSet implementedMethods;
  std::map<std::string, std::string, std::less<>> methodFunctionNames;
  for (auto &method : node->methods()) {
    auto *prototype = method->proto().get();
    auto methodName = std::string(prototype->id()->name());
    auto contract = contracts.find(methodName);
    if (contract == contracts.end() ||
        !declareName(implementedMethods, methodName)) {
      auto diagnostic =
          formatNameDiagnostic(diag::trait_method_signature, methodName);
      return emitError(prototype->id().get(), diagnostic);
    }

    auto fullName = traitMethodFunctionName(trait, targetType, methodName);
    prototype->id()->setName(fullName);
    prototype->setIsMethod(true);
    _functionPackages[fullName] = std::string(node->packageName());

    if (prototype->isGeneric()) {
      auto diagnostic =
          formatNameDiagnostic(diag::trait_method_signature, methodName);
      return emitError(prototype->id().get(), diagnostic);
    }
    if (sema(method.get()))
      return failure();

    auto *signature = lookupFunction(fullName);
    if (!signature) {
      auto diagnostic =
          formatNameDiagnostic(diag::trait_method_signature, methodName);
      return emitError(prototype->id().get(), diagnostic);
    }
    if (!traitMethodSignatureMatches(contract->second, signature, targetType)) {
      auto diagnostic =
          formatNameDiagnostic(diag::trait_method_signature, methodName);
      return emitError(prototype->id().get(), diagnostic);
    }

    methodFunctionNames.insert({methodName, fullName});
  }

  for (auto &contract : contracts) {
    if (implementedMethods.find(contract.first) != implementedMethods.end())
      continue;

    if (contract.second->hasDefaultBody()) {
      std::string functionName;
      if (instantiateTraitDefaultMethod(trait, contract.second, targetType,
                                        functionName))
        return failure();
      methodFunctionNames.insert({contract.first, std::move(functionName)});
      continue;
    }

    auto diagnostic =
        formatNameDiagnostic(diag::missing_trait_method, contract.first);
    return emitError(node, diagnostic);
  }

  if (_symbols.declareTraitImplementation(
          trait, targetType, node, std::move(methodFunctionNames))) {
    auto diagnostic = formatNameDiagnostic(diag::redefinition_trait_impl,
                                           trait->name());
    return emitError(node, diagnostic);
  }
  LLVM_DEBUG(llvm::dbgs() << "register trait implementation `"
                          << trait->name() << "` for `"
                          << formatType(targetType) << "`\n");
  return success();
}

auto SemaImpl::genericTraitImplementationMatches(const ImplDecl *impl,
                                                 const Type *type,
                                                 bool &matches)
    -> MulberryResult {
  auto &parameters = impl->comptimeParameters();
  auto argumentTypeNode = typeToTypeNode(type, impl->location());
  std::vector<TypeSubstitution> substitutions;
  substitutions.push_back(TypeSubstitution{parameters.front().name,
                                           argumentTypeNode.get(), std::nullopt});
  auto condition =
      substituteExpr(impl->whereCondition(), substitutions);

  PackageScope packageScope(_currentPackageName, impl->packageName());
  auto result = evaluateComptime(condition.get());
  if (result.kind == ComptimeEvaluation::Kind::Error)
    return failure();
  if (result.kind != ComptimeEvaluation::Kind::Value)
    return emitError(condition.get(), diag::expected_comptime_value);
  if (result.value->kind() != ComptimeValue::Kind::Bool)
    return emitError(condition.get(), diag::expected_bool);

  matches = result.value->boolValue();
  LLVM_DEBUG(llvm::dbgs() << "evaluate conditional trait implementation `"
                          << impl->trait()->name() << "` for `"
                          << formatType(type) << "`: "
                          << (matches ? "match" : "no match") << "\n");
  return success();
}

auto SemaImpl::findMatchingGenericTraitImplementations(
    const Type *type, const TraitDecl *trait,
    std::vector<const ImplDecl *> &implementations) -> MulberryResult {
  for (auto *impl : _symbols.genericTraitImplementations()) {
    if (impl->trait() != trait)
      continue;

    bool matches = false;
    if (genericTraitImplementationMatches(impl, type, matches))
      return failure();
    if (matches)
      implementations.push_back(impl);
  }
  return success();
}

auto SemaImpl::materializeGenericTraitImplementation(
    const Node *diagnosticNode, const Type *type, const TraitDecl *trait,
    bool &matched) -> MulberryResult {
  if (_symbols.lookupTraitImplementation(trait, type)) {
    matched = true;
    return success();
  }

  std::vector<const ImplDecl *> implementations;
  if (findMatchingGenericTraitImplementations(type, trait, implementations))
    return failure();
  if (implementations.empty()) {
    matched = false;
    return success();
  }
  if (implementations.size() != 1) {
    auto diagnostic =
        formatTypeTraitDiagnostic(diag::ambiguous_trait_impl, type, trait->name());
    return emitError(diagnosticNode, diagnostic);
  }

  auto *implementation = implementations.front();
  std::map<std::string, const TraitMethodDecl *, std::less<>> contracts;
  for (auto &method : trait->methods())
    contracts.insert({std::string(method->name()), method.get()});

  std::map<std::string, const FunctionDecl *, std::less<>>
      implementationMethods;
  for (auto &method : implementation->methods()) {
    implementationMethods.insert(
        {std::string(method->proto()->id()->name()), method.get()});
  }

  std::map<std::string, std::string, std::less<>> methodFunctionNames;
  for (auto &contract : contracts) {
    std::string functionName;
    auto method = implementationMethods.find(contract.first);
    if (method != implementationMethods.end()) {
      if (instantiateGenericTraitMethod(implementation, method->second,
                                        contract.second, type, functionName))
        return failure();
    } else {
      if (!contract.second->hasDefaultBody()) {
        auto diagnostic =
            formatNameDiagnostic(diag::missing_trait_method, contract.first);
        return emitError(diagnosticNode, diagnostic);
      }
      if (instantiateTraitDefaultMethod(trait, contract.second, type,
                                        functionName))
        return failure();
    }
    methodFunctionNames.insert({contract.first, std::move(functionName)});
  }

  if (_symbols.declareTraitImplementation(
          trait, type, implementation, std::move(methodFunctionNames))) {
    auto diagnostic = formatNameDiagnostic(diag::redefinition_trait_impl,
                                           trait->name());
    return emitError(diagnosticNode, diagnostic);
  }
  matched = true;
  LLVM_DEBUG(llvm::dbgs() << "materialize conditional trait implementation `"
                          << trait->name() << "` for `" << formatType(type)
                          << "`\n");
  return success();
}

auto SemaImpl::materializeGenericTraitMethod(const Node *diagnosticNode,
                                             const Type *type,
                                             std::string_view methodName,
                                             std::string &functionName)
    -> MulberryResult {
  std::vector<const TraitDecl *> traits;
  for (auto *impl : _symbols.genericTraitImplementations()) {
    auto *trait = impl->trait();
    bool declaresMethod = false;
    for (auto &method : trait->methods())
      if (method->name() == methodName)
        declaresMethod = true;
    if (!declaresMethod)
      continue;

    bool alreadySeen = false;
    for (auto *seenTrait : traits)
      if (seenTrait == trait)
        alreadySeen = true;
    if (!alreadySeen)
      traits.push_back(trait);
  }

  for (auto *trait : traits) {
    bool matched = false;
    if (materializeGenericTraitImplementation(diagnosticNode, type, trait,
                                              matched))
      return failure();
    if (!matched)
      continue;

    auto *implementation = _symbols.lookupTraitImplementation(trait, type);
    auto method = implementation->methodFunctionNames.find(methodName);
    if (method == implementation->methodFunctionNames.end())
      continue;
    functionName = method->second;
    return success();
  }
  return success();
}

auto SemaImpl::typeConformsToTrait(const Node *diagnosticNode,
                                   const Type *type, const TraitDecl *trait,
                                   bool &conforms) -> MulberryResult {
  return materializeGenericTraitImplementation(diagnosticNode, type, trait,
                                                conforms);
}

auto SemaImpl::checkTraitConstraints(
    const Node *node, const std::vector<ComptimeParam> &parameters,
    const std::vector<InferredComptimeArgument> &arguments)
    -> MulberryResult {
  for (size_t index = 0; index < parameters.size(); ++index) {
    auto &parameter = parameters[index];
    if (!parameter.trait)
      continue;
    auto *type = arguments[index].type;
    bool conforms = false;
    if (type && typeConformsToTrait(node, type, parameter.trait, conforms))
      return failure();
    if (conforms)
      continue;

    auto diagnostic = formatTypeTraitDiagnostic(
        diag::trait_constraint, type, parameter.trait->name());
    return emitError(node, diagnostic);
  }
  return success();
}

auto SemaImpl::sema(Expr *node) -> MulberryResult {
  switch (node->getKind()) {
  case Expr::Expr_Unit:
    return sema(cast<UnitExpr>(node));
  case Expr::Expr_Lambda:
    return sema(cast<LambdaExpr>(node));
  case Expr::Expr_Match:
    return sema(cast<MatchExpr>(node));
  case Expr::Expr_Try:
    return sema(cast<TryExpr>(node));
  case Expr::Expr_DataConstructor:
    return sema(cast<DataConstructorExpr>(node));
  case Expr::Expr_IntegerLiteral:
    return sema(cast<IntegerLiteralExpr>(node));
  case Expr::Expr_IntegerWiden:
    return sema(cast<IntegerWidenExpr>(node));
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
  if (auto *matchExpr = dyn_cast<MatchExpr>(node))
    return sema(matchExpr, type);

  if (auto *integerLiteral = dyn_cast<IntegerLiteralExpr>(node))
    return sema(integerLiteral, type);

  if (auto *binary = dyn_cast<BinaryExpr>(node))
    return sema(binary, type);

  if (auto *constructor = dyn_cast<DataConstructorExpr>(node)) {
    auto *dataType = getDataType(type);
    if (!dataType)
      return emitError(constructor, diag::mismatch_type);
    return sema(constructor, dataType);
  }

  if (auto *lambda = dyn_cast<LambdaExpr>(node)) {
    auto *functionType = getFunctionType(type);
    if (!functionType)
      return emitError(lambda, diag::mismatch_type);
    return sema(lambda, functionType);
  }

  auto *arrayLiteral = dyn_cast<ArrayLiteralExpr>(node);
  if (arrayLiteral) {
    // Source `[...]` defaults to Array. Explicit Array annotations keep
    // target-typed literal semantics.
    if (auto *arrayType = mulberry::getArrayType(type))
      return sema(arrayLiteral, arrayType);
  }

  auto *call = dyn_cast<CallExpr>(node);
  if (call && !call->hasReceiver()) {
    if (auto *callee = lookupVariable(call->name())) {
      auto *functionType = getFunctionType(callee->type);
      if (!functionType)
        return emitError(call, diag::mismatch_type);
      return semaIndirectCall(call, functionType, type);
    }

    auto callName = call->name();
    auto dot = callName.find('.');
    if (dot != std::string_view::npos &&
        lookupVariable(callName.substr(0, dot)))
      return semaDottedMethodCall(call, type);

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

auto SemaImpl::semaExpected(std::unique_ptr<Expr> &node, const Type *type)
    -> MulberryResult {
  if (sema(node.get(), type))
    return failure();

  if (sameType(node->type(), type))
    return success();

  if (!isIntegerWidening(node->type(), type))
    return emitError(node.get(), diag::mismatch_type);

  auto value = std::move(node);
  node = std::make_unique<IntegerWidenExpr>(value->location(),
                                             std::move(value));
  node->setType(type);
  LLVM_DEBUG(llvm::dbgs() << "widen fixed-width integer to Integer\n");
  return success();
}

auto SemaImpl::sema(DataConstructorExpr *node) -> MulberryResult {
  std::string resolvedName;
  auto *symbol = lookupDataConstructor(node->name(), resolvedName);
  if (!symbol)
    return emitError(node, diag::undefined_data_constructor);
  if (symbol->decl->isGeneric())
    return emitError(node, diag::mismatch_type);

  auto *dataType = getDataType(lookupType(symbol->decl->name()));
  if (!dataType)
    return emitError(node, diag::mismatch_type);
  return sema(node, dataType);
}

auto SemaImpl::sema(DataConstructorExpr *node,
                    const DataType *dataType) -> MulberryResult {
  std::string resolvedName;
  auto *symbol = lookupDataConstructor(node->name(), resolvedName);
  if (!symbol)
    return emitError(node, diag::undefined_data_constructor);
  if (symbol->decl->name() != dataType->declarationName())
    return emitError(node, diag::mismatch_type);

  auto &constructors = dataType->constructors();
  if (symbol->index >= constructors.size())
    return emitError(node, diag::mismatch_type);
  auto &constructor = constructors[symbol->index];
  auto &expressions = node->expressions();
  if (expressions.size() != constructor.payloadTypes().size())
    return emitError(node, diag::wrong_num_arg);

  for (size_t i = 0; i < expressions.size(); ++i) {
    auto *payloadType = constructor.payloadTypes()[i];
    if (semaExpected(expressions[i], payloadType))
      return failure();
  }

  node->setName(resolvedName);
  node->setConstructorIndex(symbol->index);
  node->setType(dataType);
  LLVM_DEBUG(llvm::dbgs() << "bind data constructor `" << resolvedName
                          << "` as " << formatType(dataType) << " tag "
                          << symbol->index << "\n");
  return success();
}

auto SemaImpl::semaGenericCall(CallExpr *node,
                               const GenericFunctionSymbol *symbol,
                               const Type *expectedType)
    -> MulberryResult {
  auto *genericFunction = symbol->decl;
  auto *genericProto = genericFunction->proto().get();
  auto name = genericProto->id()->name();
  auto callerPackageName = _currentPackageName;
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

  auto semaArgument = [&](std::unique_ptr<Expr> &argument,
                          const TypeNode *parameterTypeNode)
      -> MulberryResult {
    auto *literal = dyn_cast<ArrayLiteralExpr>(argument.get());
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
      return sema(argument.get());

    auto *parameterType = resolveSubstitutedType(
        parameterTypeNode,
        comptimeSubstitutions(comptimeParameters, inferredArguments));
    if (!parameterType)
      return failure();
    return semaExpected(argument, parameterType);
  };

  // Sibling arguments often determine a lambda's generic parameter types.
  // Analyze ordinary arguments first, then infer the callback result from its
  // body and feed that type back into generic matching.
  std::vector<size_t> deferredLambdas;
  std::vector<size_t> deferredParameters;
  for (size_t i = 0; i < expressions.size(); ++i) {
    auto *parameterTypeNode = parameters[i]->typeNode();
    if (dyn_cast<LambdaExpr>(expressions[i].get())) {
      deferredLambdas.push_back(i);
      LLVM_DEBUG(llvm::dbgs()
                 << "defer lambda parameter " << i << " of `" << name
                 << "`\n");
      continue;
    }

    if (hasComputedType(parameterTypeNode)) {
      deferredParameters.push_back(i);
      LLVM_DEBUG(llvm::dbgs()
                 << "defer computed parameter " << i << " of `" << name
                 << "`\n");
      continue;
    }

    if (semaArgument(expressions[i], parameterTypeNode))
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

  for (auto index : deferredLambdas) {
    auto *functionPattern =
        dyn_cast<FunctionTypeNode>(parameters[index]->typeNode());
    if (!functionPattern)
      return emitError(expressions[index].get(), diag::mismatch_type);

    std::vector<ComptimeParam> unresolvedParameters;
    for (size_t i = 0; i < comptimeParameters.size(); ++i)
      if (!inferredArguments[i].isResolved())
        unresolvedParameters.push_back(comptimeParameters[i]);

    auto substitutions =
        comptimeSubstitutions(comptimeParameters, inferredArguments);
    std::vector<const Type *> lambdaParameterTypes;
    for (auto &parameterTypeNode : functionPattern->parameterTypes()) {
      if (containsComptimeParameter(parameterTypeNode.get(),
                                    unresolvedParameters))
        return emitError(expressions[index].get(), diag::mismatch_type);

      auto *parameterType =
          resolveSubstitutedType(parameterTypeNode.get(), substitutions);
      if (!parameterType)
        return failure();
      lambdaParameterTypes.push_back(parameterType);
    }

    auto *lambda = cast<LambdaExpr>(expressions[index].get());
    if (semaLambda(lambda, lambdaParameterTypes,
                   functionPattern->parameterCanMutateObject(),
                   /*expectedReturnType=*/nullptr, callerPackageName))
      return failure();
    if (!matchGenericType(functionPattern, lambda->type(),
                          comptimeParameters, inferredArguments))
      return emitError(lambda, diag::mismatch_type);
  }

  for (auto &argument : inferredArguments)
    if (!argument.isResolved())
      return emitError(node, diag::mismatch_type);

  if (checkTraitConstraints(node, comptimeParameters, inferredArguments))
    return failure();

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
    } else if (semaExpected(argument, parameterType)) {
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
  if (expectedType &&
      !sameType(expectedType, signature->type->returnType()))
    return emitError(node, diag::mismatch_type);

  for (size_t i = 0; i < expressions.size(); ++i) {
    auto &arg = expressions[i];
    auto *parameterType = signature->type->parameterTypes()[i];
    if (!sameCallArgumentType(parameterType, arg->type(),
                              node->isLoweredMethodCall() && i == 0))
      return emitError(arg.get(), diag::mismatch_type);
    if (checkMutableObjectArgument(signature->type, i, arg.get()))
      return failure();
  }

  node->setName(concreteName);
  node->setType(signature->type->returnType());
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

auto SemaImpl::sema(LambdaExpr *node) -> MulberryResult {
  return emitError(node, diag::lambda_expected_function_type);
}

auto SemaImpl::sema(MatchExpr *node, const Type *expectedType)
    -> MulberryResult {
  auto *value = node->value().get();
  if (sema(value))
    return failure();

  auto *dataType = getDataType(value->type());
  if (!dataType)
    return emitError(value, diag::match_expected_data_type);

  std::vector<bool> covered(dataType->constructors().size(), false);
  auto *resultType = expectedType;
  auto canMutateObject = true;
  for (auto &arm : node->arms()) {
    auto *pattern = arm->pattern().get();
    const DataConstructor *constructor = nullptr;
    if (checkMatchPattern(pattern, dataType, covered, constructor))
      return failure();
    if (containsReturnStat(arm->bodyBlock().get()))
      return emitError(arm.get(), diag::match_expression_arm_return);

    VariableScope patternScope(_symbols);
    if (declareMatchPatternBindings(pattern, constructor))
      return failure();

    VariableScope bodyScope(_symbols);
    IsolatedWhileScope isolatedWhileScope(_whileDepth);
    for (auto &statement : arm->bodyBlock()->statements())
      if (sema(statement.get()))
        return failure();
    arm->bodyBlock()->setType(
        _typeContext.getBuiltinType(BuiltinTypeKind::Unit));

    auto &armResult = arm->resultExpr();
    if (resultType) {
      if (semaExpected(armResult, resultType))
        return failure();
    } else {
      if (sema(armResult.get()))
        return failure();
      resultType = armResult->type();
    }
    if (!resultType || !sameType(resultType, armResult->type()))
      return emitError(armResult.get(), diag::mismatch_type);
    if (isMutableSourceObjectType(resultType) &&
        !canMutateObjectReference(armResult.get()))
      canMutateObject = false;
  }

  if (checkExhaustiveMatch(node, covered))
    return failure();
  if (!resultType)
    return emitError(node, diag::mismatch_type);
  node->setType(resultType);
  node->setCanMutateObject(canMutateObject);
  return success();
}

auto SemaImpl::sema(TryExpr *node) -> MulberryResult {
  if (!_currentFunctionReturnType)
    return emitError(node, diag::try_outside_result_function);

  auto *value = node->value().get();
  if (sema(value))
    return failure();

  auto operandTypes = getResultTypeArguments(value->type());
  if (!operandTypes)
    return emitError(value, diag::try_expected_result);

  auto functionTypes = getResultTypeArguments(_currentFunctionReturnType);
  if (!functionTypes)
    return emitError(node, diag::try_outside_result_function);
  if (!sameType(operandTypes->errorType, functionTypes->errorType))
    return emitError(node, diag::try_error_type_mismatch);

  node->setType(operandTypes->valueType);
  node->setCanMutateObject(canMutateObjectReference(value));
  LLVM_DEBUG(llvm::dbgs() << "propagate `" << formatType(value->type())
                          << "` error from `?`\n");
  return success();
}

auto SemaImpl::sema(LambdaExpr *node, const FunctionType *functionType)
    -> MulberryResult {
  auto packageName = _currentPackageName;
  return semaLambda(node, functionType->parameterTypes(),
                    functionType->parameterCanMutateObject(),
                    functionType->returnType(), packageName);
}

auto SemaImpl::semaLambda(
    LambdaExpr *node, const std::vector<const Type *> &parameterTypes,
    const std::vector<bool> &parameterCanMutateObject,
    const Type *expectedReturnType, std::string_view packageName)
    -> MulberryResult {
  auto &parameters = node->parameters();
  if (parameters.size() != parameterTypes.size())
    return emitError(node, diag::wrong_num_arg);

  PackageScope packageScope(_currentPackageName, packageName);
  VariableScope variableScope(_symbols);
  NoncapturingLambdaScope lambdaScope(_noncapturingLambdaDepth);
  for (size_t i = 0; i < parameters.size(); ++i) {
    auto &parameter = parameters[i];
    parameter.type = parameterTypes[i];
    auto canMutateObject = parameterCanMutateObject[i];
    if (declareVariable(parameter.name, parameter.type,
                        /*isConstBinding=*/!canMutateObject,
                        canMutateObject))
      return emitError(parameter.location, diag::redefinition_var);
  }

  auto &body = node->body();
  FunctionReturnTypeScope returnTypeScope(_currentFunctionReturnType,
                                          expectedReturnType);
  if (expectedReturnType ? semaExpected(body, expectedReturnType)
                         : sema(body.get()))
    return failure();
  auto *returnType = body->type();
  if (!returnType)
    return emitError(body.get(), diag::mismatch_type);
  if (expectedReturnType && !sameReturnType(expectedReturnType, returnType))
    return emitError(body.get(), diag::wrong_return_type);

  auto *functionType = _typeContext.createFunctionType(
      std::vector<const Type *>(parameterTypes.begin(), parameterTypes.end()),
      parameterCanMutateObject, returnType);
  node->setType(functionType);

  // A noncapturing lambda is a normal private function plus a function value;
  // no closure environment or lambda-specific lowering is needed.
  std::string functionName =
      "__mulberry_lambda_" + std::to_string(_lambdaCounter++);
  node->setFunctionName(functionName);

  VectorUniquePtr<ParameterDecl> parameterDecls;
  for (size_t i = 0; i < parameters.size(); ++i) {
    auto &parameter = parameters[i];
    auto variable = std::make_unique<VariableExpr>(parameter.location,
                                                   parameter.name);
    variable->setType(parameter.type);
    auto parameterDecl = std::make_unique<ParameterDecl>(
        parameter.location, std::move(variable),
        typeToTypeNode(parameter.type, parameter.location),
        parameterCanMutateObject[i]);
    parameterDecl->setType(parameter.type);
    parameterDecls.push_back(std::move(parameterDecl));
  }

  auto functionNameNode =
      std::make_unique<FunctionName>(node->location(), functionName);
  auto prototype = std::make_unique<Prototype>(
      node->location(), std::move(functionNameNode),
      std::move(parameterDecls), typeToTypeNode(returnType, node->location()));
  prototype->setType(returnType);

  VectorUniquePtr<Stat> statements;
  auto bodyExpression = node->takeBody();
  if (isUnitType(returnType)) {
    statements.push_back(std::make_unique<ExprStat>(
        bodyExpression->location(), std::move(bodyExpression)));
  } else {
    statements.push_back(std::make_unique<ReturnStat>(
        bodyExpression->location(), std::move(bodyExpression)));
  }
  auto functionBody =
      std::make_unique<BlockExpr>(node->location(), std::move(statements));
  functionBody->setType(_typeContext.getBuiltinType(BuiltinTypeKind::Unit));

  if (declareFunction(functionName, functionType, /*isExtern=*/false,
                      packageName)) {
    auto diagnostic =
        formatNameDiagnostic(diag::redefinition_func, functionName);
    return emitError(node, diagnostic);
  }

  LLVM_DEBUG(llvm::dbgs() << "lift lambda to `" << functionName << "` as `"
                          << formatType(functionType) << "`\n");
  _lambdaFunctions.push_back(std::make_unique<FunctionDecl>(
      node->location(), std::move(prototype), std::move(functionBody)));
  return success();
}

auto SemaImpl::sema(CallExpr *node) -> MulberryResult {
  if (node->hasReceiver())
    return semaMethodCall(node);

  if (auto *callee = lookupVariable(node->name())) {
    auto *functionType = getFunctionType(callee->type);
    if (!functionType)
      return emitError(node, diag::mismatch_type);
    return semaIndirectCall(node, functionType);
  }

  auto callName = node->name();
  auto dot = callName.find('.');
  if (dot != std::string_view::npos &&
      lookupVariable(callName.substr(0, dot)))
    return semaDottedMethodCall(node);

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
  if (expressions.size() != signature->type->parameterTypes().size()) {
    auto diagnostic = formatNameSizeDiagnostic(
        diag::func_param, name, signature->type->parameterTypes().size());
    return emitError(node, diagnostic);
  }

  for (size_t i = 0; i < expressions.size(); ++i) {
    auto &arg = expressions[i];
    auto *parameterType = signature->type->parameterTypes()[i];
    if (node->isLoweredMethodCall() && i == 0) {
      if (sema(arg.get()))
        return failure();
    } else if (semaExpected(arg, parameterType)) {
      return failure();
    }
    if (!sameCallArgumentType(parameterType, arg->type(),
                              node->isLoweredMethodCall() && i == 0))
      return emitError(arg.get(), diag::mismatch_type);
    if (checkMutableObjectArgument(signature->type, i, arg.get()))
      return failure();
  }

  auto resolvedName = resolveFunctionName(name);
  if (!resolvedName.empty())
    node->setName(resolvedName);

  node->setType(signature->type->returnType());
  return success();
}

auto SemaImpl::semaIndirectCall(CallExpr *node,
                                const FunctionType *functionType,
                                const Type *expectedType) -> MulberryResult {
  auto &expressions = node->expressions();
  auto &parameterTypes = functionType->parameterTypes();
  if (expressions.size() != parameterTypes.size()) {
    auto diagnostic = formatNameSizeDiagnostic(
        diag::func_param, node->name(), parameterTypes.size());
    return emitError(node, diagnostic);
  }

  for (size_t i = 0; i < expressions.size(); ++i) {
    auto &argument = expressions[i];
    if (semaExpected(argument, parameterTypes[i]))
      return failure();
    if (checkMutableObjectArgument(functionType, i, argument.get()))
      return failure();
  }

  auto *returnType = functionType->returnType();
  if (expectedType && !sameType(expectedType, returnType))
    return emitError(node, diag::mismatch_type);
  node->setIndirectCall();
  node->setType(returnType);
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
  auto methodName = std::string(node->name());
  if (structType) {
    std::vector<std::string> owners;
    if (auto *origin = structType->origin())
      owners.push_back(std::string(origin->aliasName()));
    owners.push_back(std::string(structType->name()));

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
  }

  auto *traitReceiverType = ptrType ? ptrType->pointeeType() : receiverType;
  if (auto *functionName =
          _symbols.lookupTraitMethod(traitReceiverType, methodName)) {
    LLVM_DEBUG(llvm::dbgs() << "resolve trait method `" << methodName
                            << "` for `" << formatType(traitReceiverType)
                            << "` to `" << *functionName << "`\n");
    node->lowerMethodCall(*functionName);
    return sema(node);
  }

  std::string functionName;
  if (materializeGenericTraitMethod(node, traitReceiverType, methodName,
                                    functionName))
    return failure();
  if (!functionName.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "resolve conditional trait method `"
                            << methodName << "` for `"
                            << formatType(traitReceiverType) << "` to `"
                            << functionName << "`\n");
    node->lowerMethodCall(functionName);
    return sema(node);
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
    if (semaExpected(expr, field.type()))
      return failure();
  }

  node->setType(structType);
  return success();
}

auto SemaImpl::sema(VariableExpr *node) -> MulberryResult {
  auto *symbol = lookupVariable(node->name());
  if (symbol && _noncapturingLambdaDepth > 0 &&
      !_symbols.lookupCurrentVariable(node->name()))
    return emitError(node, diag::lambda_capture);

  if (!symbol) {
    auto functionName = resolveFunctionName(node->name());
    if (functionName.empty())
      return emitError(node, diag::undefined_var);

    auto *function = _symbols.lookupFunction(functionName);
    if (!function)
      return emitError(node, diag::undefined_var);
    if (function->isExtern)
      return emitError(node, diag::extern_function_value);

    node->setName(functionName);
    node->setFunctionValue();
    node->setType(function->type);
    return success();
  }

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

auto SemaImpl::sema(IntegerLiteralExpr *node) -> MulberryResult {
  if (!node->hasValidSpelling())
    return emitError(node, diag::invalid_integer_literal);
  if (!node->getUInt64Value())
    return emitError(node, diag::integer_literal_overflows);
  setBuiltinType(node, BuiltinTypeKind::UInt64);
  LLVM_DEBUG(llvm::dbgs() << "type integer literal `" << node->spelling()
                          << "` as UInt64\n");
  return success();
}

auto SemaImpl::sema(IntegerLiteralExpr *node, const Type *type)
    -> MulberryResult {
  if (!node->hasValidSpelling())
    return emitError(node, diag::invalid_integer_literal);

  if (isIntegerType(type)) {
    node->setType(type);
    LLVM_DEBUG(llvm::dbgs() << "type integer literal `" << node->spelling()
                            << "` as Integer\n");
    return success();
  }

  auto value = node->getUInt64Value();
  if (!value)
    return emitError(node, diag::integer_literal_overflows);

  if (isUInt8Type(type)) {
    if (*value > 255)
      return emitError(node, diag::mismatch_type);
    node->setType(type);
    return success();
  }

  if (isUInt64Type(type)) {
    node->setType(type);
    return success();
  }

  return sema(node);
}

auto SemaImpl::sema(IntegerWidenExpr *node) -> MulberryResult {
  if (sema(node->value().get()))
    return failure();
  if (!isIntegerWidening(
          node->value()->type(),
          _typeContext.getBuiltinType(BuiltinTypeKind::Integer)))
    return emitError(node, diag::mismatch_type);
  setBuiltinType(node, BuiltinTypeKind::Integer);
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
  if (!isSourceObjectType(valueType))
    return emitError(node->value().get(), diag::mismatch_type);

  auto *stringType = lookupType("String");
  if (!stringType)
    return emitError(node, diag::undefined_type);

  constexpr std::string_view functionName =
      "mulberry_string_object_identity";
  auto resolvedName = resolveFunctionName(functionName);
  auto *signature = lookupFunction(resolvedName);
  if (!signature || signature->type->parameterTypes().size() != 2 ||
      !sameType(signature->type->parameterTypes()[0], stringType) ||
      !sameType(signature->type->returnType(), stringType)) {
    auto diagnostic = formatNameDiagnostic(diag::undefined_func, functionName);
    return emitError(node, diagnostic);
  }

  auto *objectPtr = getPtrType(signature->type->parameterTypes()[1]);
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
      semaExpected(node->rhs(), node->lhs()->type()))
    return failure();
  if (!node->lhs()->isLvalue())
    return emitError(node->lhs().get(), diag::expected_lvalue);
  if (checkAssignable(node->lhs().get()))
    return failure();
  setBuiltinType(node, BuiltinTypeKind::Unit);
  return success();
}

auto SemaImpl::sema(BinaryExpr *node) -> MulberryResult {
  return sema(node, nullptr);
}

auto SemaImpl::sema(BinaryExpr *node, const Type *expectedType)
    -> MulberryResult {
  using Operator = BinaryExpr::Operator;
  auto &lhs = node->lhs();
  auto &rhs = node->rhs();

  if (node->opEnum() == Operator::ShiftLeft ||
      node->opEnum() == Operator::ShiftRight) {
    if (expectedType && isIntegerType(expectedType)) {
      if (semaExpected(lhs, expectedType))
        return failure();
    } else if (sema(lhs.get())) {
      return failure();
    }

    auto *countType = _typeContext.getBuiltinType(BuiltinTypeKind::UInt64);
    if (semaExpected(rhs, countType))
      return failure();
    if (!isIntegerType(lhs->type()) || !isUInt64Type(rhs->type()))
      return emitError(lhs.get(), diag::mismatch_type);
    if (expectedType && !sameType(lhs->type(), expectedType))
      return emitError(lhs.get(), diag::mismatch_type);

    node->setType(lhs->type());
    return success();
  }

  if (expectedType && isIntegerType(expectedType)) {
    if (semaExpected(lhs, expectedType) || semaExpected(rhs, expectedType))
      return failure();
  } else {
    if (sema(lhs.get()))
      return failure();

    auto *stringType = lookupType("String");
    if (node->opEnum() == Operator::Add && stringType &&
        sameType(lhs->type(), stringType)) {
      if (sema(rhs.get()))
        return failure();
      if (semaFormatValueCall(rhs, stringType))
        return failure();
      if (checkStringConcatFunction(node, stringType))
        return failure();
      node->setType(stringType);
      return success();
    }

    if (isIntegerType(lhs->type())) {
      if (semaExpected(rhs, lhs->type()))
        return failure();
    } else {
      if (sema(rhs.get()))
        return failure();
      if (isIntegerType(rhs->type()) && semaExpected(lhs, rhs->type()))
        return failure();
    }
  }

  auto *lhsType = lhs->type();
  auto *rhsType = rhs->type();
  if (!sameType(lhsType, rhsType))
    return emitError(lhs.get(), diag::mismatch_type);

  switch (node->opEnum()) {
  case Operator::ShiftLeft:
  case Operator::ShiftRight:
    llvm_unreachable("shift operations are handled before type comparison");
  case Operator::Add:
  case Operator::Mul:
  case Operator::Diff: {
    if (!isNumericType(lhsType))
      return emitError(lhs.get(), diag::mismatch_type);
    node->setType(lhsType);
    return success();
  }
  case Operator::Div: {
    // Integer division is fallible and will be exposed as bigint.div(), not
    // as the fixed-width `/` operator.
    if (isIntegerType(lhsType) || !isNumericType(lhsType))
      return emitError(lhs.get(), diag::mismatch_type);
    node->setType(lhsType);
    return success();
  }
  case Operator::Rem: {
    if (!isUInt64Type(lhsType))
      return emitError(lhs.get(), diag::mismatch_type);
    setBuiltinType(node, BuiltinTypeKind::UInt64);
    return success();
  }
  case Operator::BitAnd:
  case Operator::BitOr:
  case Operator::BitXor: {
    if (!isIntegerType(lhsType))
      return emitError(lhs.get(), diag::mismatch_type);
    node->setType(lhsType);
    return success();
  }
  case Operator::And:
  case Operator::Or: {
    if (!isBoolType(lhsType))
      return emitError(lhs.get(), diag::mismatch_type);
    setBuiltinType(node, BuiltinTypeKind::Bool);
    return success();
  }
  case Operator::EQ:
  case Operator::NEQ: {
    if (!isEquatableType(lhsType))
      return emitError(lhs.get(), diag::mismatch_type);
    setBuiltinType(node, BuiltinTypeKind::Bool);
    return success();
  }
  case Operator::LT:
  case Operator::LE:
  case Operator::GT:
  case Operator::GE: {
    if (!isNumericType(lhsType))
      return emitError(lhs.get(), diag::mismatch_type);
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
  if (signature && signature->type->parameterTypes().size() == 2 &&
      sameType(signature->type->parameterTypes()[0], stringType) &&
      sameType(signature->type->parameterTypes()[1], stringType) &&
      sameType(signature->type->returnType(), stringType))
    return success();

  auto diagnostic =
      formatNameDiagnostic(diag::undefined_func, "std.string.concat");
  return emitError(node, diagnostic);
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

  if (auto *value = dyn_cast<IntegerLiteralExpr>(node)) {
    if (!value->hasValidSpelling()) {
      emitError(value, diag::invalid_integer_literal);
      return {ComptimeEvaluation::Kind::Error, std::nullopt};
    }
    if (isIntegerType(value->type()))
      return {ComptimeEvaluation::Kind::Runtime, std::nullopt};
    auto uint64Value = value->getUInt64Value();
    if (!uint64Value) {
      emitError(value, diag::integer_literal_overflows);
      return {ComptimeEvaluation::Kind::Error, std::nullopt};
    }
    setBuiltinType(value, BuiltinTypeKind::UInt64);
    return {ComptimeEvaluation::Kind::Value,
            ComptimeValue(*uint64Value)};
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
      name == "isInteger" || name == "isFloat32" || name == "isArray" ||
      name == "isStruct" || name == "isObject") {
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
    if (name == "isInteger")
      return {ComptimeEvaluation::Kind::Value,
              ComptimeValue(isIntegerType(type)), true};
    if (name == "isFloat32")
      return {ComptimeEvaluation::Kind::Value,
              ComptimeValue(isFloat32Type(type)), true};
    if (name == "isArray")
      return {ComptimeEvaluation::Kind::Value,
              ComptimeValue(isArrayType(type)), true};
    if (name == "isObject")
      return {ComptimeEvaluation::Kind::Value,
              ComptimeValue(isSourceObjectType(type)), true};
    return {ComptimeEvaluation::Kind::Value,
            ComptimeValue(getStructType(type) != nullptr), true};
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
  case Operator::ShiftLeft:
  case Operator::ShiftRight:
  case Operator::BitAnd:
  case Operator::BitOr:
  case Operator::BitXor:
    return cannotEvaluate();
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
  if (auto *matchExpr = llvm::dyn_cast<MatchExpr>(expr))
    return matchExpr->canMutateObject();

  if (auto *tryExpr = llvm::dyn_cast<TryExpr>(expr))
    return tryExpr->canMutateObject();

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
  if (auto *matchExpr = llvm::dyn_cast<MatchExpr>(expr)) {
    if (!matchExpr->canMutateObject())
      return emitError(matchExpr, diag::readonly_to_mutable_reference);
    return success();
  }

  if (auto *tryExpr = llvm::dyn_cast<TryExpr>(expr)) {
    if (!tryExpr->canMutateObject())
      return emitError(expr, diag::readonly_to_mutable_reference);
    return success();
  }

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

auto SemaImpl::checkMutableObjectArgument(const FunctionType *functionType,
                                          size_t index, const Expr *arg)
    -> MulberryResult {
  if (!functionType->parameterCanMutateObject()[index])
    return success();
  if (!isMutableSourceObjectType(functionType->parameterTypes()[index]))
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
    if (semaArrayLiteralElement(element, type->elementType()))
      return failure();
  }

  expr->setType(type);
  return success();
}

auto SemaImpl::semaArrayLiteralElement(std::unique_ptr<Expr> &expr,
                                       const Type *type)
    -> MulberryResult {
  if (auto *arrayLiteral = llvm::dyn_cast<ArrayLiteralExpr>(expr.get())) {
    auto *arrayType = mulberry::getArrayType(type);
    if (!arrayType)
      return emitError(expr.get(), diag::mismatch_type);
    return sema(arrayLiteral, arrayType);
  }

  if (mulberry::getArrayType(type))
    return emitError(expr.get(), diag::mismatch_type);

  return semaExpected(expr, type);
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

auto SemaImpl::checkMatchPattern(
    DataPattern *pattern, const DataType *dataType,
    std::vector<bool> &covered, const DataConstructor *&constructor)
    -> MulberryResult {
  auto &constructors = dataType->constructors();
  std::string resolvedName;
  auto *symbol = lookupDataConstructor(pattern->constructorName(),
                                       resolvedName);
  if (!symbol)
    return emitError(pattern, diag::undefined_data_constructor);
  if (symbol->decl->name() != dataType->declarationName() ||
      symbol->index >= constructors.size())
    return emitError(pattern, diag::match_constructor_type);
  if (covered[symbol->index])
    return emitError(pattern, diag::duplicate_match_constructor);

  constructor = &constructors[symbol->index];
  if (pattern->bindings().size() != constructor->payloadTypes().size())
    return emitError(pattern, diag::match_binding_count);

  pattern->setConstructorName(resolvedName);
  pattern->setConstructorIndex(symbol->index);
  covered[symbol->index] = true;
  for (size_t i = 0; i < pattern->bindings().size(); ++i)
    pattern->bindings()[i]->setType(constructor->payloadTypes()[i]);
  LLVM_DEBUG(llvm::dbgs() << "bind match pattern `" << resolvedName
                          << "` as " << formatType(dataType) << " tag "
                          << symbol->index << "\n");
  return success();
}

auto SemaImpl::declareMatchPatternBindings(
    DataPattern *pattern, const DataConstructor *constructor)
    -> MulberryResult {
  for (size_t i = 0; i < pattern->bindings().size(); ++i) {
    auto *binding = pattern->bindings()[i].get();
    auto *payloadType = constructor->payloadTypes()[i];
    if (declareVariable(binding->name(), payloadType,
                        /*isConstBinding=*/true,
                        /*canMutateObject=*/false))
      return emitError(binding, diag::redefinition_var);
  }
  return success();
}

auto SemaImpl::checkExhaustiveMatch(
    const Node *node, const std::vector<bool> &covered) -> MulberryResult {
  for (auto isCovered : covered)
    if (!isCovered)
      return emitError(node, diag::non_exhaustive_match);
  return success();
}

auto SemaImpl::sema(MatchStat *node) -> MulberryResult {
  auto *value = node->value().get();
  if (sema(value))
    return failure();

  auto *dataType = getDataType(value->type());
  if (!dataType)
    return emitError(value, diag::match_expected_data_type);

  std::vector<bool> covered(dataType->constructors().size(), false);
  for (auto &arm : node->arms()) {
    auto *pattern = arm->pattern().get();
    const DataConstructor *constructor = nullptr;
    if (checkMatchPattern(pattern, dataType, covered, constructor))
      return failure();

    VariableScope armScope(_symbols);
    if (declareMatchPatternBindings(pattern, constructor) ||
        sema(arm->bodyBlock().get()))
      return failure();
  }

  return checkExhaustiveMatch(node, covered);
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
  case Stat::Stat_Match:
    return sema(cast<MatchStat>(node));
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
  auto &initExpr = node->init();

  const Type *declaredType = nullptr;
  if (node->hasExplicitType()) {
    declaredType = checkType(node->typeNode(), UnitPolicy::Allow);
    if (!declaredType)
      return failure();
  }

  std::optional<ComptimeValue> comptimeValue;
  auto initializerWasTyped = false;
  if (node->isConstBinding() && declaredType && isIntegerType(declaredType)) {
    if (semaExpected(initExpr, declaredType))
      return failure();
    initializerWasTyped = true;
  }

  if (node->isConstBinding()) {
    auto evaluation = evaluateComptime(initExpr.get());
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
        if (!valueType || !sameType(declaredType, valueType))
          return emitError(initExpr.get(), diag::mismatch_type);
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
  if (declaredType) {
    varType = declaredType;
    if (!initializerWasTyped && semaExpected(initExpr, varType))
      return failure();
  } else {
    if (sema(initExpr.get()))
      return failure();
    varType = initExpr->type();
    if (!varType)
      return emitError(initExpr.get(), diag::mismatch_type);
  }
  node->setType(varType);

  auto canMutateObject = node->canMutateObject();
  if (canMutateObject && isMutableSourceObjectType(varType) &&
      !canMutateObjectReference(initExpr.get()))
    return emitError(initExpr.get(), diag::readonly_to_mutable_binding);
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

  auto &expression = node->expression();
  if (getPtrType(_currentFunctionReturnType)) {
    // The internal Ptr<T> -> Ptr<U> cast is valid only at a declared return
    // boundary. Keep ordinary expected-value contexts type-safe.
    if (sema(expression.get()))
      return failure();
  } else if (semaExpected(expression, _currentFunctionReturnType)) {
    return failure();
  }
  if (!sameReturnType(_currentFunctionReturnType, expression->type()))
    return emitError(expression.get(), diag::wrong_return_type);
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
