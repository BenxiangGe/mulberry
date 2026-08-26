//===--- SemaSupport.cpp - Shared semantic-analysis helpers ----------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "SemaSupport.h"
#include "llvm/Support/Casting.h"

namespace mulberry {
using llvm::cast;
using llvm::dyn_cast;

auto substituteExpr(const Expr *node,
                    const std::vector<TypeSubstitution> &substitutions)
    -> std::unique_ptr<Expr>;

auto substituteBlockExpr(const BlockExpr *node,
                         const std::vector<TypeSubstitution> &substitutions)
    -> std::unique_ptr<BlockExpr>;

auto substituteComptimeBlockExpr(
    const ComptimeBlockExpr *node,
    const std::vector<TypeSubstitution> &substitutions)
    -> std::unique_ptr<ComptimeBlockExpr>;

auto substituteStat(const Stat *node,
                    const std::vector<TypeSubstitution> &substitutions)
    -> std::unique_ptr<Stat>;

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

auto formatMissingFromDiagnostic(const Type *sourceType, const Type *targetType)
    -> std::string {
  auto sourceName = formatStringificationType(sourceType);
  auto targetName = formatStringificationType(targetType);
  return "`?` cannot convert error type `" + sourceName + "` to `" + targetName +
         "`. Add a conversion like:\n"
         "  impl From<" + sourceName + "> for " + targetName + " {\n"
         "    fn from(value: " + sourceName + "): Self {\n"
         "      return /* construct " + targetName + " from value */;\n"
         "    }\n"
         "  }";
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
  return isIntegerType(targetType) && isFixedWidthIntegerType(sourceType);
}

auto mangleTypeName(std::string name) -> std::string {
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
                     const std::vector<ComptimeArgument> &arguments)
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

auto sameReturnType(const Type *returnType, const Type *actualType) -> bool {
  if (sameType(returnType, actualType))
    return true;

  // Pointer reinterpretation stays explicit in stdlib/internal source
  // through helpers such as std.internal.ptr.asUInt8<T>(). The helper's body
  // returns Ptr<T>; MLIRGen materializes the declared Ptr<UInt8> return with
  // mulberry_core.ptr.cast.
  return getPtrType(returnType) && getPtrType(actualType);
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

  if (auto *selfType = dyn_cast<SelfTypeNode>(node))
    return std::make_unique<SelfTypeNode>(selfType->location());

  if (auto *namedType = dyn_cast<NamedTypeNode>(node)) {
    auto result = std::make_unique<NamedTypeNode>(namedType->location(),
                                                  namedType->name());
    result->setResolvedType(namedType->resolvedType());
    return result;
  }

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
          cloneTypeNode(parameter->typeNode()), parameter->canMutateObject(),
          parameter->isComptime(), parameter->isPack()));
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
  if (auto *builtinType = getBuiltinType(type)) {
    auto result = std::make_unique<NamedTypeNode>(location, builtinType->name());
    result->setResolvedType(type);
    return result;
  }

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

    auto result = std::make_unique<NamedTypeNode>(location, structType->name());
    result->setResolvedType(type);
    return result;
  }

  if (auto *dataType = getDataType(type)) {
    if (dataType->arguments().empty()) {
      auto result = std::make_unique<NamedTypeNode>(
          location, dataType->declarationName());
      result->setResolvedType(type);
      return result;
    }

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

  if (auto *selfType = dyn_cast<SelfTypeNode>(node)) {
    if (substitution.parameterName == "Self" &&
        substitution.argumentTypeNode)
      return cloneTypeNode(substitution.argumentTypeNode);
    return cloneTypeNode(selfType);
  }

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
          parameter->canMutateObject(), parameter->isComptime(),
          parameter->isPack()));
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

auto containsSelfType(const TypeNode *node) -> bool {
  if (dyn_cast<SelfTypeNode>(node))
    return true;

  if (llvm::isa<UnitTypeNode, NamedTypeNode>(node))
    return false;

  if (llvm::isa<ComputedTypeNode>(node))
    return false;

  if (auto *arrayType = dyn_cast<ArrayTypeNode>(node))
    return containsSelfType(arrayType->elementTypeNode());

  if (auto *ptrType = dyn_cast<PtrTypeNode>(node))
    return containsSelfType(ptrType->pointeeTypeNode());

  if (auto *functionType = dyn_cast<FunctionTypeNode>(node)) {
    for (auto &parameterType : functionType->parameterTypes())
      if (containsSelfType(parameterType.get()))
        return true;
    return containsSelfType(functionType->returnTypeNode());
  }

  if (auto *genericType = dyn_cast<GenericTypeNode>(node)) {
    for (auto &argument : genericType->arguments())
      if (argument.kind() == ComptimeArg::Kind::Type &&
          containsSelfType(argument.typeNode()))
        return true;
    return false;
  }

  auto *structType = cast<StructTypeNode>(node);
  for (auto &field : structType->fields())
    if (containsSelfType(field->typeNode()))
      return true;
  return false;
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
  for (auto &statement : node->statements())
    statements.push_back(substituteStat(statement.get(), substitutions));

  return std::make_unique<BlockExpr>(node->location(), std::move(statements));
}

auto substituteStat(const Stat *node,
                    const std::vector<TypeSubstitution> &substitutions)
    -> std::unique_ptr<Stat> {
  if (auto *variable = dyn_cast<VariableStat>(node)) {
    auto clonedVariable = std::make_unique<VariableExpr>(
        variable->variable()->location(), variable->variable()->name());
    auto clonedInit = variable->init()
                          ? substituteExpr(variable->init().get(), substitutions)
                          : nullptr;
    return std::make_unique<VariableStat>(
        variable->location(), std::move(clonedVariable),
        variable->hasExplicitType()
            ? substituteTypeNode(variable->typeNode(), substitutions)
            : nullptr,
        std::move(clonedInit), variable->isConstBinding(),
        variable->canMutateObject());
  }

  if (auto *ifStat = dyn_cast<IfStat>(node)) {
    auto elseBlock = ifStat->hasElseBlock()
                         ? substituteBlockExpr(ifStat->elseBlock().get(),
                                               substitutions)
                         : nullptr;
    return std::make_unique<IfStat>(
        ifStat->location(),
        substituteExpr(ifStat->conditionExpr().get(), substitutions),
        substituteBlockExpr(ifStat->thenBlock().get(), substitutions),
        std::move(elseBlock));
  }

  if (auto *matchStat = dyn_cast<MatchStat>(node)) {
    VectorUniquePtr<MatchArm> arms;
    for (auto &arm : matchStat->arms())
      arms.push_back(std::make_unique<MatchArm>(
          arm->location(), cloneDataPattern(arm->pattern().get()),
          substituteBlockExpr(arm->bodyBlock().get(), substitutions)));
    return std::make_unique<MatchStat>(
        matchStat->location(),
        substituteExpr(matchStat->value().get(), substitutions),
        std::move(arms));
  }

  if (auto *whileStat = dyn_cast<WhileStat>(node))
    return std::make_unique<WhileStat>(
        whileStat->location(),
        substituteExpr(whileStat->conditionExpr().get(), substitutions),
        substituteBlockExpr(whileStat->bodyBlock().get(), substitutions));

  if (auto *forStat = dyn_cast<ForStat>(node))
    return std::make_unique<ForStat>(
        forStat->location(), forStat->variableName(),
        substituteExpr(forStat->startExpr().get(), substitutions),
        substituteExpr(forStat->endExpr().get(), substitutions),
        substituteBlockExpr(forStat->bodyBlock().get(), substitutions));

  if (auto *breakStat = dyn_cast<BreakStat>(node))
    return std::make_unique<BreakStat>(breakStat->location());

  if (auto *continueStat = dyn_cast<ContinueStat>(node))
    return std::make_unique<ContinueStat>(continueStat->location());

  if (auto *returnStat = dyn_cast<ReturnStat>(node)) {
    auto expression = returnStat->hasExpression()
                          ? substituteExpr(returnStat->expression().get(),
                                           substitutions)
                          : nullptr;
    return std::make_unique<ReturnStat>(returnStat->location(),
                                        std::move(expression));
  }

  auto *exprStat = cast<ExprStat>(node);
  return std::make_unique<ExprStat>(
      exprStat->location(),
      substituteExpr(exprStat->expression().get(), substitutions));
}

auto substituteComptimeBlockExpr(
    const ComptimeBlockExpr *node,
    const std::vector<TypeSubstitution> &substitutions)
    -> std::unique_ptr<ComptimeBlockExpr> {
  VectorUniquePtr<Stat> statements;
  for (auto &statement : node->statements())
    statements.push_back(substituteStat(statement.get(), substitutions));

  return std::make_unique<ComptimeBlockExpr>(
      node->location(), std::move(statements),
      substituteExpr(node->result().get(), substitutions));
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
  case Expr::Expr_ComptimeBlock:
    return substituteComptimeBlockExpr(cast<ComptimeBlockExpr>(node),
                                        substitutions);
  case Expr::Expr_TypeInfo: {
    auto *expr = cast<TypeInfoExpr>(node);
    return std::make_unique<TypeInfoExpr>(
        expr->location(), substituteTypeNode(expr->typeNode(), substitutions));
  }
  case Expr::Expr_CompileError: {
    auto *expr = cast<CompileErrorExpr>(node);
    return std::make_unique<CompileErrorExpr>(
        expr->location(), substituteExpr(expr->message().get(), substitutions));
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
    for (auto &argument : expr->expressions()) {
      auto substituted = substituteExpr(argument.get(), substitutions);
      if (argument->isPackExpansion())
        substituted->setPackExpansion();
      expressions.push_back(std::move(substituted));
    }
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
                             const std::vector<TypeSubstitution> &substitutions,
                             const std::vector<ComptimeValue> &comptimeArguments,
                             const std::vector<InferredComptimeArgument> *
                                 inferredArguments)
    -> std::unique_ptr<FunctionDecl> {
  VectorUniquePtr<ParameterDecl> parameters;
  size_t comptimeArgumentIndex = 0;
  std::string concretePackName;
  std::vector<std::string> concretePackElements;
  for (auto &parameter : node->proto()->parameters()) {
    if (parameter->isPack()) {
      concretePackName = std::string(parameter->variable()->name());
      auto *packType = dyn_cast<NamedTypeNode>(parameter->typeNode());
      const InferredComptimeArgument *packArgument = nullptr;
      if (packType && inferredArguments) {
        for (size_t index = 0;
             index < node->proto()->comptimeParameters().size(); ++index) {
          auto &comptimeParameter = node->proto()->comptimeParameters()[index];
          if (comptimeParameter.kind != ComptimeParam::Kind::TypePack ||
              comptimeParameter.name != packType->name())
            continue;
          if (index < inferredArguments->size())
            packArgument = &(*inferredArguments)[index];
          break;
        }
      }

      assert(packArgument && packArgument->packResolved);
      for (size_t index = 0; index < packArgument->types.size(); ++index) {
        auto parameterName = std::string(parameter->variable()->name());
        parameterName += "__";
        parameterName += std::to_string(index);
        auto variable = std::make_unique<VariableExpr>(
            parameter->variable()->location(), parameterName);
        auto concreteParameter = std::make_unique<ParameterDecl>(
            parameter->location(), std::move(variable),
            typeToTypeNode(packArgument->types[index], parameter->location()),
            parameter->canMutateObject());
        parameters.push_back(std::move(concreteParameter));
        concretePackElements.push_back(parameterName);
      }
      continue;
    }

    auto variable = std::make_unique<VariableExpr>(
        parameter->variable()->location(), parameter->variable()->name());
    auto concreteParameter = std::make_unique<ParameterDecl>(
        parameter->location(), std::move(variable),
        substituteTypeNode(parameter->typeNode(), substitutions),
        parameter->canMutateObject(), parameter->isComptime());
    if (parameter->isComptime()) {
      assert(comptimeArgumentIndex < comptimeArguments.size());
      concreteParameter->setComptimeValue(
          comptimeArguments[comptimeArgumentIndex++]);
    }
    parameters.push_back(std::move(concreteParameter));
  }

  auto functionName =
      std::make_unique<FunctionName>(node->proto()->id()->location(),
                                     concreteName);
  auto prototype = std::make_unique<Prototype>(
      node->proto()->location(), std::move(functionName),
      std::move(parameters),
      substituteTypeNode(node->proto()->returnTypeNode(), substitutions));
  prototype->setIsMethod(node->proto()->isMethod());
  if (!concretePackName.empty())
    prototype->setConcretePack(std::move(concretePackName),
                               std::move(concretePackElements));
  auto function = std::make_unique<FunctionDecl>(
      node->location(), std::move(prototype),
      substituteBlockExpr(node->body().get(), substitutions));
  function->setVisibility(node->visibility());
  return function;
}

} // namespace mulberry
