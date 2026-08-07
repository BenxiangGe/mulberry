//===--- SemaGenerics.cpp - Generic function semantic analysis ------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "SemaExpressions.h"
#include "SemaImpl.h"
#include "SemaSupport.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <utility>

#undef DEBUG_TYPE
#define DEBUG_TYPE "Sema"

namespace mulberry {
using llvm::cast;
using llvm::dyn_cast;

auto ExpressionSema::makeInferredComptimeArguments(
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

auto ExpressionSema::comptimeSubstitutions(
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
    if (parameter.kind == ComptimeParam::Kind::TypePack)
      continue;
    substitutions.push_back(TypeSubstitution{
        parameter.name, nullptr, *argument.uint64Value});
  }
  return substitutions;
}

auto ExpressionSema::comptimeParameterIndex(
    const std::vector<ComptimeParam> &parameters,
    std::string_view name) const -> std::optional<size_t> {
  for (size_t i = 0; i < parameters.size(); ++i)
    if (parameters[i].name == name)
      return i;
  return std::nullopt;
}

auto ExpressionSema::hasComputedType(const TypeNode *typeNode,
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

    auto aliasName = _sema.comptimeTypeAliasName(genericType->name());
    if (aliasName.empty())
      return false;
    auto [iter, inserted] = visitingAliases.insert(aliasName);
    if (!inserted)
      return false;

    auto *symbol = _sema._symbols.lookupComptimeTypeAlias(aliasName);
    SemaImpl::PackageScope packageScope(_sema._currentPackageName,
                                        symbol->packageName);
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

auto ExpressionSema::hasComputedType(const TypeNode *typeNode) -> bool {
  NameSet visitingAliases;
  return hasComputedType(typeNode, visitingAliases);
}

auto ExpressionSema::bindComptimeTypeArgument(
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

auto ExpressionSema::bindComptimeTypePackArgument(
    const Type *type, InferredComptimeArgument &argument) -> bool {
  if (argument.kind != ComptimeParam::Kind::TypePack)
    return false;
  argument.types.push_back(type);
  LLVM_DEBUG(llvm::dbgs() << "bind type pack element " << formatType(type)
                          << "\n");
  return true;
}

auto ExpressionSema::bindComptimeUInt64Argument(
    uint64_t value, InferredComptimeArgument &argument) -> bool {
  if (argument.kind != ComptimeParam::Kind::UInt64)
    return false;
  if (!argument.uint64Value) {
    argument.uint64Value = value;
    return true;
  }
  return *argument.uint64Value == value;
}

auto ExpressionSema::computedArrayLeafParameterIndex(
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

auto ExpressionSema::matchComptimeArgument(
    const ComptimeArg &pattern, const ComptimeValue &actual,
    const std::vector<ComptimeParam> &parameters,
    std::vector<InferredComptimeArgument> &arguments,
    std::vector<const Type *> *arrayLeafConstraints) -> bool {
  if (pattern.kind() == ComptimeArg::Kind::UInt64)
    return actual.kind() == ComptimeValue::Kind::UInt64 &&
           pattern.uint64Value() == actual.uint64Value();

  if (auto *namedType = dyn_cast<NamedTypeNode>(pattern.typeNode())) {
    if (auto index = comptimeParameterIndex(parameters, namedType->name())) {
      if (actual.kind() == ComptimeValue::Kind::Type) {
        if (parameters[*index].kind == ComptimeParam::Kind::TypePack)
          return bindComptimeTypePackArgument(actual.type(), arguments[*index]);
        return bindComptimeTypeArgument(actual.type(), arguments[*index],
                                        namedType->location());
      }
      return bindComptimeUInt64Argument(actual.uint64Value(),
                                        arguments[*index]);
    }
  }

  if (actual.kind() != ComptimeValue::Kind::Type)
    return false;
  return matchGenericType(pattern.typeNode(), actual.type(), parameters,
                          arguments, arrayLeafConstraints);
}

auto ExpressionSema::matchGenericType(
    const TypeNode *pattern, const Type *actualType,
    const std::vector<ComptimeParam> &parameters,
    std::vector<InferredComptimeArgument> &arguments,
    std::vector<const Type *> *arrayLeafConstraints) -> bool {
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
    if (auto index = comptimeParameterIndex(parameters, namedType->name())) {
      if (parameters[*index].kind == ComptimeParam::Kind::TypePack)
        return bindComptimeTypePackArgument(actualType, arguments[*index]);
      return bindComptimeTypeArgument(actualType, arguments[*index],
                                      namedType->location());
    }

    auto *patternType = _sema.resolveType(namedType);
    return sameType(patternType, actualType);
  }

  if (auto *arrayPattern = dyn_cast<ArrayTypeNode>(pattern)) {
    auto *arrayType = getArrayType(actualType);
    return arrayType && arrayPattern->shape().size() == 1 &&
           arrayPattern->shape().front() >= 0 &&
           static_cast<uint64_t>(arrayPattern->shape().front()) ==
               arrayType->size() &&
           matchGenericType(arrayPattern->elementTypeNode(),
                            arrayType->elementType(), parameters, arguments,
                            arrayLeafConstraints);
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
                            functionType->returnType(), parameters, arguments,
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
                            arrayType->elementType(), parameters, arguments,
                            arrayLeafConstraints))
        return false;
      auto sizeValue = ComptimeValue(arrayType->size());
      return matchComptimeArgument(patternArguments[1], sizeValue,
                                   parameters, arguments,
                                   arrayLeafConstraints);
    }

    auto dataName = _sema.dataDeclName(genericPattern->name());
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

    auto aliasName = _sema.comptimeTypeAliasName(genericPattern->name());
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

    auto *alias = _sema.lookupComptimeTypeAlias(genericPattern->name());
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

      auto *argumentType = _sema.resolveType(argumentTypeNode);
      if (!argumentType)
        return false;
      auto resolvedArgumentTypeNode =
          typeToTypeNode(argumentType, argumentTypeNode->location());
      substitutions.push_back(TypeSubstitution{
          aliasParameter.name, resolvedArgumentTypeNode.get(), std::nullopt});
    }

    // Expand the alias in its defining package before matching its body.
    // This lets generic methods infer T from aliases such as `List<T>`.
    auto aliasBody = substituteTypeNode(alias->bodyTypeNode, substitutions);
    SemaImpl::PackageScope packageScope(_sema._currentPackageName,
                                        alias->packageName);
    return matchGenericType(aliasBody.get(), actualType, parameters, arguments,
                            arrayLeafConstraints);
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

  auto *patternType = _sema.resolveType(pattern);
  return sameType(patternType, actualType);
}

auto ExpressionSema::matchMethodReceiverType(
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

} // namespace mulberry
