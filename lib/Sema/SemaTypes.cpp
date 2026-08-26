//===--- SemaTypes.cpp - Type semantic analysis --------------------------===//

// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information

//===----------------------------------------------------------------------===//

#include "SemaImpl.h"
#include "SemaComptime.h"
#include "SemaSupport.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "Sema"

namespace mulberry {
using llvm::cast;
using llvm::dyn_cast;

auto SemaImpl::isTypePublic(const Type *type) -> bool {
  if (!type)
    return false;

  if (getBuiltinType(type))
    return true;

  if (auto *arrayType = getArrayType(type))
    return isTypePublic(arrayType->elementType());

  if (auto *functionType = getFunctionType(type)) {
    for (auto *parameterType : functionType->parameterTypes())
      if (!isTypePublic(parameterType))
        return false;
    return isTypePublic(functionType->returnType());
  }

  if (auto *ptrType = getPtrType(type))
    return isTypePublic(ptrType->pointeeType());

  if (auto *structType = getStructType(type)) {
    if (auto *origin = structType->origin()) {
      auto *alias = lookupComptimeTypeAliasSymbol(origin->aliasName());
      if (!alias || alias->visibility != Visibility::Public)
        return false;
      for (auto &argument : origin->arguments())
        if (argument.kind() == ComptimeValue::Kind::Type &&
            !isTypePublic(argument.type()))
          return false;
      return true;
    }

    if (auto *symbol = lookupTypeSymbol(structType->name()))
      return symbol->visibility == Visibility::Public;

    // Inline anonymous structs have no declaration symbol. Their fields are
    // nevertheless part of the public type and must be checked recursively.
    for (auto &field : structType->fields())
      if (!isTypePublic(field.type()))
        return false;
    return true;
  }

  if (auto *dataType = getDataType(type)) {
    auto *symbol = lookupDataDeclSymbol(dataType->declarationName());
    if (!symbol || symbol->visibility != Visibility::Public)
      return false;
    for (auto &argument : dataType->arguments())
      if (argument.kind() == ComptimeValue::Kind::Type &&
          !isTypePublic(argument.type()))
        return false;
    return true;
  }

  return false;
}

auto SemaImpl::checkPublicTypeNode(
    const TypeNode *typeNode,
    const std::vector<ComptimeParam> &comptimeParameters, bool allowSelf)
    -> llvm::LogicalResult {
  auto rejectPrivateType = [&](const Node *node,
                               std::string_view typeName) {
    LLVM_DEBUG(llvm::dbgs() << "public type check rejected private type `"
                            << typeName << "`\n");
    auto diagnostic = formatNameDiagnostic(
        diag::public_signature_private_type, typeName);
    return emitError(node, diagnostic);
  };

  if (auto *namedType = dyn_cast<NamedTypeNode>(typeNode)) {
    for (auto &parameter : comptimeParameters)
      if (namedType->name() == parameter.name)
        return success();

    if (namedType->name() == "Self" && allowSelf)
      return success();

    if (auto *resolvedType = namedType->resolvedType()) {
      if (isTypePublic(resolvedType))
        return success();
      auto typeName = formatType(resolvedType);
      return rejectPrivateType(typeNode, typeName);
    }

    if (auto *symbol = lookupTypeSymbol(namedType->name())) {
      if (symbol->visibility == Visibility::Public)
        return success();
      return rejectPrivateType(typeNode, namedType->name());
    }

    if (auto *symbol = lookupDataDeclSymbol(namedType->name())) {
      if (symbol->visibility == Visibility::Public)
        return success();
      return rejectPrivateType(typeNode, namedType->name());
    }

    if (auto *symbol = lookupComptimeTypeAliasSymbol(namedType->name())) {
      if (symbol->visibility == Visibility::Public)
        return success();
      return rejectPrivateType(typeNode, namedType->name());
    }

    // Undefined names are diagnosed by ordinary type resolution. This check
    // only adds the visibility rule and must not replace that diagnostic.
    return success();
  }

  if (dyn_cast<SelfTypeNode>(typeNode))
    return success();

  if (auto *computedType = dyn_cast<ComputedTypeNode>(typeNode)) {
    if (containsComptimeParameter(computedType, comptimeParameters))
      return success();
    auto *type = resolveType(computedType);
    if (!type)
      return failure();
    if (isTypePublic(type))
      return success();
    auto typeName = formatType(type);
    return rejectPrivateType(typeNode, typeName);
  }

  if (auto *arrayType = dyn_cast<ArrayTypeNode>(typeNode))
    return checkPublicTypeNode(arrayType->elementTypeNode(),
                               comptimeParameters, allowSelf);

  if (auto *ptrType = dyn_cast<PtrTypeNode>(typeNode))
    return checkPublicTypeNode(ptrType->pointeeTypeNode(),
                               comptimeParameters, allowSelf);

  if (auto *functionType = dyn_cast<FunctionTypeNode>(typeNode)) {
    for (auto &parameterType : functionType->parameterTypes())
      if (llvm::failed(checkPublicTypeNode(parameterType.get(),
                                           comptimeParameters, allowSelf)))
        return failure();
    return checkPublicTypeNode(functionType->returnTypeNode(),
                               comptimeParameters, allowSelf);
  }

  if (auto *genericType = dyn_cast<GenericTypeNode>(typeNode)) {
    if (genericType->name() != "Array") {
      if (auto *data = lookupDataDeclSymbol(genericType->name())) {
        if (data->visibility != Visibility::Public)
          return rejectPrivateType(typeNode, genericType->name());
      } else if (auto *alias =
                     lookupComptimeTypeAliasSymbol(genericType->name())) {
        if (alias->visibility != Visibility::Public)
          return rejectPrivateType(typeNode, genericType->name());
      } else if (auto *type = lookupTypeSymbol(genericType->name())) {
        if (type->visibility != Visibility::Public)
          return rejectPrivateType(typeNode, genericType->name());
      }
    }

    for (auto &argument : genericType->arguments()) {
      if (argument.kind() != ComptimeArg::Kind::Type)
        continue;
      if (llvm::failed(checkPublicTypeNode(argument.typeNode(),
                                           comptimeParameters, allowSelf)))
        return failure();
    }
    return success();
  }

  auto *structType = dyn_cast<StructTypeNode>(typeNode);
  if (!structType)
    return success();
  for (auto &field : structType->fields())
    if (llvm::failed(checkPublicTypeNode(field->typeNode(),
                                         comptimeParameters, allowSelf)))
      return failure();
  return success();
}


auto SemaImpl::resolveType(const NamedTypeNode *typeNode) -> const Type * {
    if (auto *resolvedType = typeNode->resolvedType()) {
      LLVM_DEBUG(llvm::dbgs() << "reuse resolved type `" << typeNode->name()
                              << "` as `" << formatType(resolvedType)
                              << "`\n");
      return resolvedType;
    }
    auto *type = lookupType(typeNode->name());
    if (!type) {
      (void)emitError(typeNode, diag::undefined_type);
      return nullptr;
    }

    return type;
  }

auto SemaImpl::resolveType(const SelfTypeNode *typeNode) -> const Type * {
    if (!_activeSelfType) {
      (void)emitError(typeNode, diag::self_type_outside_trait_impl);
      return nullptr;
    }

    LLVM_DEBUG(llvm::dbgs() << "resolve `Self` as `"
                            << formatType(_activeSelfType) << "`\n");
    return _activeSelfType;
  }

auto SemaImpl::resolveType(const UnitTypeNode *typeNode) -> const Type * {
    return _typeContext.getBuiltinType(BuiltinTypeKind::Unit);
  }

auto SemaImpl::resolveType(const ArrayTypeNode *typeNode) -> const Type * {
    auto *elementType = resolveType(typeNode->elementTypeNode());
    if (!elementType)
      return nullptr;

    auto &shape = typeNode->shape();
    if (shape.size() == 1 && shape.front() >= 0) {
      if (!isArrayElementType(elementType)) {
        (void)emitError(typeNode->elementTypeNode(), diag::mismatch_type);
        return nullptr;
      }
      return _typeContext.createArrayType(elementType, shape.front());
    }

    (void)emitError(typeNode, diag::mismatch_type);
    return nullptr;
  }

auto SemaImpl::resolveType(const PtrTypeNode *typeNode) -> const Type * {
    if (llvm::failed(checkInternalFeature(typeNode->location())))
      return nullptr;

    auto *pointeeType = resolveType(typeNode->pointeeTypeNode());
    if (!pointeeType)
      return nullptr;

    return _typeContext.createPtrType(pointeeType);
  }

auto SemaImpl::resolveType(const FunctionTypeNode *typeNode) -> const Type * {
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

auto SemaImpl::resolveStructFields(const StructTypeNode *typeNode)
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
        (void)emitError(variable, diag::redefinition_var);
        return std::nullopt;
      }
      fields.push_back(StructField{fieldName, fieldType, fieldIndex++});
    }
    return fields;
  }

auto SemaImpl::resolveType(const StructTypeNode *typeNode, std::string_view name)
      -> const StructType * {
    auto fields = resolveStructFields(typeNode);
    if (!fields)
      return nullptr;
    return _typeContext.createStructType(name, std::move(*fields));
  }

auto SemaImpl::resolveType(const StructTypeNode *typeNode, std::string_view name,
                   ComptimeAliasOrigin origin) -> const StructType * {
    auto fields = resolveStructFields(typeNode);
    if (!fields)
      return nullptr;
    return _typeContext.createStructType(name, std::move(*fields),
                                         std::move(origin));
  }

auto SemaImpl::resolveType(const ComputedTypeNode *typeNode) -> const Type * {
    auto result = ComptimeSema(*this).evaluateComptime(
        typeNode->expression().get());
    if (result.kind == ComptimeEvaluation::Kind::Error)
      return nullptr;
    if (result.kind != ComptimeEvaluation::Kind::Static ||
        result.value->kind() != ComptimeValue::Kind::Type) {
      (void)emitError(typeNode, diag::expected_comptime_type);
      return nullptr;
    }

    auto *type = result.value->type();
    LLVM_DEBUG(llvm::dbgs() << "resolved computed type `"
                            << formatType(type) << "`\n");
    return type;
  }

auto SemaImpl::completeDataType(const DataDecl *decl, DataType *dataType,
                        const std::vector<TypeSubstitution> &substitutions)
      -> llvm::LogicalResult {
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

auto SemaImpl::instantiateDataType(
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
    if (llvm::failed(completeDataType(decl, dataType, substitutions)))
      return nullptr;
    return dataType;
  }

auto SemaImpl::resolveType(const GenericTypeNode *typeNode) -> const Type * {
    if (typeNode->name() == "Array") {
      auto &arguments = typeNode->arguments();
      if (arguments.size() != 2 ||
          arguments[0].kind() != ComptimeArg::Kind::Type ||
          arguments[1].kind() != ComptimeArg::Kind::UInt64) {
        (void)emitError(typeNode, diag::mismatch_type);
        return nullptr;
      }

      auto *elementType = resolveType(arguments[0].typeNode());
      if (!elementType)
        return nullptr;
      if (!isArrayElementType(elementType)) {
        (void)emitError(arguments[0].typeNode(), diag::mismatch_type);
        return nullptr;
      }
      return _typeContext.createArrayType(elementType,
                                          arguments[1].uint64Value());
    }

    auto declarationName = dataDeclName(typeNode->name());
    if (!declarationName.empty()) {
      auto *decl = _symbols.lookupDataDecl(declarationName)->decl;
      if (typeNode->arguments().size() != decl->parameters().size()) {
        (void)emitError(typeNode, diag::mismatch_type);
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

        (void)emitError(typeNode, diag::mismatch_type);
        return nullptr;
      }
      return instantiateDataType(decl, arguments, substitutions);
    }

    auto aliasName = comptimeTypeAliasName(typeNode->name());
    auto *alias = aliasName.empty()
                      ? nullptr
                      : _symbols.lookupComptimeTypeAlias(aliasName);
    if (!alias) {
      (void)emitError(typeNode, diag::undefined_type);
      return nullptr;
    }
    if (typeNode->arguments().size() != alias->parameters.size()) {
      (void)emitError(typeNode, diag::mismatch_type);
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

      (void)emitError(typeNode, diag::mismatch_type);
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

auto SemaImpl::resolveType(const TypeNode *typeNode) -> const Type * {
    if (auto *unitType = dyn_cast<UnitTypeNode>(typeNode))
      return resolveType(unitType);

    if (auto *selfType = dyn_cast<SelfTypeNode>(typeNode))
      return resolveType(selfType);

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

auto SemaImpl::checkType(const TypeNode *typeNode, UnitPolicy unitPolicy)
      -> const Type * {
    auto *type = resolveType(typeNode);
    if (!type)
      return nullptr;
    if (unitPolicy == UnitPolicy::Reject && isUnitType(type)) {
      (void)emitError(typeNode, diag::unexpected_unit_type);
      return nullptr;
    }
    if (hasUnitElementType(type)) {
      (void)emitError(typeNode, diag::unexpected_unit_type);
      return nullptr;
    }
    return type;
  }

auto SemaImpl::setBuiltinType(Expr *expr, BuiltinTypeKind kind) -> void {
    expr->setType(_typeContext.getBuiltinType(kind));
  }

} // namespace mulberry
