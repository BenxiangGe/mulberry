//===--- SemaDeclarations.cpp - Declaration semantic analysis --------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "SemaDeclarations.h"
#include "SemaComptime.h"
#include "SemaImpl.h"
#include "SemaSupport.h"
#include "SemaTraits.h"
#include "llvm/Support/Debug.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "Sema"

namespace mulberry {
using llvm::dyn_cast;

auto DeclarationSema::functionPackageName(std::string_view name) const
    -> std::string {
  auto package = _sema._instantiatedFunctionPackages.find(std::string(name));
  if (package != _sema._instantiatedFunctionPackages.end())
    return package->second;
  package = _sema._functionPackages.find(std::string(name));
  if (package != _sema._functionPackages.end())
    return package->second;
  package = _sema._genericFunctionPackages.find(std::string(name));
  if (package != _sema._genericFunctionPackages.end())
    return package->second;
  return packageNameOf(name);
}

auto DeclarationSema::sema(Prototype *node) -> llvm::LogicalResult {
  if (node->isGeneric())
    return success();

  return semaFunctionSignature(node);
}

auto DeclarationSema::semaFunctionParameters(
    Prototype *node, std::vector<const Type *> &parameterTypes,
    std::vector<bool> &parameterCanMutateObject) -> llvm::LogicalResult {
  for (const auto &indexedParameter : llvm::enumerate(node->parameters())) {
    auto &par = indexedParameter.value();
    auto *parameterType =
        _sema.checkType(par->typeNode(), SemaImpl::UnitPolicy::Reject);
    if (!parameterType)
      return failure();
    par->setType(parameterType);
    auto canMutateObject = par->canMutateObject();
    if (par->isComptime()) {
      auto *stringType = _sema.lookupType("String");
      if (!stringType || !sameType(parameterType, stringType))
        return _sema.emitError(par->typeNode(), diag::mismatch_type);
      if (!par->comptimeValue())
        return _sema.emitError(par->variable().get(),
                               diag::expected_comptime_value);
      auto valueType = ComptimeSema(_sema).comptimeRuntimeType(
          *par->comptimeValue());
      if (!valueType || !sameType(valueType, parameterType))
        return _sema.emitError(par->variable().get(), diag::mismatch_type);
      if (llvm::failed(_sema.declareVariable(
              par->variable()->name(), parameterType,
              /*isConstBinding=*/true,
              /*canMutateObject=*/false, par->comptimeValue(),
              /*isComptimeOnly=*/true)))
        return _sema.emitError(par->variable().get(), diag::redefinition_var);
      continue;
    }
    if (llvm::failed(_sema.declareVariable(
            par->variable()->name(), parameterType, !canMutateObject,
            canMutateObject)))
      return _sema.emitError(par->variable().get(), diag::redefinition_var);
    parameterTypes.push_back(parameterType);
    parameterCanMutateObject.push_back(canMutateObject);
  }
  return success();
}

auto DeclarationSema::checkPublicFunctionSignature(
    Prototype *node, const std::vector<ComptimeParam> &comptimeParameters,
    bool allowSelf) -> llvm::LogicalResult {
  for (auto &parameter : node->parameters())
    if (llvm::failed(_sema.checkPublicTypeNode(
            parameter->typeNode(), comptimeParameters, allowSelf)))
      return failure();
  return _sema.checkPublicTypeNode(node->returnTypeNode(), comptimeParameters,
                                  allowSelf);
}

auto DeclarationSema::checkFunctionPacks(Prototype *node)
    -> llvm::LogicalResult {
  const ComptimeParam *typePack = nullptr;
  for (auto &parameter : node->comptimeParameters()) {
    if (!parameter.isTypePack())
      continue;
    typePack = &parameter;
    break;
  }

  const ParameterDecl *valuePack = nullptr;
  for (auto &parameter : node->parameters()) {
    if (!parameter->isPack())
      continue;
    valuePack = parameter.get();
    break;
  }

  if (typePack && !valuePack)
    return _sema.emitError(node, diag::type_pack_requires_value_pack);
  if (!valuePack)
    return success();

  auto *packType = dyn_cast<NamedTypeNode>(valuePack->typeNode());
  if (!typePack || !packType || packType->name() != typePack->name)
    return _sema.emitError(valuePack, diag::value_pack_requires_type_pack);
  return success();
}

auto DeclarationSema::bindFunctionParameters(
    Prototype *node, const FunctionSymbol *signature)
    -> llvm::LogicalResult {
  auto &parameters = node->parameters();
  size_t runtimeParameterIndex = 0;
  for (size_t i = 0; i < parameters.size(); ++i) {
    auto &parameter = parameters[i];
    if (parameter->isComptime()) {
      auto *parameterType =
          _sema.checkType(parameter->typeNode(), SemaImpl::UnitPolicy::Reject);
      if (!parameterType)
        return failure();
      parameter->setType(parameterType);
      if (!parameter->comptimeValue())
        return _sema.emitError(parameter->variable().get(),
                               diag::expected_comptime_value);
      if (llvm::failed(_sema.declareVariable(
              parameter->variable()->name(), parameterType,
              /*isConstBinding=*/true,
              /*canMutateObject=*/false, parameter->comptimeValue(),
              /*isComptimeOnly=*/true)))
        return _sema.emitError(parameter->variable().get(),
                               diag::redefinition_var);
      continue;
    }
    auto *parameterType =
        signature->type->parameterTypes()[runtimeParameterIndex];
    auto canMutateObject =
        signature->type->parameterCanMutateObject()[runtimeParameterIndex];
    ++runtimeParameterIndex;
    parameter->setType(parameterType);
    if (llvm::failed(_sema.declareVariable(
            parameter->variable()->name(), parameterType, !canMutateObject,
            canMutateObject)))
      return _sema.emitError(parameter->variable().get(),
                             diag::redefinition_var);
  }
  node->setType(signature->type->returnType());
  return success();
}

auto DeclarationSema::semaFunctionSignature(
    Prototype *node, bool isExtern, std::string_view packageName,
    Visibility visibility)
    -> llvm::LogicalResult {
  std::vector<const Type *> parameterTypes;
  std::vector<bool> parameterCanMutateObject;
  if (llvm::failed(semaFunctionParameters(
          node, parameterTypes, parameterCanMutateObject)))
    return failure();

  auto *returnType = _sema.resolveType(node->returnTypeNode());
  if (!returnType)
    return failure();
  node->setType(returnType);

  auto name = node->id()->name();
  auto *functionType = _sema._typeContext.createFunctionType(
      std::move(parameterTypes), std::move(parameterCanMutateObject),
      returnType);
  if (llvm::failed(_sema.declareFunction(name, functionType, isExtern,
                                         packageName, visibility))) {
    auto diagnostic = formatNameDiagnostic(diag::redefinition_func, name);
    return _sema.emitError(node->id().get(), diagnostic);
  }
  return success();
}

auto DeclarationSema::sema(FunctionDecl *node) -> llvm::LogicalResult {
  auto functionName = node->proto()->id()->name();
  std::string functionPackage;
  if (node->isExtern()) {
    if (functionName.find('.') != std::string_view::npos)
      functionPackage = packageNameOf(functionName);
    else if (!node->sourcePackageName().empty())
      functionPackage = std::string(node->sourcePackageName());
    else
      functionPackage = _sema._currentPackageName;
  } else {
    functionPackage = functionPackageName(functionName);
  }
  SemaImpl::PackageScope packageScope(_sema._currentPackageName,
                                      functionPackage);
  if (llvm::failed(checkFunctionPacks(node->proto().get())))
    return failure();
  if (node->isPublic() && !node->proto()->isMethod() &&
      !node->specializationOrigin() &&
      llvm::failed(checkPublicFunctionSignature(
          node->proto().get(), node->proto()->comptimeParameters())))
    return failure();
  if (node->isExtern()) {
    if (node->proto()->isGeneric())
      return _sema.emitError(node->proto()->id().get(), diag::mismatch_type);
    SemaImpl::FunctionScope functionScope(_sema);
    return semaFunctionSignature(node->proto().get(), true, functionPackage,
                                 node->visibility());
  }

  if (node->proto()->isGeneric()) {
    if (llvm::failed(TraitSema(_sema).resolveConstraints(
            node->proto().get(), node->proto()->comptimeParameters())))
      return failure();
    auto name = node->proto()->id()->name();
    if (_sema.lookupFunction(name) || _sema.lookupGenericFunction(name)) {
      auto diagnostic = formatNameDiagnostic(diag::redefinition_func, name);
      return _sema.emitError(node->proto()->id().get(), diagnostic);
    }
    if (llvm::failed(_sema.declareGenericFunction(
            name, node, functionPackage, node->visibility()))) {
      auto diagnostic = formatNameDiagnostic(diag::redefinition_func, name);
      return _sema.emitError(node->proto()->id().get(), diagnostic);
    }
    return success();
  }

  auto isConcreteSpecialization = node->proto()->hasConcretePack();
  for (auto &parameter : node->proto()->parameters())
    isConcreteSpecialization =
        isConcreteSpecialization ||
        (parameter->isComptime() && parameter->comptimeValue().has_value());

  const FunctionSymbol *signature = nullptr;

  std::unique_ptr<ComptimeFrame> concretePackFrame;
  if (node->proto()->hasConcretePack()) {
    concretePackFrame = std::make_unique<ComptimeFrame>();
    std::vector<ComptimeBinding> elements;
    for (auto &elementName : node->proto()->concretePackElements()) {
      const ParameterDecl *parameter = nullptr;
      for (auto &candidate : node->proto()->parameters()) {
        if (candidate->variable()->name() == elementName) {
          parameter = candidate.get();
          break;
        }
      }
      if (!parameter || !parameter->type())
        return failure();

      auto residual = std::make_unique<VariableExpr>(node->location(),
                                                      elementName);
      residual->setType(parameter->type());
      elements.push_back(ComptimeBinding::residualValue(
          std::move(residual), parameter->type(),
          /*isConst=*/!parameter->canMutateObject()));
    }
    concretePackFrame->bind(
        node->proto()->concretePackName(),
        ComptimeBinding::pack(std::move(elements)));
    LLVM_DEBUG(llvm::dbgs()
               << "bind concrete pack `"
               << node->proto()->concretePackName() << "` with "
               << node->proto()->concretePackElements().size()
               << " elements in `" << node->proto()->id()->name() << "`\n");
  }

  {
    SemaImpl::FunctionScope functionScope(_sema);
    signature = _sema.lookupFunction(node->proto()->id()->name());
    if (signature) {
      if (llvm::failed(bindFunctionParameters(node->proto().get(), signature)))
        return failure();
    } else {
      if (llvm::failed(semaFunctionSignature(
              node->proto().get(), /*isExtern=*/false, functionPackage,
              node->visibility())))
        return failure();
      signature = _sema.lookupFunction(node->proto()->id()->name());
      if (!signature)
        return failure();
    }
    if (!node->isExtern())
      _sema.registerFunctionDecl(node->proto()->id()->name(), node);
    SemaImpl::FunctionReturnTypeScope returnTypeScope(
        _sema._currentFunctionReturnType, signature->type->returnType());

    auto hasComptimeValue = false;
    for (auto &parameter : node->proto()->parameters())
      hasComptimeValue = hasComptimeValue ||
                         (parameter->isComptime() &&
                          parameter->comptimeValue().has_value());
    if (!concretePackFrame && !hasComptimeValue) {
      if (llvm::failed(_sema.sema(node->body().get())))
        return failure();
    } else if (concretePackFrame) {
      SemaImpl::ComptimeFrameScope frameScope(_sema,
                                              concretePackFrame.get());
      if (llvm::failed(ComptimeSema(_sema).executeStagedFunctionBody(
              node->body().get(), node)))
        return failure();
    } else if (llvm::failed(
                   ComptimeSema(_sema).executeStagedFunctionBody(
                       node->body().get(), node))) {
      return failure();
    }
  }

  if (isConcreteSpecialization) {
    node->proto()->makeOrdinary();
    LLVM_DEBUG(llvm::dbgs()
               << "residualize concrete function `"
               << node->proto()->id()->name()
               << "` as an ordinary FunctionDecl\n");
    // The staged scope still contains comptime parameters. Re-enter with a
    // fresh ordinary scope so residual Sema cannot resolve a removed binding.
    SemaImpl::FunctionScope ordinaryFunctionScope(_sema);
    signature = _sema.lookupFunction(node->proto()->id()->name());
    if (!signature ||
        llvm::failed(bindFunctionParameters(node->proto().get(), signature)))
      return failure();
    SemaImpl::FunctionReturnTypeScope ordinaryReturnTypeScope(
        _sema._currentFunctionReturnType, signature->type->returnType());
    if (llvm::failed(_sema.sema(node->body().get())))
      return failure();
  }

  auto hasReturn = containsReturnStat(node->body().get());
  if (!isUnitType(signature->type->returnType()) && !hasReturn)
    return _sema.emitError(node->proto()->id().get(), diag::wrong_return_type);

  return success();
}

auto DeclarationSema::declareStructMethods(
    std::string_view ownerName, const VectorUniquePtr<FunctionDecl> &methods,
    const std::vector<ComptimeParam> &typeParameters,
    std::string_view packageName, Visibility ownerVisibility)
    -> llvm::LogicalResult {
  NameSet methodNames;
  for (auto &method : methods) {
    auto *prototype = method->proto().get();
    auto methodName = prototype->id()->name();
    if (!declareName(methodNames, methodName)) {
      auto diagnostic = formatNameDiagnostic(diag::redefinition_func,
                                             methodName);
      return _sema.emitError(prototype->id().get(), diagnostic);
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

    if (ownerVisibility == Visibility::Public && method->isPublic() &&
        llvm::failed(checkPublicFunctionSignature(
            prototype, prototype->comptimeParameters())))
      return failure();

    if (prototype->isGeneric()) {
      if (llvm::failed(TraitSema(_sema).resolveConstraints(
              prototype, prototype->comptimeParameters())))
        return failure();
      if (llvm::failed(_sema.declareGenericFunction(
              fullName, method.get(), packageName, method->visibility()))) {
        auto diagnostic = formatNameDiagnostic(diag::redefinition_func,
                                               fullName);
        return _sema.emitError(prototype->id().get(), diagnostic);
      }
      continue;
    }

    _sema._functionPackages[fullName] = std::string(packageName);
    if (llvm::failed(sema(method.get())))
      return failure();
  }
  return success();
}

auto DeclarationSema::sema(StructDecl *node) -> llvm::LogicalResult {
  SemaImpl::PackageScope packageScope(_sema._currentPackageName,
                                      packageNameOf(node->id()->name()));
  std::vector<StructField> fields;
  NameSet fieldNames;
  unsigned fieldIndex = 0;
  for (auto &varDecl : *node) {
    auto var = varDecl->variable().get();
    if (node->isPublic() && llvm::failed(_sema.checkPublicTypeNode(
            varDecl->typeNode(), {}, /*allowSelf=*/false)))
      return failure();
    auto *fieldType =
        _sema.checkType(varDecl->typeNode(), SemaImpl::UnitPolicy::Reject);
    if (!fieldType)
      return failure();
    varDecl->setType(fieldType);
    auto fieldName = var->name();
    if (!declareName(fieldNames, fieldName))
      return _sema.emitError(var, diag::redefinition_var);
    fields.push_back(StructField{fieldName, fieldType, fieldIndex++});
  }
  auto id = node->id().get();
  if (_sema.lookupType(id->name()))
    return _sema.emitError(id, diag::redefinition_type);

  auto *structType =
      _sema._typeContext.createStructType(id->name(), std::move(fields));
  id->setType(structType);
  if (llvm::failed(_sema.declareStructType(
          structType, packageNameOf(id->name()), node->visibility())))
    return _sema.emitError(id, diag::redefinition_type);
  if (llvm::failed(declareStructMethods(
          id->name(), node->methods(), {}, packageNameOf(id->name()),
          node->visibility())))
    return failure();
  return success();
}

auto DeclarationSema::sema(DataDecl *node) -> llvm::LogicalResult {
  auto dataPackage = packageNameOf(node->name());
  SemaImpl::PackageScope packageScope(_sema._currentPackageName, dataPackage);
  if (node->isPublic()) {
    for (auto &constructor : node->constructors())
      for (auto &payloadType : constructor->payloadTypes())
        if (llvm::failed(_sema.checkPublicTypeNode(
                payloadType.get(), node->parameters(), /*allowSelf=*/false)))
          return failure();
  }

  if (_sema._symbols.lookupType(node->name()) ||
      _sema._symbols.lookupComptimeTypeAlias(node->name()) ||
      _sema._symbols.lookupDataDecl(node->name()))
    return _sema.emitError(node, diag::redefinition_type);

  if (llvm::failed(_sema._symbols.declareDataDecl(
          node->name(), node, dataPackage, node->visibility())))
    return _sema.emitError(node, diag::redefinition_type);

  for (const auto &indexedConstructor :
       llvm::enumerate(node->constructors())) {
    auto &constructor = indexedConstructor.value();
    if (llvm::failed(_sema._symbols.declareDataConstructor(
            constructor->name(), node, indexedConstructor.index())))
      return _sema.emitError(constructor.get(),
                             diag::redefinition_data_constructor);
  }

  if (node->isGeneric())
    return success();

  std::vector<ComptimeArgument> arguments;
  auto concreteName = genericTypeName(node->name(), arguments);
  auto *dataType = _sema._typeContext.createDataType(
      concreteName, node->name(), {});
  _sema._dataTypes[concreteName] = dataType;

  // Publish a non-generic shell before resolving its constructors so direct
  // self-reference can resolve through the ordinary type symbol table.
  if (llvm::failed(_sema.declareType(node->name(), dataType, dataPackage,
                                     node->visibility())))
    return _sema.emitError(node, diag::redefinition_type);
  return _sema.completeDataType(node, dataType, {});
}

auto DeclarationSema::sema(ComptimeTypeAliasDecl *node)
    -> llvm::LogicalResult {
  auto packageName = packageNameOf(node->name());
  SemaImpl::PackageScope packageScope(_sema._currentPackageName, packageName);
  if (_sema._symbols.lookupType(node->name()) ||
      _sema._symbols.lookupComptimeTypeAlias(node->name()))
    return _sema.emitError(node, diag::redefinition_type);

  if (node->isPublic() && llvm::failed(_sema.checkPublicTypeNode(
          node->bodyTypeNode(), node->parameters(), /*allowSelf=*/false)))
    return failure();

  if (!node->isGeneric()) {
    auto *bodyType =
        _sema.checkType(node->bodyTypeNode(), SemaImpl::UnitPolicy::Reject);
    if (!bodyType)
      return failure();
    if (llvm::failed(_sema._symbols.declareType(
            node->name(), bodyType, packageName, node->visibility())))
      return _sema.emitError(node, diag::redefinition_type);
    if (auto *structTypeNode = dyn_cast<StructTypeNode>(node->bodyTypeNode()))
      return declareStructMethods(node->name(), structTypeNode->methods(), {},
                                  packageName, node->visibility());
    return success();
  }

  if (llvm::failed(_sema._symbols.declareComptimeTypeAlias(
          node->name(), packageName,
          std::vector<ComptimeParam>(node->parameters().begin(),
                                     node->parameters().end()),
          node->bodyTypeNode(), node->visibility())))
    return _sema.emitError(node, diag::redefinition_type);
  if (auto *structTypeNode = dyn_cast<StructTypeNode>(node->bodyTypeNode())) {
    if (llvm::failed(declareStructMethods(node->name(),
                                          structTypeNode->methods(),
                                          node->parameters(), packageName,
                                          node->visibility())))
      return failure();
  }
  return success();
}

} // namespace mulberry
