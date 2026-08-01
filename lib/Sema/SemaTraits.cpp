//===--- SemaTraits.cpp - Trait semantic analysis --------------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "SemaTraits.h"
#include "SemaComptime.h"
#include "SemaDeclarations.h"
#include "SemaImpl.h"
#include "SemaSupport.h"
#include "llvm/Support/Debug.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "Sema"

namespace mulberry {
using llvm::dyn_cast;

auto TraitSema::lookupTrait(std::string_view name) -> const TraitDecl * {
  if (auto *trait = _sema._symbols.lookupTrait(name))
    return trait->decl;

  auto importedName = _sema.canonicalizeImportedName(name);
  if (auto *trait = _sema._symbols.lookupTrait(importedName))
    return trait->decl;

  auto qualifiedName = _sema.qualifyCurrentPackageName(name);
  if (auto *trait = _sema._symbols.lookupTrait(qualifiedName))
    return trait->decl;

  return nullptr;
}

auto TraitSema::resolveConstraints(
    const Node *node, std::vector<ComptimeParam> &parameters)
    -> llvm::LogicalResult {
  for (auto &parameter : parameters) {
    if (!parameter.hasTraitConstraint())
      continue;
    if (parameter.kind != ComptimeParam::Kind::Type)
      return _sema.emitError(node, diag::mismatch_type);

    parameter.trait = lookupTrait(parameter.traitName);
    if (!parameter.trait) {
      auto diagnostic =
          formatNameDiagnostic(diag::undefined_trait, parameter.traitName);
      return _sema.emitError(node, diagnostic);
    }
  }
  return success();
}

auto TraitSema::sema(TraitDecl *node) -> llvm::LogicalResult {
  if (_sema._symbols.lookupTrait(node->name())) {
    auto diagnostic = formatNameDiagnostic(diag::redefinition_trait,
                                           node->name());
    return _sema.emitError(node, diagnostic);
  }
  if (llvm::failed(_sema._symbols.declareTrait(node->name(), node))) {
    auto diagnostic = formatNameDiagnostic(diag::redefinition_trait,
                                           node->name());
    return _sema.emitError(node, diagnostic);
  }

  SemaImpl::PackageScope packageScope(_sema._currentPackageName,
                            packageNameOf(node->name()));
  NameSet methodNames;
  for (auto &method : node->methods()) {
    if (!declareName(methodNames, method->name())) {
      auto diagnostic = formatNameDiagnostic(diag::redefinition_func,
                                             method->name());
      return _sema.emitError(method.get(), diagnostic);
    }

    NameSet parameterNames;
    for (auto &parameter : method->parameters()) {
      auto *type = _sema.checkType(parameter->typeNode(), SemaImpl::UnitPolicy::Reject);
      if (!type)
        return failure();
      parameter->setType(type);
      if (!declareName(parameterNames, parameter->variable()->name()))
        return _sema.emitError(parameter->variable().get(), diag::redefinition_var);
    }

    auto *returnType = _sema.resolveType(method->returnTypeNode());
    if (!returnType)
      return failure();
    method->setReturnType(returnType);
  }
  return success();
}

auto TraitSema::methodFunctionName(const TraitDecl *trait,
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

auto TraitSema::instantiateDefaultMethod(
    const TraitDecl *trait, const TraitMethodDecl *method,
    const Type *targetType, std::string &functionName) -> llvm::LogicalResult {
  functionName = methodFunctionName(trait, targetType, method->name());
  if (_sema.lookupFunction(functionName))
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
  SemaImpl::VariableScope signatureScope(_sema._symbols);
  SemaImpl::PackageScope packageScope(_sema._currentPackageName, packageNameOf(trait->name()));
  if (llvm::failed(DeclarationSema(_sema).semaFunctionSignature(
          function->proto().get())))
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "instantiate default trait method `"
                          << functionName << "`\n");
  _sema._instantiatedFunctions.push_back(std::move(function));
  return success();
}

auto TraitSema::instantiateGenericMethod(
    const ImplDecl *impl, const FunctionDecl *method,
    const TraitMethodDecl *contract, const Type *targetType,
    std::string &functionName) -> llvm::LogicalResult {
  functionName = methodFunctionName(impl->trait(), targetType,
                                         method->proto()->id()->name());
  if (_sema.lookupFunction(functionName))
    return success();

  auto argumentTypeNode = typeToTypeNode(targetType, method->location());
  auto function = instantiateFunctionDecl(
      method, functionName,
      std::vector<TypeSubstitution>{TypeSubstitution{
          impl->comptimeParameters().front().name, argumentTypeNode.get(),
          std::nullopt}});
  function->proto()->setIsMethod(true);
  _sema._instantiatedFunctionPackages[functionName] = impl->packageName();

  SemaImpl::VariableScope signatureScope(_sema._symbols);
  SemaImpl::PackageScope packageScope(_sema._currentPackageName, impl->packageName());
  if (llvm::failed(DeclarationSema(_sema).semaFunctionSignature(
          function->proto().get())))
    return failure();
  auto *signature = _sema.lookupFunction(functionName);
  if (!signature)
    return failure();
  if (!traitMethodSignatureMatches(contract, signature, targetType)) {
    auto diagnostic = formatNameDiagnostic(diag::trait_method_signature,
                                           method->proto()->id()->name());
    return _sema.emitError(method->proto()->id().get(), diagnostic);
  }

  LLVM_DEBUG(llvm::dbgs() << "instantiate conditional trait method `"
                          << functionName << "`\n");
  _sema._instantiatedFunctions.push_back(std::move(function));
  return success();
}

auto TraitSema::sema(ImplDecl *node) -> llvm::LogicalResult {
  SemaImpl::PackageScope packageScope(_sema._currentPackageName, node->packageName());
  auto *trait = lookupTrait(node->traitName());
  if (!trait) {
    auto diagnostic = formatNameDiagnostic(diag::undefined_trait,
                                           node->traitName());
    return _sema.emitError(node, diagnostic);
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
      return _sema.emitError(node, diag::mismatch_type);

    NameSet implementedMethods;
    for (auto &method : node->methods()) {
      auto *prototype = method->proto().get();
      auto methodName = std::string(prototype->id()->name());
      if (contracts.find(methodName) == contracts.end() ||
          !declareName(implementedMethods, methodName) ||
          prototype->isGeneric()) {
        auto diagnostic =
            formatNameDiagnostic(diag::trait_method_signature, methodName);
        return _sema.emitError(prototype->id().get(), diagnostic);
      }
    }

    for (auto &contract : contracts) {
      if (implementedMethods.find(contract.first) != implementedMethods.end() ||
          contract.second->hasDefaultBody())
        continue;
      auto diagnostic =
          formatNameDiagnostic(diag::missing_trait_method, contract.first);
      return _sema.emitError(node, diagnostic);
    }

    if (llvm::failed(_sema._symbols.declareGenericTraitImplementation(node)))
      return _sema.emitError(node, diag::mismatch_type);
    LLVM_DEBUG(llvm::dbgs() << "register conditional trait implementation `"
                            << trait->name() << "` for `"
                            << parameters.front().name << "`\n");
    return success();
  }

  auto *targetType = _sema.checkType(node->targetTypeNode(), SemaImpl::UnitPolicy::Reject);
  if (!targetType)
    return failure();
  node->setTargetType(targetType);

  if (_sema._symbols.lookupTraitImplementation(trait, targetType)) {
    auto diagnostic = formatNameDiagnostic(diag::redefinition_trait_impl,
                                           trait->name());
    return _sema.emitError(node, diagnostic);
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
      return _sema.emitError(prototype->id().get(), diagnostic);
    }

    auto fullName = methodFunctionName(trait, targetType, methodName);
    prototype->id()->setName(fullName);
    prototype->setIsMethod(true);
    _sema._functionPackages[fullName] = std::string(node->packageName());

    if (prototype->isGeneric()) {
      auto diagnostic =
          formatNameDiagnostic(diag::trait_method_signature, methodName);
      return _sema.emitError(prototype->id().get(), diagnostic);
    }
    if (llvm::failed(_sema.sema(method.get())))
      return failure();

    auto *signature = _sema.lookupFunction(fullName);
    if (!signature) {
      auto diagnostic =
          formatNameDiagnostic(diag::trait_method_signature, methodName);
      return _sema.emitError(prototype->id().get(), diagnostic);
    }
    if (!traitMethodSignatureMatches(contract->second, signature, targetType)) {
      auto diagnostic =
          formatNameDiagnostic(diag::trait_method_signature, methodName);
      return _sema.emitError(prototype->id().get(), diagnostic);
    }

    methodFunctionNames.insert({methodName, fullName});
  }

  for (auto &contract : contracts) {
    if (implementedMethods.find(contract.first) != implementedMethods.end())
      continue;

    if (contract.second->hasDefaultBody()) {
      std::string functionName;
      if (llvm::failed(instantiateDefaultMethod(trait, contract.second, targetType,
                                        functionName)))
        return failure();
      methodFunctionNames.insert({contract.first, std::move(functionName)});
      continue;
    }

    auto diagnostic =
        formatNameDiagnostic(diag::missing_trait_method, contract.first);
    return _sema.emitError(node, diagnostic);
  }

  if (llvm::failed(_sema._symbols.declareTraitImplementation(
          trait, targetType, node, std::move(methodFunctionNames)))) {
    auto diagnostic = formatNameDiagnostic(diag::redefinition_trait_impl,
                                           trait->name());
    return _sema.emitError(node, diagnostic);
  }
  LLVM_DEBUG(llvm::dbgs() << "register trait implementation `"
                          << trait->name() << "` for `"
                          << formatType(targetType) << "`\n");
  return success();
}

auto TraitSema::genericImplementationMatches(const ImplDecl *impl,
                                                 const Type *type,
                                                 bool &matches)
    -> llvm::LogicalResult {
  auto &parameters = impl->comptimeParameters();
  auto argumentTypeNode = typeToTypeNode(type, impl->location());
  std::vector<TypeSubstitution> substitutions;
  substitutions.push_back(TypeSubstitution{parameters.front().name,
                                           argumentTypeNode.get(), std::nullopt});
  auto condition =
      substituteExpr(impl->whereCondition(), substitutions);

  SemaImpl::PackageScope packageScope(_sema._currentPackageName, impl->packageName());
  auto result = ComptimeSema(_sema).evaluateComptime(condition.get());
  if (result.kind == ComptimeEvaluation::Kind::Error)
    return failure();
  if (result.kind != ComptimeEvaluation::Kind::Value)
    return _sema.emitError(condition.get(), diag::expected_comptime_value);
  if (result.value->kind() != ComptimeValue::Kind::Bool)
    return _sema.emitError(condition.get(), diag::expected_bool);

  matches = result.value->boolValue();
  LLVM_DEBUG(llvm::dbgs() << "evaluate conditional trait implementation `"
                          << impl->trait()->name() << "` for `"
                          << formatType(type) << "`: "
                          << (matches ? "match" : "no match") << "\n");
  return success();
}

auto TraitSema::findMatchingGenericImplementations(
    const Type *type, const TraitDecl *trait,
    std::vector<const ImplDecl *> &implementations) -> llvm::LogicalResult {
  for (auto *impl : _sema._symbols.genericTraitImplementations()) {
    if (impl->trait() != trait)
      continue;

    bool matches = false;
    if (llvm::failed(genericImplementationMatches(impl, type, matches)))
      return failure();
    if (matches)
      implementations.push_back(impl);
  }
  return success();
}

auto TraitSema::materializeImplementation(
    const Node *diagnosticNode, const Type *type, const TraitDecl *trait,
    bool &matched) -> llvm::LogicalResult {
  if (_sema._symbols.lookupTraitImplementation(trait, type)) {
    matched = true;
    return success();
  }

  std::vector<const ImplDecl *> implementations;
  if (llvm::failed(
          findMatchingGenericImplementations(type, trait, implementations)))
    return failure();
  if (implementations.empty()) {
    matched = false;
    return success();
  }
  if (implementations.size() != 1) {
    auto diagnostic =
        formatTypeTraitDiagnostic(diag::ambiguous_trait_impl, type, trait->name());
    return _sema.emitError(diagnosticNode, diagnostic);
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
      if (llvm::failed(instantiateGenericMethod(implementation, method->second,
                                        contract.second, type, functionName)))
        return failure();
    } else {
      if (!contract.second->hasDefaultBody()) {
        auto diagnostic =
            formatNameDiagnostic(diag::missing_trait_method, contract.first);
        return _sema.emitError(diagnosticNode, diagnostic);
      }
      if (llvm::failed(instantiateDefaultMethod(trait, contract.second, type,
                                        functionName)))
        return failure();
    }
    methodFunctionNames.insert({contract.first, std::move(functionName)});
  }

  if (llvm::failed(_sema._symbols.declareTraitImplementation(
          trait, type, implementation, std::move(methodFunctionNames)))) {
    auto diagnostic = formatNameDiagnostic(diag::redefinition_trait_impl,
                                           trait->name());
    return _sema.emitError(diagnosticNode, diagnostic);
  }
  matched = true;
  LLVM_DEBUG(llvm::dbgs() << "materialize conditional trait implementation `"
                          << trait->name() << "` for `" << formatType(type)
                          << "`\n");
  return success();
}

auto TraitSema::materializeMethod(const Node *diagnosticNode,
                                             const Type *type,
                                             std::string_view methodName,
                                             std::string &functionName)
    -> llvm::LogicalResult {
  std::vector<const TraitDecl *> traits;
  for (auto *impl : _sema._symbols.genericTraitImplementations()) {
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
    if (llvm::failed(materializeImplementation(
            diagnosticNode, type, trait, matched)))
      return failure();
    if (!matched)
      continue;

    auto *implementation = _sema._symbols.lookupTraitImplementation(trait, type);
    auto method = implementation->methodFunctionNames.find(methodName);
    if (method == implementation->methodFunctionNames.end())
      continue;
    functionName = method->second;
    return success();
  }
  return success();
}

auto TraitSema::typeConforms(const Node *diagnosticNode,
                                   const Type *type, const TraitDecl *trait,
                                   bool &conforms) -> llvm::LogicalResult {
  return materializeImplementation(diagnosticNode, type, trait,
                                                conforms);
}

auto TraitSema::checkConstraints(
    const Node *node, const std::vector<ComptimeParam> &parameters,
    const std::vector<InferredComptimeArgument> &arguments)
    -> llvm::LogicalResult {
  for (size_t index = 0; index < parameters.size(); ++index) {
    auto &parameter = parameters[index];
    if (!parameter.trait)
      continue;
    auto *type = arguments[index].type;
    bool conforms = false;
    if (type && llvm::failed(
                    typeConforms(node, type, parameter.trait, conforms)))
      return failure();
    if (conforms)
      continue;

    auto diagnostic = formatTypeTraitDiagnostic(
        diag::trait_constraint, type, parameter.trait->name());
    return _sema.emitError(node, diagnostic);
  }
  return success();
}
} // namespace mulberry
