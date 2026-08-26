//===--- Symbols.h - Symbol Table -------------------------------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_SYMBOLS_H
#define MULBERRY_SYMBOLS_H

#include "mulberry/AST/Decl.h"
#include "mulberry/Basic/ScopeStack.h"
#include "mulberry/Basic/Types.h"
#include "llvm/Support/LogicalResult.h"
#include <algorithm>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mulberry {
class DataDecl;
class FunctionDecl;
class ImplDecl;
class TraitDecl;
class TypeNode;
using llvm::failure;
using llvm::success;

struct VariableSymbol {
  const Type *type = nullptr;
  bool isConstBinding = false;
  bool canMutateObject = true;
  std::optional<ComptimeValue> comptimeValue;
  bool isComptimeOnly = false;
  std::optional<uint64_t> replSlot;
};

struct FunctionSymbol {
  const FunctionType *type = nullptr;
  bool isExtern = false;
  std::string packageName;
  Visibility visibility = Visibility::Private;
};

struct ComptimeTypeAliasSymbol {
  std::string packageName;
  std::vector<ComptimeParam> parameters;
  const TypeNode *bodyTypeNode = nullptr;
  Visibility visibility = Visibility::Private;
};

struct GenericFunctionSymbol {
  const FunctionDecl *decl = nullptr;
  std::string packageName;
  Visibility visibility = Visibility::Private;
};

struct DataDeclSymbol {
  const DataDecl *decl = nullptr;
  std::string packageName;
  Visibility visibility = Visibility::Private;
};

struct DataConstructorSymbol {
  const DataDecl *decl = nullptr;
  unsigned index = 0;
};

struct TraitSymbol {
  const TraitDecl *decl = nullptr;
  std::string packageName;
  Visibility visibility = Visibility::Private;
};

struct TypeSymbol {
  const Type *type = nullptr;
  std::string packageName;
  Visibility visibility = Visibility::Private;
};

struct TraitImplementationSymbol {
  const ImplDecl *decl = nullptr;
  std::map<std::string, std::string, std::less<>> methodFunctionNames;
};

struct TraitImplementationKey {
  const TraitDecl *trait = nullptr;
  std::vector<const Type *> traitArguments;
  const Type *targetType = nullptr;

  auto operator<(const TraitImplementationKey &other) const -> bool {
    if (trait != other.trait)
      return trait < other.trait;
    if (traitArguments != other.traitArguments)
      return traitArguments < other.traitArguments;
    return targetType < other.targetType;
  }
};

template <typename T>
using NameMap = std::map<std::string, T, std::less<>>;

class Symbols {
public:
  auto declareFunction(std::string_view name, const FunctionType *type,
                       bool isExtern, std::string_view packageName,
                       Visibility visibility)
      -> llvm::LogicalResult {
    return declareSymbol(_functionsByName, name,
                         FunctionSymbol{type, isExtern, std::string(packageName),
                                        visibility});
  }

  auto lookupFunction(std::string_view name) -> const FunctionSymbol * {
    auto symbol = _functionsByName.find(name);
    if (symbol == _functionsByName.end())
      return nullptr;
    return &symbol->second;
  }

  auto functions() const -> const NameMap<FunctionSymbol> & {
    return _functionsByName;
  }

  auto genericFunctions() const -> const NameMap<GenericFunctionSymbol> & {
    return _genericFunctionsByName;
  }

  auto declareGenericFunction(std::string_view name,
                              const FunctionDecl *decl,
                              std::string_view packageName,
                              Visibility visibility) -> llvm::LogicalResult {
    return declareSymbol(_genericFunctionsByName, name,
                         GenericFunctionSymbol{decl, std::string(packageName),
                                               visibility});
  }

  auto lookupGenericFunction(std::string_view name)
      -> const GenericFunctionSymbol * {
    auto symbol = _genericFunctionsByName.find(name);
    if (symbol == _genericFunctionsByName.end())
      return nullptr;
    return &symbol->second;
  }

  auto declareDataDecl(std::string_view name, const DataDecl *decl,
                       std::string_view packageName, Visibility visibility)
      -> llvm::LogicalResult {
    return declareSymbol(_dataDeclsByName, name,
                         DataDeclSymbol{decl, std::string(packageName),
                                        visibility});
  }

  auto lookupDataDecl(std::string_view name) -> const DataDeclSymbol * {
    auto symbol = _dataDeclsByName.find(name);
    if (symbol == _dataDeclsByName.end())
      return nullptr;
    return &symbol->second;
  }

  auto declareDataConstructor(std::string_view name, const DataDecl *decl,
                              unsigned index) -> llvm::LogicalResult {
    return declareSymbol(_dataConstructorsByName, name,
                         DataConstructorSymbol{decl, index});
  }

  auto lookupDataConstructor(std::string_view name)
      -> const DataConstructorSymbol * {
    auto symbol = _dataConstructorsByName.find(name);
    if (symbol == _dataConstructorsByName.end())
      return nullptr;
    return &symbol->second;
  }

  auto declareTrait(std::string_view name, const TraitDecl *decl,
                    std::string_view packageName, Visibility visibility)
      -> llvm::LogicalResult {
    return declareSymbol(_traitsByName, name,
                         TraitSymbol{decl, std::string(packageName), visibility});
  }

  auto lookupTrait(std::string_view name) -> const TraitSymbol * {
    auto symbol = _traitsByName.find(name);
    if (symbol == _traitsByName.end())
      return nullptr;
    return &symbol->second;
  }

  auto declareTraitImplementation(
      const TraitDecl *trait, std::vector<const Type *> traitArguments,
      const Type *targetType, const ImplDecl *decl,
      std::map<std::string, std::string, std::less<>> methodFunctionNames)
      -> llvm::LogicalResult {
    auto key = TraitImplementationKey{trait, std::move(traitArguments),
                                      targetType};
    if (_traitImplementations.find(key) != _traitImplementations.end())
      return failure();
    _traitImplementations.insert(std::make_pair(
        key, TraitImplementationSymbol{decl, std::move(methodFunctionNames)}));
    return success();
  }

  auto lookupTraitImplementation(const TraitDecl *trait,
                                 const std::vector<const Type *> &traitArguments,
                                 const Type *targetType)
      -> const TraitImplementationSymbol * {
    auto symbol = _traitImplementations.find(
        TraitImplementationKey{trait, traitArguments, targetType});
    if (symbol == _traitImplementations.end())
      return nullptr;
    return &symbol->second;
  }

  auto declareGenericTraitImplementation(const ImplDecl *decl)
      -> llvm::LogicalResult {
    _genericTraitImplementations.push_back(decl);
    return success();
  }

  auto genericTraitImplementations() const
      -> const std::vector<const ImplDecl *> & {
    return _genericTraitImplementations;
  }

  auto lookupTraitMethod(const Type *type, std::string_view methodName)
      -> const std::string * {
    for (auto &implementation : _traitImplementations) {
      if (implementation.first.targetType != type)
        continue;
      // Receiver lookup has no Trait argument context. Generic Trait methods
      // require the type-qualified lookup introduced by ER2.3.
      if (implementation.first.trait->isGeneric())
        continue;
      auto method = implementation.second.methodFunctionNames.find(methodName);
      if (method != implementation.second.methodFunctionNames.end())
        return &method->second;
    }
    return nullptr;
  }

  auto lookupStaticTraitMethods(const Type *targetType,
                                std::string_view methodName) const
      -> std::vector<const TraitImplementationSymbol *> {
    std::vector<const TraitImplementationSymbol *> matches;
    for (auto &implementation : _traitImplementations) {
      if (implementation.first.targetType != targetType)
        continue;
      auto *trait = implementation.first.trait;
      auto contract = std::find_if(
          trait->methods().begin(), trait->methods().end(),
          [&](const auto &method) { return method->name() == methodName; });
      if (contract == trait->methods().end() || (*contract)->hasReceiver())
        continue;
      if (implementation.second.methodFunctionNames.find(methodName) !=
          implementation.second.methodFunctionNames.end())
        matches.push_back(&implementation.second);
    }
    return matches;
  }

  auto declareType(std::string_view name, const Type *type,
                   std::string_view packageName, Visibility visibility)
      -> llvm::LogicalResult {
    return declareSymbol(_typesByName, name,
                         TypeSymbol{type, std::string(packageName), visibility});
  }

  auto lookupType(std::string_view name) -> const Type * {
    auto type = _typesByName.find(name);
    if (type == _typesByName.end())
      return nullptr;
    return type->second.type;
  }

  auto lookupTypeSymbol(std::string_view name) -> const TypeSymbol * {
    auto type = _typesByName.find(name);
    if (type == _typesByName.end())
      return nullptr;
    return &type->second;
  }

  auto declareComptimeTypeAlias(std::string_view name,
                                std::string_view packageName,
                                std::vector<ComptimeParam> parameters,
                                const TypeNode *bodyTypeNode,
                                Visibility visibility)
      -> llvm::LogicalResult {
    return declareSymbol(_comptimeTypeAliasesByName, name,
                         ComptimeTypeAliasSymbol{
                             std::string(packageName), std::move(parameters),
                             bodyTypeNode, visibility});
  }

  auto lookupComptimeTypeAlias(std::string_view name)
      -> const ComptimeTypeAliasSymbol * {
    auto alias = _comptimeTypeAliasesByName.find(name);
    if (alias == _comptimeTypeAliasesByName.end())
      return nullptr;
    return &alias->second;
  }

  auto resetVariables() {
    _variableScopes.reset();
    enterVariableScope();
  }

  auto variableScopeDepth() const -> size_t { return _variableScopes.size(); }

  auto enterVariableScope() -> void { _variableScopes.enterScope(); }

  auto leaveVariableScope() -> void {
    _variableScopes.leaveScope();
  }

  auto declareVariable(std::string_view name, const Type *type,
                       bool isConstBinding = false,
                       bool canMutateObject = true,
                       std::optional<ComptimeValue> comptimeValue = std::nullopt,
                       bool isComptimeOnly = false,
                       std::optional<uint64_t> replSlot = std::nullopt)
      -> llvm::LogicalResult {
    if (_variableScopes.empty())
      enterVariableScope();

    if (llvm::failed(declareSymbol(_variableScopes.currentScope(), name,
                      VariableSymbol{type, isConstBinding, canMutateObject,
                                     std::move(comptimeValue),
                                     isComptimeOnly, replSlot})))
      return failure();
    return success();
  }

  auto lookupVariable(std::string_view name) const -> const VariableSymbol * {
    return _variableScopes.lookup(name);
  }

  auto lookupCurrentVariable(std::string_view name) -> const VariableSymbol * {
    return _variableScopes.lookupCurrent(name);
  }

  auto currentVariables() const -> const NameMap<VariableSymbol> & {
    return _variableScopes.currentScope();
  }

  auto completionNames() const -> std::vector<std::string> {
    std::vector<std::string> result;
    auto appendNames = [&result](const auto &symbols) {
      for (const auto &[name, symbol] : symbols) {
        (void)symbol;
        if (name.find('.') == std::string::npos)
          result.push_back(name);
      }
    };

    appendNames(_functionsByName);
    appendNames(_genericFunctionsByName);
    appendNames(_dataDeclsByName);
    appendNames(_dataConstructorsByName);
    appendNames(_traitsByName);
    appendNames(_typesByName);
    appendNames(_comptimeTypeAliasesByName);
    if (!_variableScopes.empty())
      appendNames(_variableScopes.currentScope());
    return result;
  }

private:
  template <typename T>
  auto declareSymbol(NameMap<T> &symbols, std::string_view name, T value)
      -> llvm::LogicalResult {
    if (symbols.find(name) != symbols.end())
      return failure();
    symbols.insert(std::make_pair(std::string(name), std::move(value)));
    return success();
  }

  NameMap<FunctionSymbol> _functionsByName;
  NameMap<GenericFunctionSymbol> _genericFunctionsByName;
  NameMap<DataDeclSymbol> _dataDeclsByName;
  NameMap<DataConstructorSymbol> _dataConstructorsByName;
  NameMap<TraitSymbol> _traitsByName;
  std::map<TraitImplementationKey, TraitImplementationSymbol>
      _traitImplementations;
  std::vector<const ImplDecl *> _genericTraitImplementations;
  NameMap<TypeSymbol> _typesByName;
  NameMap<ComptimeTypeAliasSymbol> _comptimeTypeAliasesByName;
  ScopeStack<NameMap<VariableSymbol>> _variableScopes;
};

} // end namespace mulberry

#endif // MULBERRY_SYMBOLS_H
