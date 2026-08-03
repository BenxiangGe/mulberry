//===--- Symbols.h - Symbol Table -------------------------------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_SYMBOLS_H
#define MULBERRY_SYMBOLS_H

#include "mulberry/AST/Type.h"
#include "mulberry/Basic/ScopeStack.h"
#include "mulberry/Basic/Types.h"
#include "llvm/Support/LogicalResult.h"
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
};

struct ComptimeTypeAliasSymbol {
  std::string packageName;
  std::vector<ComptimeParam> parameters;
  const TypeNode *bodyTypeNode = nullptr;
};

struct GenericFunctionSymbol {
  const FunctionDecl *decl = nullptr;
};

struct DataDeclSymbol {
  const DataDecl *decl = nullptr;
};

struct DataConstructorSymbol {
  const DataDecl *decl = nullptr;
  unsigned index = 0;
};

struct TraitSymbol {
  const TraitDecl *decl = nullptr;
};

struct TraitImplementationSymbol {
  const ImplDecl *decl = nullptr;
  std::map<std::string, std::string, std::less<>> methodFunctionNames;
};

template <typename T>
using NameMap = std::map<std::string, T, std::less<>>;

class Symbols {
public:
  auto declareFunction(std::string_view name, const FunctionType *type,
                       bool isExtern)
      -> llvm::LogicalResult {
    return declareSymbol(_functionsByName, name,
                         FunctionSymbol{type, isExtern});
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
                              const FunctionDecl *decl) -> llvm::LogicalResult {
    return declareSymbol(_genericFunctionsByName, name,
                         GenericFunctionSymbol{decl});
  }

  auto lookupGenericFunction(std::string_view name)
      -> const GenericFunctionSymbol * {
    auto symbol = _genericFunctionsByName.find(name);
    if (symbol == _genericFunctionsByName.end())
      return nullptr;
    return &symbol->second;
  }

  auto declareDataDecl(std::string_view name, const DataDecl *decl)
      -> llvm::LogicalResult {
    return declareSymbol(_dataDeclsByName, name, DataDeclSymbol{decl});
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

  auto declareTrait(std::string_view name, const TraitDecl *decl)
      -> llvm::LogicalResult {
    return declareSymbol(_traitsByName, name, TraitSymbol{decl});
  }

  auto lookupTrait(std::string_view name) -> const TraitSymbol * {
    auto symbol = _traitsByName.find(name);
    if (symbol == _traitsByName.end())
      return nullptr;
    return &symbol->second;
  }

  auto declareTraitImplementation(
      const TraitDecl *trait, const Type *type, const ImplDecl *decl,
      std::map<std::string, std::string, std::less<>> methodFunctionNames)
      -> llvm::LogicalResult {
    auto key = std::make_pair(trait, type);
    if (_traitImplementations.find(key) != _traitImplementations.end())
      return failure();
    _traitImplementations.insert(std::make_pair(
        key, TraitImplementationSymbol{decl, std::move(methodFunctionNames)}));
    return success();
  }

  auto lookupTraitImplementation(const TraitDecl *trait, const Type *type)
      -> const TraitImplementationSymbol * {
    auto symbol = _traitImplementations.find(std::make_pair(trait, type));
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
      if (implementation.first.second != type)
        continue;
      auto method = implementation.second.methodFunctionNames.find(methodName);
      if (method != implementation.second.methodFunctionNames.end())
        return &method->second;
    }
    return nullptr;
  }

  auto declareType(std::string_view name, const Type *type) -> llvm::LogicalResult {
    return declareSymbol(_typesByName, name, type);
  }

  auto lookupType(std::string_view name) -> const Type * {
    auto type = _typesByName.find(name);
    if (type == _typesByName.end())
      return nullptr;
    return type->second;
  }

  auto declareComptimeTypeAlias(std::string_view name,
                                std::string_view packageName,
                                std::vector<ComptimeParam> parameters,
                                const TypeNode *bodyTypeNode)
      -> llvm::LogicalResult {
    return declareSymbol(_comptimeTypeAliasesByName, name,
                         ComptimeTypeAliasSymbol{
                             std::string(packageName), std::move(parameters),
                             bodyTypeNode});
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
  std::map<std::pair<const TraitDecl *, const Type *>,
           TraitImplementationSymbol>
      _traitImplementations;
  std::vector<const ImplDecl *> _genericTraitImplementations;
  NameMap<const Type *> _typesByName;
  NameMap<ComptimeTypeAliasSymbol> _comptimeTypeAliasesByName;
  ScopeStack<NameMap<VariableSymbol>> _variableScopes;
};

} // end namespace mulberry

#endif // MULBERRY_SYMBOLS_H
