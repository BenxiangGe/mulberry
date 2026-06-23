//===--- Symbols.h - Symbol Table -------------------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_SYMBOLS_H
#define CHERRY_SYMBOLS_H

#include "cherry/AST/Type.h"
#include "cherry/Basic/CherryResult.h"
#include "cherry/Basic/ScopeStack.h"
#include "cherry/Basic/Types.h"
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cherry {
class FunctionDecl;
class TypeNode;
using llvm::failure;
using llvm::success;

struct VariableSymbol {
  const Type *type = nullptr;
  bool isConst = false;
};

struct FunctionSymbol {
  std::vector<const Type *> parameterTypes;
  const Type *returnType = nullptr;
};

struct ComptimeTypeAliasSymbol {
  std::string packageName;
  std::vector<ComptimeParam> parameters;
  const TypeNode *bodyTypeNode = nullptr;

  auto parameterName() const -> std::string_view {
    if (parameters.empty())
      return {};
    return parameters.front().name;
  }
};

struct GenericFunctionSymbol {
  const FunctionDecl *decl = nullptr;
};

template <typename T>
using NameMap = std::map<std::string, T, std::less<>>;

class Symbols {
public:
  auto declareFunction(std::string_view name,
                       std::vector<const Type *> parameterTypes,
                       const Type *returnType)
      -> CherryResult {
    return declareSymbol(
        _functionsByName, name,
        FunctionSymbol{std::move(parameterTypes), returnType});
  }

  auto lookupFunction(std::string_view name) -> const FunctionSymbol * {
    auto symbol = _functionsByName.find(name);
    if (symbol == _functionsByName.end())
      return nullptr;
    return &symbol->second;
  }

  auto declareGenericFunction(std::string_view name,
                              const FunctionDecl *decl) -> CherryResult {
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

  auto declareType(std::string_view name, const Type *type) -> CherryResult {
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
                                std::string_view parameterName,
                                const TypeNode *bodyTypeNode)
      -> CherryResult {
    std::vector<ComptimeParam> parameters;
    if (!parameterName.empty())
      parameters.push_back(ComptimeParam(parameterName));
    return declareComptimeTypeAlias(name, packageName, std::move(parameters),
                                    bodyTypeNode);
  }

  auto declareComptimeTypeAlias(std::string_view name,
                                std::string_view packageName,
                                std::vector<ComptimeParam> parameters,
                                const TypeNode *bodyTypeNode)
      -> CherryResult {
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

  auto enterVariableScope() -> void { _variableScopes.enterScope(); }

  auto leaveVariableScope() -> void {
    _variableScopes.leaveScope();
  }

  auto declareVariable(std::string_view name, const Type *type,
                       bool isConst = false)
      -> CherryResult {
    if (_variableScopes.empty())
      enterVariableScope();

    if (declareSymbol(_variableScopes.currentScope(), name,
                      VariableSymbol{type, isConst}))
      return failure();
    return success();
  }

  auto lookupVariable(std::string_view name) -> const VariableSymbol * {
    return _variableScopes.lookup(name);
  }

private:
  template <typename T>
  auto declareSymbol(NameMap<T> &symbols, std::string_view name, T value)
      -> CherryResult {
    if (symbols.find(name) != symbols.end())
      return failure();
    symbols.insert(std::make_pair(std::string(name), std::move(value)));
    return success();
  }

  NameMap<FunctionSymbol> _functionsByName;
  NameMap<GenericFunctionSymbol> _genericFunctionsByName;
  NameMap<const Type *> _typesByName;
  NameMap<ComptimeTypeAliasSymbol> _comptimeTypeAliasesByName;
  ScopeStack<NameMap<VariableSymbol>> _variableScopes;
};

} // end namespace cherry

#endif // CHERRY_SYMBOLS_H
