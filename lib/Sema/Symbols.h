//===--- Symbols.h - Symbol Table -------------------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_SYMBOLS_H
#define CHERRY_SYMBOLS_H

#include "cherry/Basic/CherryResult.h"
#include "cherry/Basic/Types.h"
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cherry {
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

  auto declareType(std::string_view name, const Type *type) -> CherryResult {
    return declareSymbol(_typesByName, name, type);
  }

  auto lookupType(std::string_view name) -> const Type * {
    auto type = _typesByName.find(name);
    if (type == _typesByName.end())
      return nullptr;
    return type->second;
  }

  auto resetVariables() { _variablesByName = {}; }

  auto declareVariable(std::string_view name, const Type *type,
                       bool isConst = false)
      -> CherryResult {
    return declareSymbol(_variablesByName, name, VariableSymbol{type, isConst});
  }

  auto lookupVariable(std::string_view name) -> const VariableSymbol * {
    auto symbol = _variablesByName.find(name);
    if (symbol == _variablesByName.end())
      return nullptr;
    return &symbol->second;
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
  NameMap<const Type *> _typesByName;
  NameMap<VariableSymbol> _variablesByName;
};

} // end namespace cherry

#endif // CHERRY_SYMBOLS_H
