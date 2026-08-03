//===--- Module.h - Mulberry Language Module AST ------------------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_MODULE_H
#define MULBERRY_MODULE_H

#include "mulberry/AST/Node.h"
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace mulberry {
class Decl;
class FunctionType;
class Stat;
class Type;

struct ReplVariableBinding {
  std::string name;
  const Type *type = nullptr;
  uint64_t slot = 0;
  bool isConstBinding = false;
  bool canMutateObject = true;
};

struct ReplFunctionBinding {
  std::string name;
  const FunctionType *type = nullptr;
  bool isExtern = false;
};

class Module : public Node {
public:
  explicit Module(llvm::SMLoc location, VectorUniquePtr<Decl> declarations)
      : Node{location}, _declarations{std::move(declarations)} {};

  auto packageName() const -> std::string_view { return _packageName; }

  auto setPackageName(std::string_view packageName) -> void {
    _packageName = packageName;
  }

  auto declarations() const -> const VectorUniquePtr<Decl> & {
    return _declarations;
  }

  auto takeDeclarations() -> VectorUniquePtr<Decl> {
    return std::move(_declarations);
  }

  auto setDeclarations(VectorUniquePtr<Decl> declarations) -> void {
    _declarations = std::move(declarations);
  }

  auto statements() const -> const VectorUniquePtr<Stat> & {
    return _statements;
  }

  auto setStatements(VectorUniquePtr<Stat> statements) -> void {
    _statements = std::move(statements);
  }

  auto isRepl() const -> bool { return !_replFunctionName.empty(); }

  auto setReplFunctionName(std::string_view functionName) -> void {
    _replFunctionName = functionName;
  }

  auto replFunctionName() const -> std::string_view {
    return _replFunctionName;
  }

  auto replVariables() const -> const std::vector<ReplVariableBinding> & {
    return _replVariables;
  }

  auto setReplVariables(std::vector<ReplVariableBinding> variables) -> void {
    _replVariables = std::move(variables);
  }

  auto replFunctions() const -> const std::vector<ReplFunctionBinding> & {
    return _replFunctions;
  }

  auto setReplFunctions(std::vector<ReplFunctionBinding> functions) -> void {
    _replFunctions = std::move(functions);
  }

private:
  std::string _packageName;
  VectorUniquePtr<Decl> _declarations;
  VectorUniquePtr<Stat> _statements;
  std::string _replFunctionName;
  std::vector<ReplVariableBinding> _replVariables;
  std::vector<ReplFunctionBinding> _replFunctions;

public:
  auto begin() const -> decltype(_declarations.begin()) {
    return _declarations.begin();
  }
  auto end() const -> decltype(_declarations.end()) {
    return _declarations.end();
  }
};

} // end namespace mulberry

#endif // MULBERRY_MODULE_H
