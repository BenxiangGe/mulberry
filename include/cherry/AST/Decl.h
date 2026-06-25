//===--- Decl.h - Cherry Language Declaration ASTs --------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_DECL_H
#define CHERRY_DECL_H

#include "cherry/AST/Name.h"
#include "cherry/AST/Node.h"
#include "cherry/AST/Type.h"
#include <string>
#include <utility>
#include <vector>

namespace cherry {

class BlockExpr;
class Expr;
class Type;
class VariableExpr;
class VariableStat;

// _____________________________________________________________________________
// Declaration

class Decl : public Node {
public:
  enum DeclarationKind {
    Decl_Import,
    Decl_Function,
    Decl_Struct,
    Decl_ComptimeTypeAlias,
  };

  explicit Decl(DeclarationKind kind, llvm::SMLoc location)
      : Node{location}, _kind{kind} {};

  auto getKind() const -> DeclarationKind { return _kind; }

private:
  const DeclarationKind _kind;
};

// _____________________________________________________________________________
// Import declaration

class ImportDecl final : public Decl {
public:
  explicit ImportDecl(llvm::SMLoc location, std::string_view moduleName)
      : Decl{Decl_Import, location}, _moduleName{moduleName} {}

  static auto classof(const Decl *node) -> bool {
    return node->getKind() == Decl_Import;
  }

  auto moduleName() const -> std::string_view { return _moduleName; }

private:
  std::string _moduleName;
};

// _____________________________________________________________________________
// Function declaration

class Prototype final : public Node {
public:
  explicit Prototype(llvm::SMLoc location, std::unique_ptr<FunctionName> id,
                     VectorUniquePtr<VariableStat> parameters,
                     std::unique_ptr<TypeNode> returnTypeNode,
                     std::vector<ComptimeParam> comptimeParameters = {})
      : Node{location},
        _id(std::move(id)), _parameters{std::move(parameters)},
        _returnTypeNode{std::move(returnTypeNode)},
        _comptimeParameters(std::move(comptimeParameters)) {};

  auto id() const -> const std::unique_ptr<FunctionName> & { return _id; }

  auto comptimeParameters() const -> const std::vector<ComptimeParam> & {
    return _comptimeParameters;
  }

  auto setComptimeParameters(std::vector<ComptimeParam> parameters) -> void {
    _comptimeParameters = std::move(parameters);
  }

  auto isGeneric() const -> bool { return !_comptimeParameters.empty(); }

  auto parameters() const -> const VectorUniquePtr<VariableStat> & {
    return _parameters;
  }

  auto returnTypeNode() const -> const TypeNode * {
    return _returnTypeNode.get();
  }

  auto setType(const Type *type) -> void { _type = type; }

  auto type() const -> const Type * { return _type; }

private:
  std::unique_ptr<FunctionName> _id;
  VectorUniquePtr<VariableStat> _parameters;
  std::unique_ptr<TypeNode> _returnTypeNode;
  std::vector<ComptimeParam> _comptimeParameters;
  const Type *_type = nullptr;
};

class FunctionDecl final : public Decl {
public:
  explicit FunctionDecl(llvm::SMLoc location, std::unique_ptr<Prototype> proto,
                        std::unique_ptr<BlockExpr> body, bool isExtern = false)
      : Decl{Decl_Function, location}, _proto(std::move(proto)),
        _body(std::move(body)), _isExtern(isExtern){};

  static auto classof(const Decl *node) -> bool {
    return node->getKind() == Decl_Function;
  }

  auto proto() const -> const std::unique_ptr<Prototype> & { return _proto; }

  auto body() const -> const std::unique_ptr<BlockExpr> & { return _body; }

  auto isExtern() const -> bool { return _isExtern; }

private:
  std::unique_ptr<Prototype> _proto;
  std::unique_ptr<BlockExpr> _body;
  bool _isExtern = false;
};

// _____________________________________________________________________________
// Struct declaration

class StructDecl final : public Decl {
public:
  explicit StructDecl(llvm::SMLoc location, std::unique_ptr<StructName> id,
                      VectorUniquePtr<VariableStat> variables,
                      VectorUniquePtr<FunctionDecl> methods)
      : Decl{Decl_Struct, location}, _id(std::move(id)),
        _variables(std::move(variables)), _methods(std::move(methods)){};

  static auto classof(const Decl *node) -> bool {
    return node->getKind() == Decl_Struct;
  }

  auto id() const -> const std::unique_ptr<StructName> & { return _id; }

  auto variables() const -> const VectorUniquePtr<VariableStat> & {
    return _variables;
  }

  auto methods() const -> const VectorUniquePtr<FunctionDecl> & {
    return _methods;
  }

private:
  std::unique_ptr<StructName> _id;
  VectorUniquePtr<VariableStat> _variables;
  VectorUniquePtr<FunctionDecl> _methods;

public:
  auto begin() const -> decltype(_variables.begin()) {
    return _variables.begin();
  }
  auto end() const -> decltype(_variables.end()) { return _variables.end(); }
};

// Type-level comptime alias. It can be generic, e.g.
// `comptime List<T> = struct {...};`, or zero-parameter, e.g.
// `comptime String = SomeStruct;`.
class ComptimeTypeAliasDecl final : public Decl {
public:
  ComptimeTypeAliasDecl(llvm::SMLoc location, std::string_view name,
                        std::vector<ComptimeParam> parameters,
                        std::unique_ptr<TypeNode> bodyTypeNode)
      : Decl{Decl_ComptimeTypeAlias, location}, _name(name),
        _parameters(std::move(parameters)),
        _bodyTypeNode(std::move(bodyTypeNode)) {}

  static auto classof(const Decl *node) -> bool {
    return node->getKind() == Decl_ComptimeTypeAlias;
  }

  auto name() const -> std::string_view { return _name; }

  auto isGeneric() const -> bool { return !_parameters.empty(); }

  auto parameters() const -> const std::vector<ComptimeParam> & {
    return _parameters;
  }

  auto bodyTypeNode() const -> const TypeNode * {
    return _bodyTypeNode.get();
  }

  auto methods() const -> const VectorUniquePtr<FunctionDecl> & {
    return _methods;
  }

  auto setMethods(VectorUniquePtr<FunctionDecl> methods) -> void {
    _methods = std::move(methods);
  }

private:
  std::string _name;
  std::vector<ComptimeParam> _parameters;
  std::unique_ptr<TypeNode> _bodyTypeNode;
  VectorUniquePtr<FunctionDecl> _methods;
};

} // end namespace cherry

#endif // CHERRY_DECL_H
