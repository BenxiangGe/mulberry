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
    Decl_Function,
    Decl_Struct,
  };

  explicit Decl(DeclarationKind kind, llvm::SMLoc location)
      : Node{location}, _kind{kind} {};

  auto getKind() const -> DeclarationKind { return _kind; }

private:
  const DeclarationKind _kind;
};

// _____________________________________________________________________________
// Function declaration

class Prototype final : public Node {
public:
  explicit Prototype(llvm::SMLoc location, std::unique_ptr<FunctionName> id,
                     VectorUniquePtr<VariableStat> parameters,
                     std::unique_ptr<TypeNode> returnTypeNode)
      : Node{location},
        _id(std::move(id)), _parameters{std::move(parameters)},
        _returnTypeNode{std::move(returnTypeNode)} {};

  auto id() const -> const std::unique_ptr<FunctionName> & { return _id; }

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
  const Type *_type = nullptr;
};

class FunctionDecl final : public Decl {
public:
  explicit FunctionDecl(llvm::SMLoc location, std::unique_ptr<Prototype> proto,
                        std::unique_ptr<BlockExpr> body)
      : Decl{Decl_Function, location}, _proto(std::move(proto)),
        _body(std::move(body)){};

  static auto classof(const Decl *node) -> bool {
    return node->getKind() == Decl_Function;
  }

  auto proto() const -> const std::unique_ptr<Prototype> & { return _proto; }

  auto body() const -> const std::unique_ptr<BlockExpr> & { return _body; }

private:
  std::unique_ptr<Prototype> _proto;
  std::unique_ptr<BlockExpr> _body;
};

// _____________________________________________________________________________
// Struct declaration

class StructDecl final : public Decl {
public:
  explicit StructDecl(llvm::SMLoc location, std::unique_ptr<StructName> id,
                      VectorUniquePtr<VariableStat> variables)
      : Decl{Decl_Struct, location}, _id(std::move(id)),
        _variables(std::move(variables)){};

  static auto classof(const Decl *node) -> bool {
    return node->getKind() == Decl_Struct;
  }

  auto id() const -> const std::unique_ptr<StructName> & { return _id; }

  auto variables() const -> const VectorUniquePtr<VariableStat> & {
    return _variables;
  }

private:
  std::unique_ptr<StructName> _id;
  VectorUniquePtr<VariableStat> _variables;

public:
  auto begin() const -> decltype(_variables.begin()) {
    return _variables.begin();
  }
  auto end() const -> decltype(_variables.end()) { return _variables.end(); }
};

} // end namespace cherry

#endif // CHERRY_DECL_H
