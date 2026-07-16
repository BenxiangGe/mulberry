//===--- Decl.h - Mulberry Language Declaration ASTs --------------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_DECL_H
#define MULBERRY_DECL_H

#include "mulberry/AST/Name.h"
#include "mulberry/AST/Node.h"
#include "mulberry/AST/Type.h"
#include <string>
#include <utility>
#include <vector>

namespace mulberry {

class BlockExpr;
class Expr;
class Type;
class VariableExpr;

// _____________________________________________________________________________
// Declaration

class Decl : public Node {
public:
  enum DeclarationKind {
    Decl_Import,
    Decl_Function,
    Decl_Struct,
    Decl_Data,
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
                     VectorUniquePtr<ParameterDecl> parameters,
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

  auto setIsMethod(bool isMethod) -> void { _isMethod = isMethod; }

  auto isMethod() const -> bool { return _isMethod; }

  auto parameters() const -> const VectorUniquePtr<ParameterDecl> & {
    return _parameters;
  }

  auto returnTypeNode() const -> const TypeNode * {
    return _returnTypeNode.get();
  }

  auto setType(const Type *type) -> void { _type = type; }

  auto type() const -> const Type * { return _type; }

private:
  std::unique_ptr<FunctionName> _id;
  VectorUniquePtr<ParameterDecl> _parameters;
  std::unique_ptr<TypeNode> _returnTypeNode;
  std::vector<ComptimeParam> _comptimeParameters;
  const Type *_type = nullptr;
  bool _isMethod = false;
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
                      VectorUniquePtr<FieldDecl> fields,
                      VectorUniquePtr<FunctionDecl> methods)
      : Decl{Decl_Struct, location}, _id(std::move(id)),
        _fields(std::move(fields)), _methods(std::move(methods)){};

  static auto classof(const Decl *node) -> bool {
    return node->getKind() == Decl_Struct;
  }

  auto id() const -> const std::unique_ptr<StructName> & { return _id; }

  auto fields() const -> const VectorUniquePtr<FieldDecl> & {
    return _fields;
  }

  auto methods() const -> const VectorUniquePtr<FunctionDecl> & {
    return _methods;
  }

private:
  std::unique_ptr<StructName> _id;
  VectorUniquePtr<FieldDecl> _fields;
  VectorUniquePtr<FunctionDecl> _methods;

public:
  auto begin() const -> decltype(_fields.begin()) {
    return _fields.begin();
  }
  auto end() const -> decltype(_fields.end()) { return _fields.end(); }
};

// _____________________________________________________________________________
// Data declaration

class DataConstructorDecl final : public Node {
public:
  DataConstructorDecl(llvm::SMLoc location, std::string_view name,
                      VectorUniquePtr<TypeNode> payloadTypes)
      : Node(location), _name(name),
        _payloadTypes(std::move(payloadTypes)) {}

  auto name() const -> std::string_view { return _name; }

  auto payloadTypes() const -> const VectorUniquePtr<TypeNode> & {
    return _payloadTypes;
  }

private:
  std::string _name;
  VectorUniquePtr<TypeNode> _payloadTypes;
};

class DataDecl final : public Decl {
public:
  DataDecl(llvm::SMLoc location, std::string_view name,
           std::vector<ComptimeParam> parameters,
           VectorUniquePtr<DataConstructorDecl> constructors)
      : Decl(Decl_Data, location), _name(name),
        _parameters(std::move(parameters)),
        _constructors(std::move(constructors)) {}

  static auto classof(const Decl *node) -> bool {
    return node->getKind() == Decl_Data;
  }

  auto name() const -> std::string_view { return _name; }

  auto parameters() const -> const std::vector<ComptimeParam> & {
    return _parameters;
  }

  auto isGeneric() const -> bool { return !_parameters.empty(); }

  auto constructors() const -> const VectorUniquePtr<DataConstructorDecl> & {
    return _constructors;
  }

private:
  std::string _name;
  std::vector<ComptimeParam> _parameters;
  VectorUniquePtr<DataConstructorDecl> _constructors;
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

} // end namespace mulberry

#endif // MULBERRY_DECL_H
