//===--- LiteralExpr.h - Mulberry Language Literal Expression ASTs -*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_LITERAL_EXPR_H
#define MULBERRY_LITERAL_EXPR_H

#include "mulberry/AST/Expr.h"
#include "mulberry/Basic/IntegerLiteral.h"
#include "llvm/ADT/APFloat.h"
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace mulberry {

class UnitExpr final : public Expr {
public:
  explicit UnitExpr(llvm::SMLoc location) : Expr{Expr_Unit, location} {};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_Unit;
  }
};

class IntegerLiteralExpr final : public Expr {
public:
  explicit IntegerLiteralExpr(llvm::SMLoc location, std::string spelling)
      : Expr{Expr_IntegerLiteral, location}, _spelling(std::move(spelling)){};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_IntegerLiteral;
  }

  auto spelling() const -> std::string_view { return _spelling; }

  auto hasValidSpelling() const -> bool {
    return isValidIntegerLiteralSpelling(_spelling);
  }

  auto getUInt64Value() const -> std::optional<uint64_t> {
    return parseUInt64IntegerLiteral(_spelling);
  }

private:
  std::string _spelling;
};

class FloatLiteralExpr final : public Expr {
public:
  explicit FloatLiteralExpr(llvm::SMLoc location, llvm::APFloat value)
      : Expr{Expr_FloatLiteral, location}, _value(std::move(value)){};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_FloatLiteral;
  }

  auto value() const -> const llvm::APFloat & { return _value; }

private:
  llvm::APFloat _value;
};

class BoolLiteralExpr final : public Expr {
public:
  explicit BoolLiteralExpr(llvm::SMLoc location, bool value)
      : Expr{Expr_BoolLiteral, location}, _value(value){};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_BoolLiteral;
  }

  auto value() const -> bool { return _value; }

private:
  bool _value;
};

class StringLiteralExpr final : public Expr {
public:
  explicit StringLiteralExpr(llvm::SMLoc location, std::string value)
      : Expr{Expr_StringLiteral, location}, _value(std::move(value)){};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_StringLiteral;
  }

  auto value() const -> std::string_view { return _value; }

private:
  std::string _value;
};

class InterpolatedStringExpr final : public Expr {
public:
  InterpolatedStringExpr(llvm::SMLoc location,
                         VectorUniquePtr<Expr> segments)
      : Expr{Expr_InterpolatedString, location},
        _segments(std::move(segments)) {}

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_InterpolatedString;
  }

  // Text remains a StringLiteralExpr; embedded values keep their normal
  // VariableExpr/MemberExpr/IndexExpr shape.
  auto segments() const -> const VectorUniquePtr<Expr> & { return _segments; }

  auto segments() -> VectorUniquePtr<Expr> & { return _segments; }

private:
  VectorUniquePtr<Expr> _segments;
};

// Sema inserts this node only for the default object stringification path.
// The source object remains a reference; MLIRGen adapts it to the hidden
// runtime pointer ABI without exposing Ptr<T> in Mulberry source.
class ObjectIdentityExpr final : public Expr {
public:
  explicit ObjectIdentityExpr(llvm::SMLoc location,
                              std::unique_ptr<Expr> value)
      : Expr{Expr_ObjectIdentity, location}, _value(std::move(value)) {}

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_ObjectIdentity;
  }

  auto value() const -> const std::unique_ptr<Expr> & { return _value; }

  auto typeName() const -> std::string_view { return _typeName; }

  auto setTypeName(std::string_view typeName) -> void {
    _typeName = typeName;
  }

private:
  std::unique_ptr<Expr> _value;
  std::string _typeName;
};

class CharLiteralExpr final : public Expr {
public:
  explicit CharLiteralExpr(llvm::SMLoc location, uint8_t value)
      : Expr{Expr_CharLiteral, location}, _value(value) {};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_CharLiteral;
  }

  auto value() const -> uint8_t { return _value; }

private:
  uint8_t _value;
};

} // end namespace mulberry

#endif // MULBERRY_LITERAL_EXPR_H
