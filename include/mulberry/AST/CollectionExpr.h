//===--- CollectionExpr.h - Mulberry collection expressions -------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_COLLECTION_EXPR_H
#define MULBERRY_COLLECTION_EXPR_H

#include "mulberry/AST/Expr.h"
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace mulberry {

// Source `[...]` defaults to the plain language Array.
class ArrayLiteralExpr final : public Expr {
public:
  ArrayLiteralExpr(llvm::SMLoc loc,
                   std::vector<std::unique_ptr<Expr>> elements)
      : Expr(Expr_ArrayLiteral, loc), _elements(std::move(elements)) {}

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_ArrayLiteral;
  }

  auto getElements() const
      -> const std::vector<std::unique_ptr<Expr>> & {
    return _elements;
  }

  auto getElements() -> std::vector<std::unique_ptr<Expr>> & {
    return _elements;
  }

private:
  std::vector<std::unique_ptr<Expr>> _elements;
};

// Source `base[...]` is type-neutral. Sema classifies it by base type.
class IndexExpr final : public Expr {
public:
  enum class IndexKind {
    Unknown,
    Ptr,
    Array,
    StdlibTensor,
    StdlibList,
  };

  IndexExpr(llvm::SMLoc loc, std::unique_ptr<Expr> base,
            std::vector<std::unique_ptr<Expr>> indices)
      : Expr(Expr_Index, loc), _base(std::move(base)),
        _indices(std::move(indices)) {}

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_Index;
  }

  auto base() const -> const std::unique_ptr<Expr> & { return _base; }

  auto indices() const -> const std::vector<std::unique_ptr<Expr>> & {
    return _indices;
  }

  auto isLvalue() const -> bool override { return _isLvalue; }

  auto setLvalue(bool isLvalue) -> void { _isLvalue = isLvalue; }

  auto indexKind() const -> IndexKind { return _indexKind; }

  auto setPtrIndex() -> void { _indexKind = IndexKind::Ptr; }

  auto setArrayIndex() -> void { _indexKind = IndexKind::Array; }

  auto setStdlibTensorIndex() -> void {
    _indexKind = IndexKind::StdlibTensor;
  }

  auto setStdlibListIndex(std::string_view getFunctionName,
                          std::string_view setFunctionName) -> void {
    _indexKind = IndexKind::StdlibList;
    _getFunctionName = getFunctionName;
    _setFunctionName = setFunctionName;
  }

  auto getFunctionName() const -> std::string_view {
    return _getFunctionName;
  }

  auto setFunctionName() const -> std::string_view {
    return _setFunctionName;
  }

private:
  std::unique_ptr<Expr> _base;
  std::vector<std::unique_ptr<Expr>> _indices;
  bool _isLvalue = false;
  IndexKind _indexKind = IndexKind::Unknown;
  std::string _getFunctionName;
  std::string _setFunctionName;
};

} // end namespace mulberry

#endif // MULBERRY_COLLECTION_EXPR_H
