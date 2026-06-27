//===--- CollectionExpr.h - Mulberry collection expressions -------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_COLLECTION_EXPR_H
#define MULBERRY_COLLECTION_EXPR_H

#include "mulberry/AST/Expr.h"
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace mulberry {

// Source `[...]` is type-neutral. Sema classifies it as Tensor or List using
// expected type where available.
class ArrayLiteralExpr final : public Expr {
public:
  enum class LiteralKind {
    Unknown,
    Tensor,
    StdlibList,
  };

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

  auto setInferredShape(std::vector<int64_t> shape) -> void {
    _inferredShape = std::move(shape);
  }

  auto getInferredShape() const -> const std::vector<int64_t> & {
    return _inferredShape;
  }

  auto literalKind() const -> LiteralKind { return _literalKind; }

  auto setTensorLiteral() -> void {
    _literalKind = LiteralKind::Tensor;
  }

  auto setStdlibListLiteral(const Type *elementType,
                            std::string_view withCapacityFunctionName,
                            std::string_view pushFunctionName) -> void {
    _literalKind = LiteralKind::StdlibList;
    _stdlibListElementType = elementType;
    _withCapacityFunctionName = withCapacityFunctionName;
    _pushFunctionName = pushFunctionName;
  }

  auto stdlibListElementType() const -> const Type * {
    return _stdlibListElementType;
  }

  auto withCapacityFunctionName() const -> std::string_view {
    return _withCapacityFunctionName;
  }

  auto pushFunctionName() const -> std::string_view {
    return _pushFunctionName;
  }

private:
  std::vector<std::unique_ptr<Expr>> _elements;
  std::vector<int64_t> _inferredShape;
  LiteralKind _literalKind = LiteralKind::Unknown;
  const Type *_stdlibListElementType = nullptr;
  std::string _withCapacityFunctionName;
  std::string _pushFunctionName;
};

// Source `base[...]` is type-neutral. Sema classifies it by base type.
class IndexExpr final : public Expr {
public:
  enum class IndexKind {
    Unknown,
    Ptr,
    Tensor,
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

  auto setTensorIndex() -> void { _indexKind = IndexKind::Tensor; }

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
