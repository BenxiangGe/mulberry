//===--- AST.h - AST nodes and AST Dumper -----------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_AST_H
#define CHERRY_AST_H

#include "cherry/AST/Identifier.h"
#include "cherry/AST/Decl.h"
#include "cherry/AST/Expr.h"
#include "cherry/AST/Identifier.h"
#include "cherry/AST/Module.h"
#include "cherry/AST/Stat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/Location.h"
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cherry {

struct ParsedListType {
  std::string elementType;
  std::vector<int64_t> shape;
};

inline auto formatListTypeName(llvm::StringRef elementType,
                               llvm::ArrayRef<int64_t> shape) -> std::string {
  std::string name = "list<";
  name += elementType.str();
  for (auto dim : shape) {
    name += ", ";
    name += dim < 0 ? "?" : std::to_string(dim);
  }
  name += ">";
  return name;
}

inline auto parseListTypeName(llvm::StringRef typeName)
    -> std::optional<ParsedListType> {
  if (!typeName.consume_front("list<") || !typeName.consume_back(">"))
    return std::nullopt;

  llvm::SmallVector<llvm::StringRef, 4> parts;
  typeName.split(parts, ",");
  if (parts.size() < 2)
    return std::nullopt;

  ParsedListType parsed;
  parsed.elementType = parts.front().trim().str();
  for (auto part : llvm::drop_begin(parts)) {
    auto dimText = part.trim();
    if (dimText == "?") {
      parsed.shape.push_back(-1);
      continue;
    }

    int64_t dim = 0;
    if (dimText.getAsInteger(10, dim) || dim <= 0)
      return std::nullopt;
    parsed.shape.push_back(dim);
  }
  return parsed;
}

inline auto isListTypeName(llvm::StringRef typeName) -> bool {
  return parseListTypeName(typeName).has_value();
}

// list type. e.g. `list<f64, 10, 20>`
class ListType : public Type {
    std::unique_ptr<Type> elementType_;
    std::vector<int64_t> shape_;
public:
    ListType(std::unique_ptr<Type> elementType, std::vector<int64_t> shape, llvm::SMLoc location)
        : Type(location, formatListTypeName(elementType->name(), shape)),
          elementType_(std::move(elementType)), shape_(shape) {}

    Type* getElementType() const { return elementType_.get(); }
    const std::vector<int64_t>& getShape() const { return shape_; }
};

// list literal. e.g. `[[1.0, 2.0], [3.0, 4.0]]`
class ListLiteralExpr : public Expr {
    std::vector<std::unique_ptr<Expr>> elements;
    std::vector<int64_t> inferredShape;
public:
    ListLiteralExpr(llvm::SMLoc loc, std::vector<std::unique_ptr<Expr>> elements)
        : Expr(Expr_ListLiteral, loc), elements(std::move(elements)) {}

    static auto classof(const Expr *node) -> bool {
      return node->getKind() == Expr_ListLiteral;
    }

    const std::vector<std::unique_ptr<Expr>>& getElements() const { return elements; }
    void setInferredShape(std::vector<int64_t> shape) { inferredShape = shape; }
    const std::vector<int64_t>& getInferredShape() const { return inferredShape; }
};

// list access. e.g. `my_list[i, j]`
class ListAccessExpr : public Expr {
    std::string varName;
    std::vector<std::unique_ptr<Expr>> indices;
public:
    ListAccessExpr(llvm::SMLoc loc, llvm::StringRef varName,
                   std::vector<std::unique_ptr<Expr>> indices)
        : Expr(Expr_ListAccess, loc), varName(varName.str()),
          indices(std::move(indices)) {}

    static auto classof(const Expr *node) -> bool {
      return node->getKind() == Expr_ListAccess;
    }

    const std::string& getVarName() const { return varName; }
    const std::vector<std::unique_ptr<Expr>>& getIndices() const { return indices; }
    auto isLvalue() const -> bool override { return true; }
};

auto dumpAST(const llvm::SourceMgr &sourceManager, const Module &module)
    -> void;
} // end namespace cherry

#endif // CHERRY_AST_H
