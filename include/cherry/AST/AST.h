//===--- AST.h - AST nodes and AST Dumper -----------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_AST_H
#define CHERRY_AST_H

#include "cherry/AST/Decl.h"
#include "cherry/AST/Expr.h"
#include "cherry/AST/Identifier.h"
#include "cherry/AST/Module.h"
#include "cherry/AST/Name.h"
#include "cherry/AST/Stat.h"
#include "cherry/AST/Type.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cherry {

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

  const std::vector<std::unique_ptr<Expr>> &getElements() const {
    return elements;
  }
  void setInferredShape(std::vector<int64_t> shape) { inferredShape = shape; }
  const std::vector<int64_t> &getInferredShape() const { return inferredShape; }
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

  const std::string &getVarName() const { return varName; }
  const std::vector<std::unique_ptr<Expr>> &getIndices() const {
    return indices;
  }
  auto isLvalue() const -> bool override { return true; }
};

auto dumpAST(const llvm::SourceMgr &sourceManager, const Module &module)
    -> void;
} // end namespace cherry

#endif // CHERRY_AST_H
