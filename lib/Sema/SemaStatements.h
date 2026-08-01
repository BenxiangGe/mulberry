//===--- SemaStatements.h - Statement semantic analysis ------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_SEMA_STATEMENTS_H
#define MULBERRY_SEMA_STATEMENTS_H

#include "mulberry/AST/AST.h"
#include <vector>

namespace mulberry {

class SemaImpl;

class StatementSema {
public:
  explicit StatementSema(SemaImpl &sema) : _sema(sema) {}

  auto sema(BlockExpr *node) -> llvm::LogicalResult;
  auto sema(Stat *node) -> llvm::LogicalResult;

  auto checkMatchPattern(DataPattern *pattern, const DataType *dataType,
                         std::vector<bool> &covered,
                         const DataConstructor *&constructor)
      -> llvm::LogicalResult;
  auto declareMatchPatternBindings(DataPattern *pattern,
                                   const DataConstructor *constructor)
      -> llvm::LogicalResult;
  auto checkExhaustiveMatch(const Node *node,
                            const std::vector<bool> &covered)
      -> llvm::LogicalResult;

private:
  auto sema(VariableStat *node) -> llvm::LogicalResult;
  auto sema(ExprStat *node) -> llvm::LogicalResult;
  auto sema(IfStat *node) -> llvm::LogicalResult;
  auto sema(MatchStat *node) -> llvm::LogicalResult;
  auto sema(WhileStat *node) -> llvm::LogicalResult;
  auto sema(ForStat *node) -> llvm::LogicalResult;
  auto sema(BreakStat *node) -> llvm::LogicalResult;
  auto sema(ContinueStat *node) -> llvm::LogicalResult;
  auto sema(ReturnStat *node) -> llvm::LogicalResult;

  SemaImpl &_sema;
};

} // namespace mulberry

#endif // MULBERRY_SEMA_STATEMENTS_H
