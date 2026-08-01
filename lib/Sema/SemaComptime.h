//===--- SemaComptime.h - Comptime semantic analysis ----------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_SEMA_COMPTIME_H
#define MULBERRY_SEMA_COMPTIME_H

#include "mulberry/AST/AST.h"
#include <optional>

namespace mulberry {

class SemaImpl;

struct ComptimeEvaluation {
  enum class Kind {
    Runtime,
    Value,
    Error,
  };

  Kind kind;
  std::optional<ComptimeValue> value;
  bool isComptimeOnly = false;
};

class ComptimeSema {
public:
  explicit ComptimeSema(SemaImpl &sema) : _sema(sema) {}

  auto evaluateComptime(Expr *node) -> ComptimeEvaluation;
  auto comptimeRuntimeType(const ComptimeValue &value) -> const Type *;

private:
  auto evaluateComptimeBlock(ComptimeBlockExpr *node) -> ComptimeEvaluation;
  auto evaluateComptimeCall(CallExpr *node) -> ComptimeEvaluation;
  auto evaluateComptimeBinary(BinaryExpr *node) -> ComptimeEvaluation;
  auto setComptimeResultType(Expr *node, const ComptimeValue &value) -> void;

  SemaImpl &_sema;
};

} // namespace mulberry

#endif // MULBERRY_SEMA_COMPTIME_H
