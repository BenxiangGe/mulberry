//===--- SemaComptime.h - Comptime semantic analysis ----------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_SEMA_COMPTIME_H
#define MULBERRY_SEMA_COMPTIME_H

#include "mulberry/AST/AST.h"
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mulberry {

class SemaImpl;

struct ComptimeEvaluation {
  enum class Kind {
    Static,
    Residual,
    Error,
  };

  // Residual is the legal runtime path. Its AST payload will be owned by the
  // interpreter frame and is never borrowed from the source AST.
  Kind kind;
  std::optional<ComptimeValue> value;
  bool isComptimeOnly = false;
  std::unique_ptr<Expr> residual;

  ComptimeEvaluation(
      Kind kind, std::optional<ComptimeValue> value = std::nullopt,
      bool isComptimeOnly = false, std::unique_ptr<Expr> residual = nullptr)
      : kind(kind), value(std::move(value)),
        isComptimeOnly(isComptimeOnly), residual(std::move(residual)) {}

  ComptimeEvaluation(ComptimeEvaluation &&) = default;
  auto operator=(ComptimeEvaluation &&) -> ComptimeEvaluation & = default;
  ComptimeEvaluation(const ComptimeEvaluation &) = delete;
  auto operator=(const ComptimeEvaluation &) -> ComptimeEvaluation & = delete;

  auto takeResidual() -> std::unique_ptr<Expr> { return std::move(residual); }
};

struct ComptimeBinding {
  enum class Kind {
    Static,
    Residual,
    Pack,
  };

  Kind kind = Kind::Static;
  const Type *type = nullptr;
  bool isConst = false;
  bool isComptimeOnly = false;
  std::optional<ComptimeValue> value;
  std::unique_ptr<Expr> residual;
  std::vector<ComptimeBinding> elements;

  static auto staticValue(ComptimeValue value, const Type *type,
                          bool isConst, bool isComptimeOnly)
      -> ComptimeBinding;
  static auto residualValue(std::unique_ptr<Expr> residual, const Type *type,
                            bool isConst) -> ComptimeBinding;
  static auto pack(std::vector<ComptimeBinding> elements) -> ComptimeBinding;
};

// A frame owns staged values for one lexical comptime scope. The map is keyed
// by source name for now because Symbols has not yet exposed declaration
// identity; the parent chain still gives the frame proper lexical behavior.
class ComptimeFrame {
public:
  explicit ComptimeFrame(ComptimeFrame *parent = nullptr)
      : _parent(parent) {}

  auto bind(std::string_view name, ComptimeBinding binding) -> bool;
  auto assign(std::string_view name, ComptimeBinding binding) -> bool;
  auto lookup(std::string_view name) -> ComptimeBinding *;
  auto lookup(std::string_view name) const -> const ComptimeBinding *;

private:
  ComptimeFrame *_parent = nullptr;
  std::map<std::string, ComptimeBinding, std::less<>> _bindings;
};

struct ComptimeControlResult {
  enum class Kind {
    Normal,
    Break,
    Continue,
    Return,
    Error,
  };

  Kind kind = Kind::Normal;
  std::optional<ComptimeEvaluation> value;
  bool keepStatement = false;

  static auto normal() -> ComptimeControlResult { return {}; }
  static auto breakControl() -> ComptimeControlResult {
    return {Kind::Break, std::nullopt};
  }
  static auto continueControl() -> ComptimeControlResult {
    return {Kind::Continue, std::nullopt};
  }
  static auto error() -> ComptimeControlResult {
    return {Kind::Error, std::nullopt};
  }
  static auto returnControl(ComptimeEvaluation value)
      -> ComptimeControlResult {
    return {Kind::Return, std::optional<ComptimeEvaluation>(
                              std::move(value))};
  }

};

struct ComptimeCallFrame {
  std::string name;
  llvm::SMLoc callLocation;
};

// Nested evaluator instances share this state so a comptime operation cannot
// evade its safety limits by entering another lexical frame.
struct ComptimeExecutionState {
  // These are compiler safety limits, not source-language semantics.
  static constexpr uint64_t stepLimit = 1'000'000;
  static constexpr unsigned callDepthLimit = 256;

  uint64_t steps = 0;
  unsigned callDepth = 0;
  bool stepLimitExceeded = false;
  bool callDepthLimitExceeded = false;
  std::vector<ComptimeCallFrame> callStack;
};

class ComptimeSema {
public:
  explicit ComptimeSema(SemaImpl &sema, ComptimeFrame *frame = nullptr,
                        unsigned *loopDepth = nullptr,
                        ComptimeExecutionState *executionState = nullptr,
                        bool isStagedFunctionBody = false);

  auto evaluateComptime(Expr *node, const Type *expectedType = nullptr)
      -> ComptimeEvaluation;
  auto comptimeRuntimeType(const ComptimeValue &value) -> const Type *;
  auto executeStagedFunctionBody(
      BlockExpr *node, const FunctionDecl *specialization = nullptr)
      -> llvm::LogicalResult;

private:
  auto evaluateComptimeBlock(ComptimeBlockExpr *node,
                             const Type *expectedType) -> ComptimeEvaluation;
  auto executeComptimeBlock(BlockExpr *node) -> ComptimeControlResult;
  auto executeComptimeStatements(VectorUniquePtr<Stat> &statements)
      -> ComptimeControlResult;
  auto executeComptimeStatement(Stat *node) -> ComptimeControlResult;
  auto executeComptimeAssignment(AssignExpr *node) -> ComptimeControlResult;
  auto evaluateComptimeCall(CallExpr *node, const Type *expectedType)
      -> ComptimeEvaluation;
  auto evaluateComptimeCallBody(CallExpr *node, const Type *expectedType)
      -> ComptimeEvaluation;
  auto evaluateComptimeStringSlice(CallExpr *node) -> ComptimeEvaluation;
  auto evaluateSourceFunctionCall(CallExpr *node,
                                  const FunctionDecl *function,
                                  const FunctionType *signature)
      -> ComptimeEvaluation;
  auto evaluateComptimeBinary(BinaryExpr *node) -> ComptimeEvaluation;
  auto consumeStep(const Node *node) -> bool;
  auto enterComptimeCall(const CallExpr *node) -> bool;
  auto leaveComptimeCall(const CallExpr *node) -> void;
  auto emitComptimeError(const Node *node, std::string message) -> void;
  auto emitExecutionLimitError(const Node *node, const char *diagnostic)
      -> void;
  auto setComptimeResultType(Expr *node, const ComptimeValue &value) -> void;
  auto makeResidual(Expr *node) -> ComptimeEvaluation;
  auto bindingFromEvaluation(ComptimeEvaluation evaluation,
                             const Type *type, bool isConst)
      -> std::optional<ComptimeBinding>;
  auto evaluationFromBinding(const ComptimeBinding &binding)
      -> ComptimeEvaluation;

  SemaImpl &_sema;
  ComptimeFrame *_frame = nullptr;
  unsigned _localLoopDepth = 0;
  unsigned *_loopDepth = nullptr;
  ComptimeExecutionState _localExecutionState;
  ComptimeExecutionState *_executionState = nullptr;
  bool _isStagedFunctionBody = false;
};

} // namespace mulberry

#endif // MULBERRY_SEMA_COMPTIME_H
