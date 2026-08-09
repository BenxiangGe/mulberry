//===--- SemaComptime.cpp - Comptime semantic analysis --------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "SemaComptime.h"
#include "SemaImpl.h"
#include "SemaSupport.h"
#include "mulberry/Basic/Builtins.h"
#include "llvm/Support/Debug.h"
#include <utility>

#undef DEBUG_TYPE
#define DEBUG_TYPE "Sema"

namespace mulberry {
using llvm::dyn_cast;

namespace {

auto makeComptimeLiteral(const ComptimeValue &value, llvm::SMLoc location)
    -> std::unique_ptr<Expr> {
  switch (value.kind()) {
  case ComptimeValue::Kind::Type:
    return nullptr;
  case ComptimeValue::Kind::Bool:
    return std::make_unique<BoolLiteralExpr>(location, value.boolValue());
  case ComptimeValue::Kind::UInt8:
    return std::make_unique<CharLiteralExpr>(location, value.uint8Value());
  case ComptimeValue::Kind::UInt64:
    return std::make_unique<IntegerLiteralExpr>(
        location, std::to_string(value.uint64Value()));
  case ComptimeValue::Kind::String:
    return std::make_unique<StringLiteralExpr>(
        location, std::string(value.stringValue()));
  }
  return nullptr;
}

} // namespace

auto ComptimeBinding::staticValue(ComptimeValue value, const Type *type,
                                  bool isConst, bool isComptimeOnly)
    -> ComptimeBinding {
  ComptimeBinding binding;
  binding.kind = Kind::Static;
  binding.type = type;
  binding.isConst = isConst;
  binding.isComptimeOnly = isComptimeOnly;
  binding.value = std::move(value);
  return binding;
}

auto ComptimeBinding::residualValue(std::unique_ptr<Expr> residual,
                                    const Type *type, bool isConst)
    -> ComptimeBinding {
  ComptimeBinding binding;
  binding.kind = Kind::Residual;
  binding.type = type;
  binding.isConst = isConst;
  binding.residual = std::move(residual);
  return binding;
}

auto ComptimeBinding::pack(std::vector<ComptimeBinding> elements)
    -> ComptimeBinding {
  ComptimeBinding binding;
  binding.kind = Kind::Pack;
  binding.elements = std::move(elements);
  return binding;
}

auto ComptimeFrame::bind(std::string_view name, ComptimeBinding binding)
    -> bool {
  return _bindings.emplace(std::string(name), std::move(binding)).second;
}

auto ComptimeFrame::assign(std::string_view name, ComptimeBinding binding)
    -> bool {
  if (auto *current = lookup(name)) {
    if (current->isConst)
      return false;
    *current = std::move(binding);
    return true;
  }
  return false;
}

auto ComptimeFrame::lookup(std::string_view name) -> ComptimeBinding * {
  auto binding = _bindings.find(name);
  if (binding != _bindings.end())
    return &binding->second;
  return _parent ? _parent->lookup(name) : nullptr;
}

auto ComptimeFrame::lookup(std::string_view name) const
    -> const ComptimeBinding * {
  auto binding = _bindings.find(name);
  if (binding != _bindings.end())
    return &binding->second;
  return _parent ? _parent->lookup(name) : nullptr;
}

ComptimeSema::ComptimeSema(SemaImpl &sema, ComptimeFrame *frame,
                           unsigned *loopDepth,
                           ComptimeExecutionState *executionState,
                           bool isStagedFunctionBody)
    : _sema(sema), _frame(frame ? frame : sema._activeComptimeFrame),
      _loopDepth(loopDepth ? loopDepth : &_localLoopDepth),
      _executionState(
          executionState
              ? executionState
              : (sema._activeComptimeExecutionState
                     ? sema._activeComptimeExecutionState
                     : &_localExecutionState)),
      _isStagedFunctionBody(isStagedFunctionBody) {}

auto ComptimeSema::emitComptimeError(const Node *node, std::string message)
    -> void {
  auto location = node->location();
  if (!_executionState->callStack.empty()) {
    location = _executionState->callStack.front().callLocation;
    message += " (comptime call stack: ";
    bool first = true;
    for (const auto &frame : _executionState->callStack) {
      if (!first)
        message += " -> ";
      message += frame.name.empty() ? "<anonymous>" : frame.name;
      first = false;
    }
    message += ")";
  }
  (void)_sema.emitError(location, message);

  if (location != node->location())
    _sema.emitNote(node->location(), "comptime failure was triggered here");
  for (size_t index = 1; index < _executionState->callStack.size(); ++index) {
    const auto &frame = _executionState->callStack[index];
    std::string note = "while evaluating comptime call `";
    note += frame.name.empty() ? "<anonymous>" : frame.name;
    note += "`";
    _sema.emitNote(frame.callLocation, note);
  }
}

auto ComptimeSema::emitExecutionLimitError(const Node *node,
                                           const char *diagnostic) -> void {
  emitComptimeError(node, diagnostic);
}

auto ComptimeSema::consumeStep(const Node *node) -> bool {
  if (_executionState->stepLimitExceeded)
    return false;
  if (_executionState->steps >= ComptimeExecutionState::stepLimit) {
    _executionState->stepLimitExceeded = true;
    LLVM_DEBUG({
      llvm::dbgs() << "comptime step limit reached; call stack:";
      for (const auto &frame : _executionState->callStack)
        llvm::dbgs() << " " << frame.name;
      llvm::dbgs() << "\n";
    });
    emitExecutionLimitError(node, diag::comptime_step_limit);
    return false;
  }
  ++_executionState->steps;
  return true;
}

auto ComptimeSema::enterComptimeCall(const CallExpr *node) -> bool {
  if (_executionState->callDepthLimitExceeded)
    return false;
  if (_executionState->callDepth >=
      ComptimeExecutionState::callDepthLimit) {
    _executionState->callDepthLimitExceeded = true;
    LLVM_DEBUG({
      llvm::dbgs() << "comptime call depth limit reached; call stack:";
      for (const auto &frame : _executionState->callStack)
        llvm::dbgs() << " " << frame.name;
      llvm::dbgs() << "\n";
    });
    emitExecutionLimitError(node, diag::comptime_call_depth_limit);
    return false;
  }
  ++_executionState->callDepth;
  _executionState->callStack.push_back(
      {std::string(node->name()), node->location()});
  LLVM_DEBUG(llvm::dbgs() << "enter comptime call `" << node->name()
                          << "` depth=" << _executionState->callDepth
                          << "\n");
  return true;
}

auto ComptimeSema::leaveComptimeCall(const CallExpr *node) -> void {
  LLVM_DEBUG(llvm::dbgs() << "leave comptime call `" << node->name()
                          << "` depth=" << _executionState->callDepth
                          << "\n");
  if (!_executionState->callStack.empty())
    _executionState->callStack.pop_back();
  if (_executionState->callDepth != 0)
    --_executionState->callDepth;
}

auto ComptimeSema::comptimeRuntimeType(const ComptimeValue &value)
    -> const Type * {
  switch (value.kind()) {
  case ComptimeValue::Kind::Type:
    return nullptr;
  case ComptimeValue::Kind::Bool:
    return _sema._typeContext.getBuiltinType(BuiltinTypeKind::Bool);
  case ComptimeValue::Kind::UInt8:
    return _sema._typeContext.getBuiltinType(BuiltinTypeKind::UInt8);
  case ComptimeValue::Kind::UInt64:
    return _sema._typeContext.getBuiltinType(BuiltinTypeKind::UInt64);
  case ComptimeValue::Kind::String:
    return _sema.lookupType("String");
  }
  llvm_unreachable("unexpected comptime value");
}

auto ComptimeSema::setComptimeResultType(Expr *node,
                                         const ComptimeValue &value) -> void {
  if (auto *type = comptimeRuntimeType(value))
    node->setType(type);
}

auto ComptimeSema::makeResidual(Expr *node) -> ComptimeEvaluation {
  auto residual = substituteExpr(node, {});
  residual->setType(node->type());
  return {ComptimeEvaluation::Kind::Residual, std::nullopt, false,
          std::move(residual)};
}

auto ComptimeSema::bindingFromEvaluation(
    ComptimeEvaluation evaluation, const Type *type, bool isConst)
    -> std::optional<ComptimeBinding> {
  if (evaluation.kind == ComptimeEvaluation::Kind::Static) {
    if (!evaluation.value)
      return std::nullopt;
    return ComptimeBinding::staticValue(
        *evaluation.value, type, isConst, evaluation.isComptimeOnly);
  }
  if (evaluation.kind == ComptimeEvaluation::Kind::Residual)
    return ComptimeBinding::residualValue(evaluation.takeResidual(), type,
                                          isConst);
  return std::nullopt;
}

auto ComptimeSema::evaluationFromBinding(const ComptimeBinding &binding)
    -> ComptimeEvaluation {
  if (binding.kind == ComptimeBinding::Kind::Static) {
    if (!binding.value)
      return {ComptimeEvaluation::Kind::Error, std::nullopt};
    return {ComptimeEvaluation::Kind::Static, *binding.value,
            binding.isComptimeOnly};
  }
  if (binding.kind == ComptimeBinding::Kind::Residual) {
    auto residual = substituteExpr(binding.residual.get(), {});
    residual->setType(binding.type);
    return {ComptimeEvaluation::Kind::Residual, std::nullopt, false,
            std::move(residual)};
  }
  return {ComptimeEvaluation::Kind::Error, std::nullopt};
}

auto ComptimeSema::executeComptimeAssignment(AssignExpr *node)
    -> ComptimeControlResult {
  if (llvm::failed(_sema.sema(node->lhs().get())))
    return ComptimeControlResult::error();
  auto evaluation = evaluateComptime(node->rhs().get(), node->lhs()->type());
  if (evaluation.kind == ComptimeEvaluation::Kind::Error)
    return ComptimeControlResult::error();
  if (evaluation.kind == ComptimeEvaluation::Kind::Static &&
      _isStagedFunctionBody && evaluation.value &&
      evaluation.value->kind() != ComptimeValue::Kind::Type) {
    auto literal = makeComptimeLiteral(*evaluation.value,
                                        node->rhs()->location());
    if (!literal)
      return ComptimeControlResult::error();
    node->rhs() = std::move(literal);
    LLVM_DEBUG(llvm::dbgs()
               << "materialize static assignment rhs in staged body\n");
  }
  if (evaluation.kind == ComptimeEvaluation::Kind::Residual) {
    node->rhs() = evaluation.takeResidual();
    LLVM_DEBUG(llvm::dbgs()
               << "replace comptime assignment rhs with residual expression\n");
  }

  if (llvm::failed(_sema.sema(node)))
    return ComptimeControlResult::error();

  auto *variable = dyn_cast<VariableExpr>(node->lhs().get());
  if (!variable || !_frame || !_frame->lookup(variable->name()))
    return ComptimeControlResult::error();

  auto *current = _frame->lookup(variable->name());
  if (current->isConst) {
    (void)_sema.emitError(variable, diag::assign_const);
    return ComptimeControlResult::error();
  }

  std::optional<ComptimeBinding> binding;
  if (evaluation.kind == ComptimeEvaluation::Kind::Residual) {
    // The RHS already belongs to the runtime assignment. Keep only a reference
    // in the comptime frame; cloning the RHS here would evaluate it again every
    // time the local is read during residualization.
    binding = ComptimeBinding::residualValue(
        std::make_unique<VariableExpr>(variable->location(), variable->name()),
        node->lhs()->type(), /*isConst=*/false);
  } else {
    binding = bindingFromEvaluation(std::move(evaluation),
                                    node->lhs()->type(), /*isConst=*/false);
  }
  if (!binding || !_frame->assign(variable->name(), std::move(*binding)))
    return ComptimeControlResult::error();

  LLVM_DEBUG(llvm::dbgs() << "update comptime binding `" << variable->name()
                          << "`\n");
  return ComptimeControlResult::normal();
}

auto ComptimeSema::executeComptimeStatement(Stat *node)
    -> ComptimeControlResult {
  SemaImpl::ComptimeExecutionScope executionScope(_sema, _executionState);
  if (!consumeStep(node))
    return ComptimeControlResult::error();

  if (auto *variable = dyn_cast<VariableStat>(node)) {
    const Type *expectedType = nullptr;
    if (variable->hasExplicitType()) {
      expectedType = _sema.checkType(variable->typeNode(),
                                     SemaImpl::UnitPolicy::Allow);
      if (!expectedType)
        return ComptimeControlResult::error();
    }
    auto evaluation = evaluateComptime(variable->init().get(), expectedType);
    if (evaluation.kind == ComptimeEvaluation::Kind::Error)
      return ComptimeControlResult::error();
    if (evaluation.kind == ComptimeEvaluation::Kind::Static &&
        _isStagedFunctionBody && evaluation.value &&
        evaluation.value->kind() != ComptimeValue::Kind::Type) {
      auto literal = makeComptimeLiteral(*evaluation.value,
                                          variable->init()->location());
      if (!literal)
        return ComptimeControlResult::error();
      variable->init() = std::move(literal);
      LLVM_DEBUG(llvm::dbgs()
                 << "materialize static initializer in staged body\n");
    }
    if (evaluation.kind == ComptimeEvaluation::Kind::Residual) {
      variable->init() = evaluation.takeResidual();
      LLVM_DEBUG(llvm::dbgs()
                 << "replace comptime initializer with residual expression\n");
    }

    if (llvm::failed(_sema.sema(variable)))
      return ComptimeControlResult::error();

    std::optional<ComptimeBinding> binding;
    if (evaluation.kind == ComptimeEvaluation::Kind::Residual) {
      // The declaration owns the runtime initializer. The frame must model
      // the declared local, not replay that initializer on every lookup.
      binding = ComptimeBinding::residualValue(
          std::make_unique<VariableExpr>(variable->variable()->location(),
                                         variable->variable()->name()),
          variable->type(), variable->isConstBinding());
    } else {
      binding = bindingFromEvaluation(
          std::move(evaluation), variable->type(), variable->isConstBinding());
    }
    if (!binding || !_frame ||
        !_frame->bind(variable->variable()->name(), std::move(*binding)))
      return ComptimeControlResult::error();

    LLVM_DEBUG(llvm::dbgs() << "bind comptime "
                            << (variable->isConstBinding() ? "const " : "var ")
                            << "`" << variable->variable()->name() << "`\n");
    return ComptimeControlResult::normal();
  }

  if (auto *expression = dyn_cast<ExprStat>(node)) {
    if (auto *assignment = dyn_cast<AssignExpr>(expression->expression().get()))
      return executeComptimeAssignment(assignment);
    if (llvm::failed(_sema.sema(expression->expression().get())))
      return ComptimeControlResult::error();
    auto evaluation = evaluateComptime(expression->expression().get());
    if (evaluation.kind == ComptimeEvaluation::Kind::Error)
      return ComptimeControlResult::error();
    if (evaluation.kind == ComptimeEvaluation::Kind::Residual) {
      (void)_sema.emitError(expression, diag::expected_comptime_value);
      return ComptimeControlResult::error();
    }
    return ComptimeControlResult::normal();
  }

  if (auto *ifStat = dyn_cast<IfStat>(node)) {
    auto condition = evaluateComptime(ifStat->conditionExpr().get());
    if (condition.kind == ComptimeEvaluation::Kind::Error)
      return ComptimeControlResult::error();
    if (condition.kind != ComptimeEvaluation::Kind::Static) {
      (void)_sema.emitError(ifStat->conditionExpr().get(),
                            diag::expected_comptime_value);
      return ComptimeControlResult::error();
    }
    if (condition.value->kind() != ComptimeValue::Kind::Bool) {
      (void)_sema.emitError(ifStat->conditionExpr().get(), diag::expected_bool);
      return ComptimeControlResult::error();
    }
    auto value = condition.value->boolValue();
    ifStat->setComptimeValue(value);
    ComptimeControlResult result;
    if (value)
      result = executeComptimeBlock(ifStat->thenBlock().get());
    else if (ifStat->hasElseBlock())
      result = executeComptimeBlock(ifStat->elseBlock().get());
    if (result.kind != ComptimeControlResult::Kind::Normal) {
      result.keepStatement =
          _isStagedFunctionBody &&
          result.kind == ComptimeControlResult::Kind::Return;
      return result;
    }
    return ComptimeControlResult::normal();
  }

  if (auto *whileStat = dyn_cast<WhileStat>(node)) {
    VectorUniquePtr<BlockExpr> unrolledBodies;
    ++*_loopDepth;
    auto loopResult = ComptimeControlResult::normal();
    while (true) {
      auto condition = evaluateComptime(whileStat->conditionExpr().get());
      if (condition.kind == ComptimeEvaluation::Kind::Error) {
        loopResult = ComptimeControlResult::error();
        break;
      }
      if (condition.kind != ComptimeEvaluation::Kind::Static) {
        (void)_sema.emitError(whileStat->conditionExpr().get(),
                              diag::expected_comptime_value);
        loopResult = ComptimeControlResult::error();
        break;
      }
      if (condition.value->kind() != ComptimeValue::Kind::Bool) {
        (void)_sema.emitError(whileStat->conditionExpr().get(),
                              diag::expected_bool);
        loopResult = ComptimeControlResult::error();
        break;
      }
      if (!condition.value->boolValue())
        break;

      // Each iteration needs its own AST because residual assignments evolve
      // as the static loop state changes.
      auto iterationBody =
          substituteBlockExpr(whileStat->bodyBlock().get(), {});
      loopResult = executeComptimeBlock(iterationBody.get());
      unrolledBodies.push_back(std::move(iterationBody));
      if (loopResult.kind == ComptimeControlResult::Kind::Break) {
        loopResult = ComptimeControlResult::normal();
        break;
      }
      if (loopResult.kind == ComptimeControlResult::Kind::Continue)
        continue;
      if (loopResult.kind != ComptimeControlResult::Kind::Normal)
        break;
    }
    --*_loopDepth;
    whileStat->setComptimeUnrolledBodies(std::move(unrolledBodies));
    if (loopResult.kind == ComptimeControlResult::Kind::Return)
      loopResult.keepStatement = _isStagedFunctionBody;
    return loopResult;
  }

  if (dyn_cast<BreakStat>(node)) {
    if (*_loopDepth == 0) {
      (void)_sema.emitError(node, diag::loop_control_outside_loop);
      return ComptimeControlResult::error();
    }
    return ComptimeControlResult::breakControl();
  }

  if (dyn_cast<ContinueStat>(node)) {
    if (*_loopDepth == 0) {
      (void)_sema.emitError(node, diag::loop_control_outside_loop);
      return ComptimeControlResult::error();
    }
    return ComptimeControlResult::continueControl();
  }

  if (auto *returnStat = dyn_cast<ReturnStat>(node)) {
    if (!returnStat->hasExpression()) {
      (void)_sema.emitError(returnStat, diag::expected_comptime_value);
      return ComptimeControlResult::error();
    }
    auto evaluation = evaluateComptime(returnStat->expression().get(),
                                       _sema._currentFunctionReturnType);
    if (evaluation.kind == ComptimeEvaluation::Kind::Error)
      return ComptimeControlResult::error();
    if (evaluation.kind == ComptimeEvaluation::Kind::Static &&
        _isStagedFunctionBody && evaluation.value &&
        evaluation.value->kind() != ComptimeValue::Kind::Type) {
      auto literal = makeComptimeLiteral(*evaluation.value,
                                          returnStat->expression()->location());
      if (!literal)
        return ComptimeControlResult::error();
      returnStat->expression() = std::move(literal);
      LLVM_DEBUG(llvm::dbgs()
                 << "materialize static return expression in staged body\n");
    }
    if (evaluation.kind == ComptimeEvaluation::Kind::Residual) {
      returnStat->expression() = evaluation.takeResidual();
      evaluation.residual = substituteExpr(returnStat->expression().get(), {});
    }
    if (_isStagedFunctionBody && llvm::failed(_sema.sema(returnStat)))
      return ComptimeControlResult::error();
    auto result = ComptimeControlResult::returnControl(std::move(evaluation));
    result.keepStatement = _isStagedFunctionBody;
    return result;
  }

  (void)_sema.emitError(node, diag::expected_comptime_value);
  return ComptimeControlResult::error();
}

auto ComptimeSema::executeComptimeStatements(
    VectorUniquePtr<Stat> &statements) -> ComptimeControlResult {
  for (size_t index = 0; index < statements.size();) {
    auto &statement = statements[index];
    auto result = executeComptimeStatement(statement.get());

    // A staged specialization must leave ordinary blocks behind. Keeping a
    // constant IfStat/WhileStat and asking MLIRGen to interpret its marker
    // would leak the interpreter protocol into the residual AST. ExprStat
    // already provides the right lexical scope for a residual BlockExpr.
    auto replaceWithBlocks = [&](VectorUniquePtr<BlockExpr> blocks) {
      auto position = statements.begin() + index;
      statements.erase(position);
      size_t inserted = 0;
      for (auto &block : blocks) {
        block->setType(
            _sema._typeContext.getBuiltinType(BuiltinTypeKind::Unit));
        auto blockStatement = std::make_unique<ExprStat>(
            block->location(), std::move(block));
        statements.insert(statements.begin() + index + inserted,
                          std::move(blockStatement));
        ++inserted;
      }
      return inserted;
    };

    if (_isStagedFunctionBody) {
      if (auto *ifStat = dyn_cast<IfStat>(statement.get());
          ifStat && ifStat->comptimeValue()) {
        auto selectedBlock = *ifStat->comptimeValue()
                                 ? std::move(ifStat->thenBlock())
                                 : (ifStat->hasElseBlock()
                                        ? std::move(ifStat->elseBlock())
                                        : std::make_unique<BlockExpr>(
                                              ifStat->location(),
                                              VectorUniquePtr<Stat>{}));
        VectorUniquePtr<BlockExpr> blocks;
        blocks.push_back(std::move(selectedBlock));
        auto inserted = replaceWithBlocks(std::move(blocks));
        if (result.kind != ComptimeControlResult::Kind::Normal) {
          statements.resize(index + inserted);
          return result;
        }
        index += inserted;
        continue;
      }

      if (auto *whileStat = dyn_cast<WhileStat>(statement.get());
          whileStat && whileStat->isComptimeUnrolled()) {
        auto inserted = replaceWithBlocks(
            std::move(whileStat->comptimeUnrolledBodies()));
        if (result.kind != ComptimeControlResult::Kind::Normal) {
          statements.resize(index + inserted);
          return result;
        }
        index += inserted;
        continue;
      }

      if (result.kind == ComptimeControlResult::Kind::Normal) {
        if (auto *variable = dyn_cast<VariableStat>(statement.get())) {
          auto isComptimeOnly = variable->comptimeValue().has_value();
          if (!isComptimeOnly && _frame) {
            if (auto *binding =
                    _frame->lookup(variable->variable()->name()))
              isComptimeOnly = binding->kind == ComptimeBinding::Kind::Static &&
                               binding->isComptimeOnly;
          }
          // Ordinary Sema rechecks a materialized initializer and can clear
          // the AST marker. The frame retains the phase information needed to
          // remove the compile-time-only declaration here.
          if (isComptimeOnly) {
            statements.erase(statements.begin() + index);
            continue;
          }
        }
      }

      if (result.kind != ComptimeControlResult::Kind::Normal) {
        if (result.keepStatement)
          statements.resize(index + 1);
        else
          statements.resize(index);
        return result;
      }

      ++index;
      continue;
    }

    if (result.kind != ComptimeControlResult::Kind::Normal) {
      if (result.keepStatement)
        statements.resize(index + 1);
      else
        statements.resize(index);
      return result;
    }

    ++index;
  }
  return ComptimeControlResult::normal();
}

auto ComptimeSema::executeComptimeBlock(BlockExpr *node)
    -> ComptimeControlResult {
  SemaImpl::VariableScope symbolScope(_sema._symbols);
  ComptimeFrame frame(_frame);
  SemaImpl::ComptimeFrameScope frameScope(_sema, &frame);
  ComptimeSema blockSema(_sema, &frame, _loopDepth, _executionState,
                         _isStagedFunctionBody);
  return blockSema.executeComptimeStatements(node->statements());
}

auto ComptimeSema::executeStagedFunctionBody(
    BlockExpr *node, const FunctionDecl *specialization)
    -> llvm::LogicalResult {
  _isStagedFunctionBody = true;
  auto stackSize = _executionState->callStack.size();
  if (specialization && specialization->specializationOrigin()) {
    const auto &origin = *specialization->specializationOrigin();
    _executionState->callStack.push_back(
        {origin.genericName, origin.callLocation});
    LLVM_DEBUG(llvm::dbgs() << "enter comptime specialization `"
                            << origin.genericName << "`\n");
  }
  auto control = executeComptimeBlock(node);
  _executionState->callStack.resize(stackSize);
  if (control.kind == ComptimeControlResult::Kind::Error)
    return failure();
  if (control.kind == ComptimeControlResult::Kind::Break ||
      control.kind == ComptimeControlResult::Kind::Continue)
    return _sema.emitError(node, diag::loop_control_outside_loop);
  return success();
}

auto ComptimeSema::evaluateComptime(Expr *node, const Type *expectedType)
    -> ComptimeEvaluation {
  SemaImpl::ComptimeExecutionScope executionScope(_sema, _executionState);
  if (!consumeStep(node))
    return {ComptimeEvaluation::Kind::Error, std::nullopt};

  if (auto *block = dyn_cast<ComptimeBlockExpr>(node))
    return evaluateComptimeBlock(block, expectedType);

  if (auto *typeExpr = dyn_cast<TypeInfoExpr>(node)) {
    auto *type = _sema.resolveType(typeExpr->typeNode());
    if (!type)
      return {ComptimeEvaluation::Kind::Error, std::nullopt};
    return {ComptimeEvaluation::Kind::Static, ComptimeValue(type), true};
  }

  if (auto *compileError = dyn_cast<CompileErrorExpr>(node)) {
    auto message = evaluateComptime(compileError->message().get());
    if (message.kind == ComptimeEvaluation::Kind::Error)
      return message;
    if (message.kind != ComptimeEvaluation::Kind::Static ||
        !message.value ||
        message.value->kind() != ComptimeValue::Kind::String) {
      (void)_sema.emitError(compileError->message().get(),
                            diag::comptime_compile_error_message);
      return {ComptimeEvaluation::Kind::Error, std::nullopt};
    }

    LLVM_DEBUG(llvm::dbgs() << "emit comptime compile error\n");
    emitComptimeError(compileError,
                      std::string(message.value->stringValue()));
    return {ComptimeEvaluation::Kind::Error, std::nullopt};
  }

  if (auto *value = dyn_cast<BoolLiteralExpr>(node)) {
    _sema.setBuiltinType(value, BuiltinTypeKind::Bool);
    return {ComptimeEvaluation::Kind::Static,
            ComptimeValue(value->value())};
  }

  if (auto *value = dyn_cast<IntegerLiteralExpr>(node)) {
    if (!value->hasValidSpelling()) {
      (void)_sema.emitError(value, diag::invalid_integer_literal);
      return {ComptimeEvaluation::Kind::Error, std::nullopt};
    }
    if (value->isNegative() || isIntegerType(value->type()) ||
        isInt64Type(value->type()))
      return makeResidual(node);
    auto uint64Value = value->getUInt64Value();
    if (!uint64Value) {
      (void)_sema.emitError(value, diag::integer_literal_overflows);
      return {ComptimeEvaluation::Kind::Error, std::nullopt};
    }
    _sema.setBuiltinType(value, BuiltinTypeKind::UInt64);
    return {ComptimeEvaluation::Kind::Static,
            ComptimeValue(*uint64Value)};
  }

  if (auto *value = dyn_cast<StringLiteralExpr>(node))
    return {ComptimeEvaluation::Kind::Static,
            ComptimeValue(value->value())};

  if (auto *value = dyn_cast<CharLiteralExpr>(node)) {
    _sema.setBuiltinType(value, BuiltinTypeKind::UInt8);
    return {ComptimeEvaluation::Kind::Static,
            ComptimeValue(value->value())};
  }

  if (auto *variable = dyn_cast<VariableExpr>(node)) {
    if (_frame) {
      if (auto *binding = _frame->lookup(variable->name())) {
        if (binding->type)
          variable->setType(binding->type);
        if (binding->kind == ComptimeBinding::Kind::Pack) {
          (void)_sema.emitError(variable, diag::comptime_pack_escape);
          return {ComptimeEvaluation::Kind::Error, std::nullopt};
        }
        return evaluationFromBinding(*binding);
      }
    }
    if (auto *symbol = _sema.lookupVariable(variable->name())) {
      if (symbol->comptimeValue) {
        if (symbol->isComptimeOnly)
          variable->setComptimeValue(*symbol->comptimeValue);
        return {ComptimeEvaluation::Kind::Static, *symbol->comptimeValue,
                symbol->isComptimeOnly};
      }
      return makeResidual(node);
    }
    if (auto *type = _sema.lookupType(variable->name()))
      return {ComptimeEvaluation::Kind::Static, ComptimeValue(type), true};
    return makeResidual(node);
  }

  if (auto *member = dyn_cast<MemberExpr>(node)) {
    auto *base = dyn_cast<VariableExpr>(member->base().get());
    auto *binding = base && _frame ? _frame->lookup(base->name()) : nullptr;
    if (binding && binding->kind == ComptimeBinding::Kind::Pack) {
      if (member->fieldName() != "length") {
        (void)_sema.emitError(member, diag::comptime_pack_member);
        return {ComptimeEvaluation::Kind::Error, std::nullopt};
      }

      auto *uint64Type = _sema._typeContext.getBuiltinType(
          BuiltinTypeKind::UInt64);
      member->setType(uint64Type);
      return {ComptimeEvaluation::Kind::Static,
              ComptimeValue(static_cast<uint64_t>(binding->elements.size())),
              true};
    }

    auto baseEvaluation = evaluateComptime(member->base().get());
    if (baseEvaluation.kind == ComptimeEvaluation::Kind::Error)
      return baseEvaluation;
    if (baseEvaluation.kind == ComptimeEvaluation::Kind::Static &&
        baseEvaluation.value &&
        baseEvaluation.value->kind() == ComptimeValue::Kind::String) {
      auto *uint64Type = _sema._typeContext.getBuiltinType(
          BuiltinTypeKind::UInt64);
      if (member->fieldName() != "length") {
        (void)_sema.emitError(member, diag::comptime_string_member);
        return {ComptimeEvaluation::Kind::Error, std::nullopt};
      }
      member->setType(uint64Type);
      return {ComptimeEvaluation::Kind::Static,
              ComptimeValue(static_cast<uint64_t>(
                  baseEvaluation.value->stringValue().size())),
              false};
    }

    return makeResidual(node);
  }

  if (auto *index = dyn_cast<IndexExpr>(node)) {
    auto *base = dyn_cast<VariableExpr>(index->base().get());
    auto *binding = base && _frame ? _frame->lookup(base->name()) : nullptr;
    if (binding && binding->kind == ComptimeBinding::Kind::Pack) {
      if (index->indices().size() != 1) {
        (void)_sema.emitError(index, diag::comptime_pack_index);
        return {ComptimeEvaluation::Kind::Error, std::nullopt};
      }

      auto indexEvaluation =
          evaluateComptime(index->indices().front().get());
      if (indexEvaluation.kind == ComptimeEvaluation::Kind::Error)
        return indexEvaluation;
      if (indexEvaluation.kind != ComptimeEvaluation::Kind::Static ||
          !indexEvaluation.value ||
          indexEvaluation.value->kind() != ComptimeValue::Kind::UInt64) {
        (void)_sema.emitError(index->indices().front().get(),
                              diag::comptime_pack_index);
        return {ComptimeEvaluation::Kind::Error, std::nullopt};
      }

      auto elementIndex = indexEvaluation.value->uint64Value();
      if (elementIndex >= binding->elements.size()) {
        (void)_sema.emitError(index, diag::comptime_pack_index_out_of_range);
        return {ComptimeEvaluation::Kind::Error, std::nullopt};
      }

      auto &element = binding->elements[static_cast<size_t>(elementIndex)];
      index->setType(element.type);
      auto result = evaluationFromBinding(element);
      if (result.kind == ComptimeEvaluation::Kind::Static && result.value)
        setComptimeResultType(index, *result.value);
      return result;
    }

    auto baseEvaluation = evaluateComptime(index->base().get());
    if (baseEvaluation.kind == ComptimeEvaluation::Kind::Error)
      return baseEvaluation;
    if (baseEvaluation.kind == ComptimeEvaluation::Kind::Static &&
        baseEvaluation.value &&
        baseEvaluation.value->kind() == ComptimeValue::Kind::String) {
      if (index->indices().size() != 1) {
        (void)_sema.emitError(index, diag::comptime_string_index);
        return {ComptimeEvaluation::Kind::Error, std::nullopt};
      }
      auto indexEvaluation =
          evaluateComptime(index->indices().front().get());
      if (indexEvaluation.kind == ComptimeEvaluation::Kind::Error)
        return indexEvaluation;
      if (indexEvaluation.kind != ComptimeEvaluation::Kind::Static ||
          !indexEvaluation.value ||
          indexEvaluation.value->kind() != ComptimeValue::Kind::UInt64) {
        (void)_sema.emitError(index->indices().front().get(),
                              diag::comptime_string_index);
        return {ComptimeEvaluation::Kind::Error, std::nullopt};
      }

      auto elementIndex = indexEvaluation.value->uint64Value();
      auto stringValue = baseEvaluation.value->stringValue();
      if (elementIndex >= stringValue.size()) {
        (void)_sema.emitError(index, diag::comptime_string_index_out_of_range);
        return {ComptimeEvaluation::Kind::Error, std::nullopt};
      }

      auto *uint8Type = _sema._typeContext.getBuiltinType(
          BuiltinTypeKind::UInt8);
      index->setType(uint8Type);
      return {ComptimeEvaluation::Kind::Static,
              ComptimeValue(static_cast<uint8_t>(
                  static_cast<unsigned char>(stringValue[elementIndex]))),
              false};
    }

    return makeResidual(node);
  }

  if (auto *call = dyn_cast<CallExpr>(node)) {
    auto result = evaluateComptimeCall(call, expectedType);
    if (result.kind == ComptimeEvaluation::Kind::Static)
      setComptimeResultType(call, *result.value);
    return result;
  }

  if (auto *binary = dyn_cast<BinaryExpr>(node)) {
    auto result = evaluateComptimeBinary(binary);
    if (result.kind == ComptimeEvaluation::Kind::Static)
      setComptimeResultType(binary, *result.value);
    return result;
  }

  return makeResidual(node);
}

auto ComptimeSema::evaluateComptimeStringSlice(CallExpr *node)
    -> ComptimeEvaluation {
  auto &arguments = node->expressions();
  if (arguments.size() != 3)
    return makeResidual(node);

  std::vector<ComptimeEvaluation> evaluations;
  for (auto &argument : arguments) {
    auto evaluation = evaluateComptime(argument.get());
    if (evaluation.kind == ComptimeEvaluation::Kind::Error)
      return evaluation;
    if (evaluation.kind != ComptimeEvaluation::Kind::Static ||
        !evaluation.value)
      return makeResidual(node);
    evaluations.push_back(std::move(evaluation));
  }

  if (evaluations[0].value->kind() != ComptimeValue::Kind::String ||
      evaluations[1].value->kind() != ComptimeValue::Kind::UInt64 ||
      evaluations[2].value->kind() != ComptimeValue::Kind::UInt64)
    return makeResidual(node);

  auto stringValue = evaluations[0].value->stringValue();
  auto begin = evaluations[1].value->uint64Value();
  auto end = evaluations[2].value->uint64Value();
  auto stringLength = static_cast<uint64_t>(stringValue.size());
  if (begin > end || end > stringLength) {
    (void)_sema.emitError(node, diag::comptime_string_slice_range);
    return {ComptimeEvaluation::Kind::Error, std::nullopt};
  }

  auto result = std::string(stringValue.substr(
      static_cast<size_t>(begin), static_cast<size_t>(end - begin)));
  LLVM_DEBUG(llvm::dbgs() << "fold comptime String.slice [" << begin << ", "
                          << end << ")\n");
  return {ComptimeEvaluation::Kind::Static, ComptimeValue(result), false};
}

auto ComptimeSema::evaluateComptimeBlock(ComptimeBlockExpr *node,
                                         const Type *expectedType)
    -> ComptimeEvaluation {
  SemaImpl::VariableScope symbolScope(_sema._symbols);
  ComptimeFrame frame(_frame);
  SemaImpl::ComptimeFrameScope frameScope(_sema, &frame);
  ComptimeSema blockSema(_sema, &frame, _loopDepth, _executionState,
                         _isStagedFunctionBody);

  auto control = blockSema.executeComptimeStatements(node->statements());
  if (control.kind == ComptimeControlResult::Kind::Error)
    return {ComptimeEvaluation::Kind::Error, std::nullopt};
  if (control.kind == ComptimeControlResult::Kind::Return) {
    if (!control.value)
      return {ComptimeEvaluation::Kind::Error, std::nullopt};
    auto result = std::move(*control.value);
    if (result.kind == ComptimeEvaluation::Kind::Static && result.value)
      setComptimeResultType(node, *result.value);
    return result;
  }
  if (control.kind != ComptimeControlResult::Kind::Normal)
    return {ComptimeEvaluation::Kind::Error, std::nullopt};

  auto result = blockSema.evaluateComptime(node->result().get(), expectedType);
  if (result.kind == ComptimeEvaluation::Kind::Static)
    setComptimeResultType(node, *result.value);
  return result;
}

auto ComptimeSema::evaluateComptimeCall(CallExpr *node,
                                        const Type *expectedType)
    -> ComptimeEvaluation {
  if (!enterComptimeCall(node))
    return {ComptimeEvaluation::Kind::Error, std::nullopt};
  auto result = evaluateComptimeCallBody(node, expectedType);
  leaveComptimeCall(node);
  return result;
}

auto ComptimeSema::evaluateComptimeCallBody(CallExpr *node,
                                            const Type *expectedType)
    -> ComptimeEvaluation {
  auto &arguments = node->expressions();

  auto evaluateSourceOrResidualCall = [&]() -> ComptimeEvaluation {
    // Only a staged source body may execute a source helper. Ordinary Sema
    // must retain control of calls outside that boundary so expected types,
    // generic inference, and indirect-call resolution are still available.
    if (!_isStagedFunctionBody)
      return makeResidual(node);

    const FunctionDecl *function = nullptr;
    const FunctionSymbol *signature = nullptr;

    if (!node->hasReceiver()) {
      if (node->isIndirectCall() || _sema.lookupVariable(node->name()))
        return makeResidual(node);

      // Generic calls need ordinary Sema to infer arguments and create the
      // concrete source declaration before the interpreter can inspect its
      // body. Calls that are not generic still use the existing direct-source
      // fast path below.
      auto dot = node->name().find('.');
      auto isDottedValueCall =
          dot != std::string_view::npos &&
          _sema.lookupVariable(node->name().substr(0, dot));
      if (!_sema.lookupGenericFunction(node->name()) &&
          !isDottedValueCall) {
        signature = _sema.lookupFunction(node->name());
        function = _sema.lookupFunctionDecl(node->name());
        if (!function || !signature || function->isExtern() ||
            signature->isExtern || function->proto()->isGeneric() ||
            !function->body())
          return makeResidual(node);
      }
    }

    auto normalizeExpression = [&](std::unique_ptr<Expr> &expression)
        -> bool {
      // Pack expansion is owned by ordinary call Sema. Evaluating `args...`
      // here would turn the pack into an escape before expandPackArguments().
      if (expression->isPackExpansion())
        return true;
      auto evaluation = evaluateComptime(expression.get());
      if (evaluation.kind == ComptimeEvaluation::Kind::Error)
        return false;
      if (evaluation.kind == ComptimeEvaluation::Kind::Residual) {
        expression = evaluation.takeResidual();
        return true;
      }
      if (evaluation.isComptimeOnly || !evaluation.value)
        return true;
      auto literal = makeComptimeLiteral(*evaluation.value,
                                          expression->location());
      if (!literal)
        return false;
      expression = std::move(literal);
      return true;
    };

    if (node->hasReceiver()) {
      auto receiver = evaluateComptime(node->receiver().get());
      if (receiver.kind == ComptimeEvaluation::Kind::Error)
        return receiver;
      if (receiver.kind == ComptimeEvaluation::Kind::Residual) {
        node->receiver() = receiver.takeResidual();
      } else {
        if (receiver.kind != ComptimeEvaluation::Kind::Static ||
            !receiver.value ||
            (receiver.isComptimeOnly &&
             receiver.value->kind() != ComptimeValue::Kind::String))
          return makeResidual(node);
        auto literal = makeComptimeLiteral(*receiver.value,
                                            node->receiver()->location());
        if (!literal)
          return {ComptimeEvaluation::Kind::Error, std::nullopt};
        node->receiver() = std::move(literal);
      }
    }

    for (auto &argument : node->expressions())
      if (!normalizeExpression(argument))
        return {ComptimeEvaluation::Kind::Error, std::nullopt};

    if (llvm::failed(_sema.sema(node, expectedType)))
      return {ComptimeEvaluation::Kind::Error, std::nullopt};

    // Method Sema lowers an inherent/trait call to its concrete source
    // function. Re-read both records after that resolution.
    function = _sema.lookupFunctionDecl(node->name());
    signature = _sema.lookupFunction(node->name());
    LLVM_DEBUG(llvm::dbgs() << "comptime call after Sema name=`" << node->name()
                            << "` lowered=" << node->isLoweredMethodCall()
                            << "\n");
    if (!function || !signature || signature->isExtern || !function->body())
      return makeResidual(node);
    if (function->isExtern() || function->proto()->isGeneric())
      return makeResidual(node);

    auto dot = node->name().rfind('.');
    if (node->isLoweredMethodCall() && dot != std::string_view::npos &&
        node->name().substr(dot + 1) == "slice")
      return evaluateComptimeStringSlice(node);

    return evaluateSourceFunctionCall(node, function, signature->type);
  };

  if (!node->hasReceiver()) {
    if (node->name() == builtins::typeOf) {
      if (arguments.size() != 1) {
        auto diagnostic = formatNameSizeDiagnostic(
            diag::func_param, node->name(), 1);
        (void)_sema.emitError(node, diagnostic);
        return {ComptimeEvaluation::Kind::Error, std::nullopt};
      }
      auto *expression = arguments.front().get();
      if (llvm::failed(_sema.sema(expression)))
        return {ComptimeEvaluation::Kind::Error, std::nullopt};
      return {ComptimeEvaluation::Kind::Static,
              ComptimeValue(expression->type()), true};
    }

    // The parser keeps dotted calls package-like until Sema can prove that the
    // left side is a receiver. Reflection needs the same delayed decision.
    auto name = std::string(node->name());
    auto dot = name.rfind('.');
    if (dot != std::string::npos) {
      auto receiver = createMemberAccessChain(
          node->location(), std::string_view(name).substr(0, dot));
      auto receiverValue = evaluateComptime(receiver.get());
      if (receiverValue.kind == ComptimeEvaluation::Kind::Error)
        return receiverValue;
      if (receiverValue.kind == ComptimeEvaluation::Kind::Static &&
          receiverValue.value->kind() == ComptimeValue::Kind::Type) {
        LLVM_DEBUG(llvm::dbgs()
                   << "resolve comptime method call `" << name << "`\n");
        node->setReceiver(std::move(receiver),
                          std::string_view(name).substr(dot + 1));
        return evaluateComptimeCall(node, expectedType);
      }
    }

    return evaluateSourceOrResidualCall();
  }

  auto receiver = evaluateComptime(node->receiver().get());
  if (receiver.kind == ComptimeEvaluation::Kind::Static && receiver.value &&
      receiver.value->kind() == ComptimeValue::Kind::Type) {
    auto *type = receiver.value->type();
    auto name = node->name();
    auto requireNoArguments = [&]() -> bool {
      if (arguments.empty())
        return true;
      auto diagnostic = formatNameSizeDiagnostic(diag::func_param, name, 0);
      (void)_sema.emitError(node, diagnostic);
      return false;
    };

    if (name == "isBool" || name == "isUInt8" || name == "isUInt64" ||
        name == "isInt64" || name == "isInteger" || name == "isFloat32" ||
        name == "isFloat64" || name == "isArray" || name == "isStruct" ||
        name == "isObject") {
      if (!requireNoArguments())
        return {ComptimeEvaluation::Kind::Error, std::nullopt};
      if (name == "isBool")
        return {ComptimeEvaluation::Kind::Static,
                ComptimeValue(isBoolType(type)), true};
      if (name == "isUInt8")
        return {ComptimeEvaluation::Kind::Static,
                ComptimeValue(isUInt8Type(type)), true};
      if (name == "isUInt64")
        return {ComptimeEvaluation::Kind::Static,
                ComptimeValue(isUInt64Type(type)), true};
      if (name == "isInt64")
        return {ComptimeEvaluation::Kind::Static,
                ComptimeValue(isInt64Type(type)), true};
      if (name == "isInteger")
        return {ComptimeEvaluation::Kind::Static,
                ComptimeValue(isIntegerType(type)), true};
      if (name == "isFloat32")
        return {ComptimeEvaluation::Kind::Static,
                ComptimeValue(isFloat32Type(type)), true};
      if (name == "isFloat64")
        return {ComptimeEvaluation::Kind::Static,
                ComptimeValue(isFloat64Type(type)), true};
      if (name == "isArray")
        return {ComptimeEvaluation::Kind::Static,
                ComptimeValue(isArrayType(type)), true};
      if (name == "isObject")
        return {ComptimeEvaluation::Kind::Static,
                ComptimeValue(isSourceObjectType(type)), true};
      return {ComptimeEvaluation::Kind::Static,
              ComptimeValue(getStructType(type) != nullptr), true};
    }

    if (name == "arrayElementType" || name == "arrayLeafElementType" ||
        name == "arrayLength" || name == "arrayRank") {
      if (!requireNoArguments())
        return {ComptimeEvaluation::Kind::Error, std::nullopt};
      auto *arrayType = getArrayType(type);
      if (!arrayType) {
        (void)_sema.emitError(node, diag::invalid_reflection_query);
        return {ComptimeEvaluation::Kind::Error, std::nullopt};
      }
      if (name == "arrayElementType")
        return {ComptimeEvaluation::Kind::Static,
                ComptimeValue(arrayType->elementType()), true};
      if (name == "arrayLeafElementType")
        return {ComptimeEvaluation::Kind::Static,
                ComptimeValue(getArrayLeafElementType(type)), true};
      if (name == "arrayLength")
        return {ComptimeEvaluation::Kind::Static,
                ComptimeValue(arrayType->size()), true};
      return {ComptimeEvaluation::Kind::Static,
              ComptimeValue(static_cast<uint64_t>(getArrayShape(type).size())),
              true};
    }

    auto diagnostic = formatNameDiagnostic(diag::undefined_func, name);
    (void)_sema.emitError(node, diagnostic);
    return {ComptimeEvaluation::Kind::Error, std::nullopt};
  }

  return evaluateSourceOrResidualCall();
}

auto ComptimeSema::evaluateSourceFunctionCall(
    CallExpr *node, const FunctionDecl *function,
    const FunctionType *signature) -> ComptimeEvaluation {
  std::vector<ComptimeEvaluation> arguments;
  for (auto &argument : node->expressions()) {
    auto evaluation = evaluateComptime(argument.get());
    if (evaluation.kind == ComptimeEvaluation::Kind::Error)
      return evaluation;
    if (evaluation.kind != ComptimeEvaluation::Kind::Static ||
        !evaluation.value || evaluation.isComptimeOnly)
      return makeResidual(node);
    arguments.push_back(std::move(evaluation));
  }

  ComptimeFrame functionFrame;
  SemaImpl::VariableScope functionScope(_sema._symbols);
  SemaImpl::ComptimeFrameScope frameScope(_sema, &functionFrame);
  SemaImpl::FunctionReturnTypeScope returnTypeScope(
      _sema._currentFunctionReturnType, signature->returnType());

  auto functionName = function->proto()->id()->name();
  auto packageName = packageNameOf(functionName);
  if (auto package = _sema._functionPackages.find(std::string(functionName));
      package != _sema._functionPackages.end())
    packageName = package->second;
  SemaImpl::PackageScope packageScope(_sema._currentPackageName, packageName);

  size_t argumentIndex = 0;
  std::vector<ComptimeBinding> concretePackElements;
  for (auto &parameter : function->proto()->parameters()) {
    auto *parameterType = parameter->type();
    if (!parameterType)
      return {ComptimeEvaluation::Kind::Error, std::nullopt};

    std::optional<ComptimeValue> value;
    bool isComptimeOnly = false;
    if (parameter->isComptime()) {
      if (!parameter->comptimeValue())
        return {ComptimeEvaluation::Kind::Error, std::nullopt};
      value = parameter->comptimeValue();
      isComptimeOnly = true;
    } else {
      if (argumentIndex >= arguments.size())
        return {ComptimeEvaluation::Kind::Error, std::nullopt};
      value = arguments[argumentIndex++].value;
      isComptimeOnly = arguments[argumentIndex - 1].isComptimeOnly;
    }

    if (!value ||
        llvm::failed(_sema.declareVariable(
            parameter->variable()->name(), parameterType,
            /*isConstBinding=*/!parameter->canMutateObject(),
            parameter->canMutateObject(), value, isComptimeOnly)))
      return {ComptimeEvaluation::Kind::Error, std::nullopt};

    if (function->proto()->hasConcretePack()) {
      for (auto &elementName : function->proto()->concretePackElements()) {
        if (elementName != parameter->variable()->name())
          continue;
        concretePackElements.push_back(ComptimeBinding::staticValue(
            *value, parameterType, !parameter->canMutateObject(),
            isComptimeOnly));
        break;
      }
    }

    if (!functionFrame.bind(
            parameter->variable()->name(),
            ComptimeBinding::staticValue(
                *value, parameterType, !parameter->canMutateObject(),
                isComptimeOnly)))
      return {ComptimeEvaluation::Kind::Error, std::nullopt};
  }

  if (argumentIndex != arguments.size())
    return {ComptimeEvaluation::Kind::Error, std::nullopt};

  if (function->proto()->hasConcretePack() &&
      !functionFrame.bind(function->proto()->concretePackName(),
                          ComptimeBinding::pack(
                              std::move(concretePackElements))))
    return {ComptimeEvaluation::Kind::Error, std::nullopt};

  auto body = substituteBlockExpr(function->body().get(), {});
  unsigned loopDepth = 0;
  ComptimeSema functionSema(_sema, &functionFrame, &loopDepth,
                           _executionState, /*isStagedFunctionBody=*/true);
  auto control = functionSema.executeComptimeBlock(body.get());
  if (control.kind == ComptimeControlResult::Kind::Return && control.value)
    return std::move(*control.value);
  if (control.kind == ComptimeControlResult::Kind::Error)
    return {ComptimeEvaluation::Kind::Error, std::nullopt};

  (void)_sema.emitError(node, diag::expected_comptime_value);
  return {ComptimeEvaluation::Kind::Error, std::nullopt};
}

auto ComptimeSema::evaluateComptimeBinary(BinaryExpr *node)
    -> ComptimeEvaluation {
  using EvaluationKind = ComptimeEvaluation::Kind;
  using Operator = BinaryExpr::Operator;

  auto lhs = evaluateComptime(node->lhs().get());
  if (lhs.kind == EvaluationKind::Error)
    return lhs;

  auto op = node->opEnum();
  if (lhs.kind == EvaluationKind::Static && lhs.value &&
      lhs.value->kind() == ComptimeValue::Kind::Bool) {
    if (op == Operator::And && !lhs.value->boolValue())
      return {EvaluationKind::Static, ComptimeValue(false),
              lhs.isComptimeOnly};
    if (op == Operator::Or && lhs.value->boolValue())
      return {EvaluationKind::Static, ComptimeValue(true),
              lhs.isComptimeOnly};
  }

  auto rhs = evaluateComptime(node->rhs().get());
  if (rhs.kind == EvaluationKind::Error)
    return rhs;

  auto isRuntimeMaterializable = [](const ComptimeEvaluation &evaluation) {
    return evaluation.kind != EvaluationKind::Static ||
           (evaluation.value &&
            evaluation.value->kind() != ComptimeValue::Kind::Type);
  };

  if (lhs.kind != EvaluationKind::Static ||
      rhs.kind != EvaluationKind::Static) {
    if (!isRuntimeMaterializable(lhs) || !isRuntimeMaterializable(rhs)) {
      (void)_sema.emitError(node, diag::expected_comptime_value);
      return {EvaluationKind::Error, std::nullopt};
    }

    auto materialize = [&](ComptimeEvaluation &evaluation,
                           llvm::SMLoc location) -> std::unique_ptr<Expr> {
      if (evaluation.kind == EvaluationKind::Residual)
        return evaluation.takeResidual();
      if (evaluation.value)
        return makeComptimeLiteral(*evaluation.value, location);
      return nullptr;
    };

    auto residual = std::make_unique<BinaryExpr>(
        node->location(), op,
        materialize(lhs, node->lhs()->location()),
        materialize(rhs, node->rhs()->location()));
    if (!residual->lhs() || !residual->rhs()) {
      (void)_sema.emitError(node, diag::expected_comptime_value);
      return {EvaluationKind::Error, std::nullopt};
    }
    residual->setType(node->type());
    LLVM_DEBUG(llvm::dbgs()
               << "compose static and residual binary expression\n");
    return {EvaluationKind::Residual, std::nullopt, false,
            std::move(residual)};
  }

  if (!lhs.value || !rhs.value)
    return {EvaluationKind::Error, std::nullopt};

  auto isComptimeOnly = lhs.isComptimeOnly || rhs.isComptimeOnly;
  auto cannotEvaluate = [&]() -> ComptimeEvaluation {
    if (!isComptimeOnly)
      return makeResidual(node);
    (void)_sema.emitError(node, diag::expected_comptime_value);
    return {EvaluationKind::Error, std::nullopt};
  };

  if (op == Operator::EQ || op == Operator::NEQ) {
    if (lhs.value->kind() != rhs.value->kind())
      return cannotEvaluate();

    bool equal = false;
    switch (lhs.value->kind()) {
    case ComptimeValue::Kind::Type:
      equal = sameType(lhs.value->type(), rhs.value->type());
      break;
    case ComptimeValue::Kind::Bool:
      equal = lhs.value->boolValue() == rhs.value->boolValue();
      break;
    case ComptimeValue::Kind::UInt8:
      equal = lhs.value->uint8Value() == rhs.value->uint8Value();
      break;
    case ComptimeValue::Kind::UInt64:
      equal = lhs.value->uint64Value() == rhs.value->uint64Value();
      break;
    case ComptimeValue::Kind::String:
      equal = lhs.value->stringValue() == rhs.value->stringValue();
      break;
    }
    return {EvaluationKind::Static,
            ComptimeValue(op == Operator::EQ ? equal : !equal),
            isComptimeOnly};
  }

  if (lhs.value->kind() == ComptimeValue::Kind::Bool &&
      rhs.value->kind() == ComptimeValue::Kind::Bool) {
    if (op == Operator::And)
      return {EvaluationKind::Static,
              ComptimeValue(lhs.value->boolValue() &&
                            rhs.value->boolValue()),
              isComptimeOnly};
    if (op == Operator::Or)
      return {EvaluationKind::Static,
              ComptimeValue(lhs.value->boolValue() ||
                            rhs.value->boolValue()),
              isComptimeOnly};
    return cannotEvaluate();
  }

  if (lhs.value->kind() == ComptimeValue::Kind::String &&
      rhs.value->kind() == ComptimeValue::Kind::String) {
    if (op != Operator::Add)
      return cannotEvaluate();
    auto result = std::string(lhs.value->stringValue());
    result += rhs.value->stringValue();
    return {EvaluationKind::Static, ComptimeValue(result), false};
  }

  if (lhs.value->kind() != ComptimeValue::Kind::UInt64 ||
      rhs.value->kind() != ComptimeValue::Kind::UInt64)
    return cannotEvaluate();

  auto left = lhs.value->uint64Value();
  auto right = rhs.value->uint64Value();
  switch (op) {
  case Operator::Add:
    return {EvaluationKind::Static, ComptimeValue(left + right),
            isComptimeOnly};
  case Operator::Mul:
    return {EvaluationKind::Static, ComptimeValue(left * right),
            isComptimeOnly};
  case Operator::Diff:
    return {EvaluationKind::Static, ComptimeValue(left - right),
            isComptimeOnly};
  case Operator::Div:
    if (right != 0)
      return {EvaluationKind::Static, ComptimeValue(left / right),
              isComptimeOnly};
    break;
  case Operator::Rem:
    if (right != 0)
      return {EvaluationKind::Static, ComptimeValue(left % right),
              isComptimeOnly};
    break;
  case Operator::ShiftLeft:
  case Operator::ShiftRight:
  case Operator::BitAnd:
  case Operator::BitOr:
  case Operator::BitXor:
    return cannotEvaluate();
  case Operator::LT:
    return {EvaluationKind::Static, ComptimeValue(left < right),
            isComptimeOnly};
  case Operator::LE:
    return {EvaluationKind::Static, ComptimeValue(left <= right),
            isComptimeOnly};
  case Operator::GT:
    return {EvaluationKind::Static, ComptimeValue(left > right),
            isComptimeOnly};
  case Operator::GE:
    return {EvaluationKind::Static, ComptimeValue(left >= right),
            isComptimeOnly};
  case Operator::And:
  case Operator::Or:
  case Operator::EQ:
  case Operator::NEQ:
    break;
  }

  return cannotEvaluate();
}

} // namespace mulberry
