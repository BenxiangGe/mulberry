//===--- SemaStatements.cpp - Statement semantic analysis ----------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "SemaStatements.h"
#include "SemaComptime.h"
#include "SemaExpressions.h"
#include "SemaImpl.h"
#include "SemaSupport.h"
#include "llvm/Support/Debug.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "Sema"

namespace mulberry {
using llvm::cast;

auto StatementSema::sema(BlockExpr *node) -> llvm::LogicalResult {
  SemaImpl::VariableScope blockScope(_sema._symbols);
  for (auto &statement : node->statements())
    if (llvm::failed(sema(statement.get())))
      return failure();

  node->setType(_sema._typeContext.getBuiltinType(BuiltinTypeKind::Unit));
  return success();
}

auto StatementSema::checkMatchPattern(
    DataPattern *pattern, const DataType *dataType,
    std::vector<bool> &covered, const DataConstructor *&constructor)
    -> llvm::LogicalResult {
  auto &constructors = dataType->constructors();
  std::string resolvedName;
  auto *symbol = _sema.lookupDataConstructor(pattern->constructorName(),
                                              resolvedName);
  if (!symbol)
    return _sema.emitError(pattern, diag::undefined_data_constructor);
  if (symbol->decl->name() != dataType->declarationName() ||
      symbol->index >= constructors.size())
    return _sema.emitError(pattern, diag::match_constructor_type);
  if (covered[symbol->index])
    return _sema.emitError(pattern, diag::duplicate_match_constructor);

  constructor = &constructors[symbol->index];
  if (pattern->bindings().size() != constructor->payloadTypes().size())
    return _sema.emitError(pattern, diag::match_binding_count);

  pattern->setConstructorName(resolvedName);
  pattern->setConstructorIndex(symbol->index);
  covered[symbol->index] = true;
  for (size_t i = 0; i < pattern->bindings().size(); ++i)
    pattern->bindings()[i]->setType(constructor->payloadTypes()[i]);
  LLVM_DEBUG(llvm::dbgs() << "bind match pattern `" << resolvedName
                          << "` as " << formatType(dataType) << " tag "
                          << symbol->index << "\n");
  return success();
}

auto StatementSema::declareMatchPatternBindings(
    DataPattern *pattern, const DataConstructor *constructor)
    -> llvm::LogicalResult {
  for (size_t i = 0; i < pattern->bindings().size(); ++i) {
    auto *binding = pattern->bindings()[i].get();
    auto *payloadType = constructor->payloadTypes()[i];
    if (llvm::failed(_sema.declareVariable(binding->name(), payloadType,
                                           /*isConstBinding=*/true,
                                           /*canMutateObject=*/false)))
      return _sema.emitError(binding, diag::redefinition_var);
  }
  return success();
}

auto StatementSema::checkExhaustiveMatch(
    const Node *node, const std::vector<bool> &covered)
    -> llvm::LogicalResult {
  for (auto isCovered : covered)
    if (!isCovered)
      return _sema.emitError(node, diag::non_exhaustive_match);
  return success();
}

auto StatementSema::sema(IfStat *node) -> llvm::LogicalResult {
  auto condition = ComptimeSema(_sema).evaluateComptime(
      node->conditionExpr().get());
  if (condition.kind == ComptimeEvaluation::Kind::Error)
    return failure();

  if (condition.kind == ComptimeEvaluation::Kind::Static) {
    if (condition.value->kind() != ComptimeValue::Kind::Bool)
      return _sema.emitError(node->conditionExpr().get(), diag::expected_bool);

    auto value = condition.value->boolValue();
    node->setComptimeValue(value);
    LLVM_DEBUG(llvm::dbgs() << "evaluate if condition at comptime as `"
                            << (value ? "true" : "false") << "`\n");
    if (value)
      return _sema.sema(node->thenBlock().get());
    if (node->hasElseBlock())
      return _sema.sema(node->elseBlock().get());
    return success();
  }

  auto conditionExpr = node->conditionExpr().get();
  if (llvm::failed(_sema.sema(conditionExpr)))
    return failure();
  if (!isBoolType(conditionExpr->type()))
    return _sema.emitError(conditionExpr, diag::expected_bool);

  auto thenBlock = node->thenBlock().get();
  if (llvm::failed(_sema.sema(thenBlock)))
    return failure();

  if (!node->hasElseBlock())
    return success();

  auto elseBlock = node->elseBlock().get();
  if (llvm::failed(_sema.sema(elseBlock)))
    return failure();

  return success();
}

auto StatementSema::sema(MatchStat *node) -> llvm::LogicalResult {
  auto *value = node->value().get();
  if (llvm::failed(_sema.sema(value)))
    return failure();

  auto *dataType = getDataType(value->type());
  if (!dataType)
    return _sema.emitError(value, diag::match_expected_data_type);

  std::vector<bool> covered(dataType->constructors().size(), false);
  for (auto &arm : node->arms()) {
    auto *pattern = arm->pattern().get();
    const DataConstructor *constructor = nullptr;
    if (llvm::failed(checkMatchPattern(pattern, dataType, covered,
                                       constructor)))
      return failure();

    SemaImpl::VariableScope armScope(_sema._symbols);
    if (llvm::failed(declareMatchPatternBindings(pattern, constructor)) ||
        llvm::failed(_sema.sema(arm->bodyBlock().get())))
      return failure();
  }

  return checkExhaustiveMatch(node, covered);
}

auto StatementSema::sema(WhileStat *node) -> llvm::LogicalResult {
  auto conditionExpr = node->conditionExpr().get();
  if (llvm::failed(_sema.sema(conditionExpr)))
    return failure();
  if (!isBoolType(conditionExpr->type()))
    return _sema.emitError(conditionExpr, diag::expected_bool);

  auto bodyBlock = node->bodyBlock().get();
  SemaImpl::WhileScope whileScope(_sema._whileDepth);
  if (llvm::failed(_sema.sema(bodyBlock)))
    return failure();

  return success();
}

auto StatementSema::sema(BreakStat *node) -> llvm::LogicalResult {
  if (_sema._whileDepth == 0)
    return _sema.emitError(node, diag::loop_control_outside_loop);
  return success();
}

auto StatementSema::sema(ContinueStat *node) -> llvm::LogicalResult {
  if (_sema._whileDepth == 0)
    return _sema.emitError(node, diag::loop_control_outside_loop);
  return success();
}

auto StatementSema::sema(ForStat *node) -> llvm::LogicalResult {
  if (llvm::failed(_sema.sema(node->startExpr().get())) ||
      llvm::failed(_sema.sema(node->endExpr().get())))
    return failure();

  if (!isUInt64Type(node->startExpr()->type()))
    return _sema.emitError(node->startExpr().get(), diag::mismatch_type);
  if (!isUInt64Type(node->endExpr()->type()))
    return _sema.emitError(node->endExpr().get(), diag::mismatch_type);

  SemaImpl::VariableScope loopScope(_sema._symbols);
  auto *uint64Type = _sema._typeContext.getBuiltinType(
      BuiltinTypeKind::UInt64);
  if (llvm::failed(_sema.declareVariable(
          node->variableName(), uint64Type,
          /*isConstBinding=*/true,
          /*canMutateObject=*/false)))
    return _sema.emitError(node, diag::redefinition_var);

  auto bodyBlock = node->bodyBlock().get();
  if (llvm::failed(_sema.sema(bodyBlock)))
    return failure();

  return success();
}

auto StatementSema::sema(Stat *node) -> llvm::LogicalResult {
  switch (node->getKind()) {
  case Stat::Stat_VariableDecl:
    return sema(cast<VariableStat>(node));
  case Stat::Stat_Expression:
    return sema(cast<ExprStat>(node));
  case Stat::Stat_If:
    return sema(cast<IfStat>(node));
  case Stat::Stat_Match:
    return sema(cast<MatchStat>(node));
  case Stat::Stat_While:
    return sema(cast<WhileStat>(node));
  case Stat::Stat_For:
    return sema(cast<ForStat>(node));
  case Stat::Stat_Break:
    return sema(cast<BreakStat>(node));
  case Stat::Stat_Continue:
    return sema(cast<ContinueStat>(node));
  case Stat::Stat_Return:
    return sema(cast<ReturnStat>(node));
  }
}

auto StatementSema::sema(VariableStat *node) -> llvm::LogicalResult {
  auto var = node->variable().get();
  auto &initExpr = node->init();

  const Type *declaredType = nullptr;
  if (node->hasExplicitType()) {
    declaredType = _sema.checkType(node->typeNode(), SemaImpl::UnitPolicy::Allow);
    if (!declaredType)
      return failure();
  }

  std::optional<ComptimeValue> comptimeValue;
  auto initializerWasTyped = false;
  if (node->isConstBinding() && declaredType && isIntegerType(declaredType)) {
    if (llvm::failed(_sema.semaExpected(initExpr, declaredType)))
      return failure();
    initializerWasTyped = true;
  }

  if (node->isConstBinding()) {
    auto evaluation = ComptimeSema(_sema).evaluateComptime(initExpr.get());
    if (evaluation.kind == ComptimeEvaluation::Kind::Error)
      return failure();
    if (evaluation.kind == ComptimeEvaluation::Kind::Residual) {
      initExpr = evaluation.takeResidual();
      initializerWasTyped = false;
      LLVM_DEBUG(llvm::dbgs()
                 << "replace const initializer with residual expression\n");
    }
    if (evaluation.kind == ComptimeEvaluation::Kind::Static)
      comptimeValue = evaluation.value;

    // Ordinary consts keep their runtime binding and only cache the value.
    // Reflection-derived initializers cannot enter MLIR, so their declaration
    // is comptime-only.
    if (evaluation.kind == ComptimeEvaluation::Kind::Static &&
        evaluation.isComptimeOnly) {
      auto *valueType = ComptimeSema(_sema).comptimeRuntimeType(
          *evaluation.value);
      if (node->hasExplicitType()) {
        if (!valueType || !sameType(declaredType, valueType))
          return _sema.emitError(initExpr.get(), diag::mismatch_type);
        valueType = declaredType;
      }

      node->setType(valueType);
      node->setComptimeValue(*evaluation.value);
      if (llvm::failed(_sema.declareVariable(
              var->name(), valueType,
              /*isConstBinding=*/true,
              /*canMutateObject=*/false, evaluation.value,
              /*isComptimeOnly=*/true)))
        return _sema.emitError(var, diag::redefinition_var);
      return success();
    }
  }

  const Type *varType = nullptr;
  if (declaredType) {
    varType = declaredType;
    if (!initializerWasTyped &&
        llvm::failed(_sema.semaExpected(initExpr, varType)))
      return failure();
  } else {
    if (llvm::failed(_sema.sema(initExpr.get())))
      return failure();
    varType = initExpr->type();
    if (!varType)
      return _sema.emitError(initExpr.get(), diag::mismatch_type);
  }
  node->setType(varType);

  auto canMutateObject = node->canMutateObject();
  if (canMutateObject && isMutableSourceObjectType(varType) &&
      !ExpressionSema(_sema).canMutateObjectReference(initExpr.get()))
    return _sema.emitError(initExpr.get(), diag::readonly_to_mutable_binding);
  if (llvm::failed(_sema.declareVariable(
          var->name(), varType, node->isConstBinding(), canMutateObject,
          std::move(comptimeValue))))
    return _sema.emitError(var, diag::redefinition_var);
  return success();
}

auto StatementSema::sema(ExprStat *node) -> llvm::LogicalResult {
  return _sema.sema(node->expression().get());
}

auto StatementSema::sema(ReturnStat *node) -> llvm::LogicalResult {
  if (!_sema._currentFunctionReturnType)
    return _sema.emitError(node, diag::return_outside_function);

  if (!node->hasExpression()) {
    if (!isUnitType(_sema._currentFunctionReturnType))
      return _sema.emitError(node, diag::wrong_return_type);
    return success();
  }

  auto &expression = node->expression();
  if (getPtrType(_sema._currentFunctionReturnType)) {
    // The internal Ptr<T> -> Ptr<U> cast is valid only at a declared return
    // boundary. Keep ordinary expected-value contexts type-safe.
    if (llvm::failed(_sema.sema(expression.get())))
      return failure();
  } else if (llvm::failed(
                 _sema.semaExpected(expression, _sema._currentFunctionReturnType))) {
    return failure();
  }
  if (!sameReturnType(_sema._currentFunctionReturnType, expression->type()))
    return _sema.emitError(expression.get(), diag::wrong_return_type);
  return success();
}

} // namespace mulberry
