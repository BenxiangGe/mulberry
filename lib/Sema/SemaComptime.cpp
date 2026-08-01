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

#undef DEBUG_TYPE
#define DEBUG_TYPE "Sema"

namespace mulberry {
using llvm::dyn_cast;

auto ComptimeSema::comptimeRuntimeType(const ComptimeValue &value)
    -> const Type * {
  switch (value.kind()) {
  case ComptimeValue::Kind::Type:
    return nullptr;
  case ComptimeValue::Kind::Bool:
    return _sema._typeContext.getBuiltinType(BuiltinTypeKind::Bool);
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

auto ComptimeSema::evaluateComptime(Expr *node) -> ComptimeEvaluation {
  if (auto *block = dyn_cast<ComptimeBlockExpr>(node))
    return evaluateComptimeBlock(block);

  if (auto *typeExpr = dyn_cast<TypeInfoExpr>(node)) {
    auto *type = _sema.resolveType(typeExpr->typeNode());
    if (!type)
      return {ComptimeEvaluation::Kind::Error, std::nullopt};
    return {ComptimeEvaluation::Kind::Value, ComptimeValue(type), true};
  }

  if (auto *value = dyn_cast<BoolLiteralExpr>(node)) {
    _sema.setBuiltinType(value, BuiltinTypeKind::Bool);
    return {ComptimeEvaluation::Kind::Value,
            ComptimeValue(value->value())};
  }

  if (auto *value = dyn_cast<IntegerLiteralExpr>(node)) {
    if (!value->hasValidSpelling()) {
      (void)_sema.emitError(value, diag::invalid_integer_literal);
      return {ComptimeEvaluation::Kind::Error, std::nullopt};
    }
    if (isIntegerType(value->type()))
      return {ComptimeEvaluation::Kind::Runtime, std::nullopt};
    auto uint64Value = value->getUInt64Value();
    if (!uint64Value) {
      (void)_sema.emitError(value, diag::integer_literal_overflows);
      return {ComptimeEvaluation::Kind::Error, std::nullopt};
    }
    _sema.setBuiltinType(value, BuiltinTypeKind::UInt64);
    return {ComptimeEvaluation::Kind::Value,
            ComptimeValue(*uint64Value)};
  }

  if (auto *value = dyn_cast<StringLiteralExpr>(node))
    return {ComptimeEvaluation::Kind::Value,
            ComptimeValue(value->value())};

  if (auto *variable = dyn_cast<VariableExpr>(node)) {
    if (auto *symbol = _sema.lookupVariable(variable->name())) {
      if (symbol->comptimeValue) {
        if (symbol->isComptimeOnly)
          variable->setComptimeValue(*symbol->comptimeValue);
        return {ComptimeEvaluation::Kind::Value, *symbol->comptimeValue,
                symbol->isComptimeOnly};
      }
      return {ComptimeEvaluation::Kind::Runtime, std::nullopt};
    }
    if (auto *type = _sema.lookupType(variable->name()))
      return {ComptimeEvaluation::Kind::Value, ComptimeValue(type), true};
    return {ComptimeEvaluation::Kind::Runtime, std::nullopt};
  }

  if (auto *call = dyn_cast<CallExpr>(node)) {
    auto result = evaluateComptimeCall(call);
    if (result.kind == ComptimeEvaluation::Kind::Value)
      setComptimeResultType(call, *result.value);
    return result;
  }

  if (auto *binary = dyn_cast<BinaryExpr>(node)) {
    auto result = evaluateComptimeBinary(binary);
    if (result.kind == ComptimeEvaluation::Kind::Value)
      setComptimeResultType(binary, *result.value);
    return result;
  }

  return {ComptimeEvaluation::Kind::Runtime, std::nullopt};
}

auto ComptimeSema::evaluateComptimeBlock(ComptimeBlockExpr *node)
    -> ComptimeEvaluation {
  SemaImpl::VariableScope blockScope(_sema._symbols);
  for (auto &statement : node->statements()) {
    if (llvm::failed(_sema.sema(statement.get())))
      return {ComptimeEvaluation::Kind::Error, std::nullopt};
  }

  auto result = evaluateComptime(node->result().get());
  if (result.kind == ComptimeEvaluation::Kind::Value)
    setComptimeResultType(node, *result.value);
  return result;
}

auto ComptimeSema::evaluateComptimeCall(CallExpr *node)
    -> ComptimeEvaluation {
  auto &arguments = node->expressions();
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
      return {ComptimeEvaluation::Kind::Value,
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
      if (receiverValue.kind == ComptimeEvaluation::Kind::Value &&
          receiverValue.value->kind() == ComptimeValue::Kind::Type) {
        LLVM_DEBUG(llvm::dbgs()
                   << "resolve comptime method call `" << name << "`\n");
        node->setReceiver(std::move(receiver),
                          std::string_view(name).substr(dot + 1));
        return evaluateComptimeCall(node);
      }
    }

    return {ComptimeEvaluation::Kind::Runtime, std::nullopt};
  }

  auto receiver = evaluateComptime(node->receiver().get());
  if (receiver.kind != ComptimeEvaluation::Kind::Value)
    return receiver;
  if (receiver.value->kind() != ComptimeValue::Kind::Type)
    return {ComptimeEvaluation::Kind::Runtime, std::nullopt};

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
      name == "isInteger" || name == "isFloat32" || name == "isArray" ||
      name == "isStruct" || name == "isObject") {
    if (!requireNoArguments())
      return {ComptimeEvaluation::Kind::Error, std::nullopt};
    if (name == "isBool")
      return {ComptimeEvaluation::Kind::Value,
              ComptimeValue(isBoolType(type)), true};
    if (name == "isUInt8")
      return {ComptimeEvaluation::Kind::Value,
              ComptimeValue(isUInt8Type(type)), true};
    if (name == "isUInt64")
      return {ComptimeEvaluation::Kind::Value,
              ComptimeValue(isUInt64Type(type)), true};
    if (name == "isInteger")
      return {ComptimeEvaluation::Kind::Value,
              ComptimeValue(isIntegerType(type)), true};
    if (name == "isFloat32")
      return {ComptimeEvaluation::Kind::Value,
              ComptimeValue(isFloat32Type(type)), true};
    if (name == "isArray")
      return {ComptimeEvaluation::Kind::Value,
              ComptimeValue(isArrayType(type)), true};
    if (name == "isObject")
      return {ComptimeEvaluation::Kind::Value,
              ComptimeValue(isSourceObjectType(type)), true};
    return {ComptimeEvaluation::Kind::Value,
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
      return {ComptimeEvaluation::Kind::Value,
              ComptimeValue(arrayType->elementType()), true};
    if (name == "arrayLeafElementType")
      return {ComptimeEvaluation::Kind::Value,
              ComptimeValue(getArrayLeafElementType(type)), true};
    if (name == "arrayLength")
      return {ComptimeEvaluation::Kind::Value,
              ComptimeValue(arrayType->size()), true};
    return {ComptimeEvaluation::Kind::Value,
            ComptimeValue(static_cast<uint64_t>(getArrayShape(type).size())),
            true};
  }

  auto diagnostic = formatNameDiagnostic(diag::undefined_func, name);
  (void)_sema.emitError(node, diagnostic);
  return {ComptimeEvaluation::Kind::Error, std::nullopt};
}

auto ComptimeSema::evaluateComptimeBinary(BinaryExpr *node)
    -> ComptimeEvaluation {
  using EvaluationKind = ComptimeEvaluation::Kind;
  using Operator = BinaryExpr::Operator;

  auto lhs = evaluateComptime(node->lhs().get());
  if (lhs.kind != EvaluationKind::Value)
    return lhs;

  auto op = node->opEnum();
  if (lhs.value->kind() == ComptimeValue::Kind::Bool) {
    if (op == Operator::And && !lhs.value->boolValue())
      return {EvaluationKind::Value, ComptimeValue(false),
              lhs.isComptimeOnly};
    if (op == Operator::Or && lhs.value->boolValue())
      return {EvaluationKind::Value, ComptimeValue(true),
              lhs.isComptimeOnly};
  }

  auto rhs = evaluateComptime(node->rhs().get());
  if (rhs.kind != EvaluationKind::Value)
    return rhs;
  auto isComptimeOnly = lhs.isComptimeOnly || rhs.isComptimeOnly;
  auto cannotEvaluate = [&]() -> ComptimeEvaluation {
    if (!isComptimeOnly)
      return {EvaluationKind::Runtime, std::nullopt};
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
    case ComptimeValue::Kind::UInt64:
      equal = lhs.value->uint64Value() == rhs.value->uint64Value();
      break;
    case ComptimeValue::Kind::String:
      equal = lhs.value->stringValue() == rhs.value->stringValue();
      break;
    }
    return {EvaluationKind::Value,
            ComptimeValue(op == Operator::EQ ? equal : !equal),
            isComptimeOnly};
  }

  if (lhs.value->kind() == ComptimeValue::Kind::Bool &&
      rhs.value->kind() == ComptimeValue::Kind::Bool) {
    if (op == Operator::And)
      return {EvaluationKind::Value,
              ComptimeValue(lhs.value->boolValue() &&
                            rhs.value->boolValue()),
              isComptimeOnly};
    if (op == Operator::Or)
      return {EvaluationKind::Value,
              ComptimeValue(lhs.value->boolValue() ||
                            rhs.value->boolValue()),
              isComptimeOnly};
    return cannotEvaluate();
  }

  if (lhs.value->kind() != ComptimeValue::Kind::UInt64 ||
      rhs.value->kind() != ComptimeValue::Kind::UInt64)
    return cannotEvaluate();

  auto left = lhs.value->uint64Value();
  auto right = rhs.value->uint64Value();
  switch (op) {
  case Operator::Add:
    return {EvaluationKind::Value, ComptimeValue(left + right),
            isComptimeOnly};
  case Operator::Mul:
    return {EvaluationKind::Value, ComptimeValue(left * right),
            isComptimeOnly};
  case Operator::Diff:
    return {EvaluationKind::Value, ComptimeValue(left - right),
            isComptimeOnly};
  case Operator::Div:
    if (right != 0)
      return {EvaluationKind::Value, ComptimeValue(left / right),
              isComptimeOnly};
    break;
  case Operator::Rem:
    if (right != 0)
      return {EvaluationKind::Value, ComptimeValue(left % right),
              isComptimeOnly};
    break;
  case Operator::ShiftLeft:
  case Operator::ShiftRight:
  case Operator::BitAnd:
  case Operator::BitOr:
  case Operator::BitXor:
    return cannotEvaluate();
  case Operator::LT:
    return {EvaluationKind::Value, ComptimeValue(left < right),
            isComptimeOnly};
  case Operator::LE:
    return {EvaluationKind::Value, ComptimeValue(left <= right),
            isComptimeOnly};
  case Operator::GT:
    return {EvaluationKind::Value, ComptimeValue(left > right),
            isComptimeOnly};
  case Operator::GE:
    return {EvaluationKind::Value, ComptimeValue(left >= right),
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
