//===--- SemaBuiltins.cpp - Builtin semantic analysis -------------------===//

// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information

//===----------------------------------------------------------------------===//

#include "SemaImpl.h"
#include "SemaExpressions.h"
#include "SemaSupport.h"
#include "mulberry/Basic/Builtins.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "Sema"

namespace mulberry {
using llvm::cast;

auto SemaImpl::registerBuiltinHandlers() -> void {
  registerBuiltinHandler(
      "std.tensor.__allocate",
      [this](Expr *node, const Type *expectedType) {
        return ExpressionSema(*this).semaTensorStorageAllocCall(
            cast<CallExpr>(node), expectedType);
      });
  registerBuiltinHandler(
      "std.tensor.__dispose",
      [this](Expr *node, const Type *) {
        return ExpressionSema(*this).semaTensorDisposeCall(
            cast<CallExpr>(node));
      });
  registerBuiltinHandler(
      "std.core.toUInt8",
      [this](Expr *node, const Type *expectedType) {
        return semaToUInt8(cast<CallExpr>(node), expectedType);
      });
  registerBuiltinHandler(
      "std.core.toUInt64",
      [this](Expr *node, const Type *expectedType) {
        return semaToUInt64(cast<CallExpr>(node), expectedType);
      });
  registerBuiltinHandler(
      "std.core.toFloat32",
      [this](Expr *node, const Type *expectedType) {
        return semaToFloat32(cast<CallExpr>(node), expectedType);
      });
}

auto SemaImpl::registerBuiltinHandler(std::string_view name,
                                      BuiltinHandler handler) -> void {
  auto [iter, inserted] =
      _builtinHandlers.try_emplace(std::string(name), std::move(handler));
  if (!inserted)
    llvm_unreachable("duplicate builtin Sema handler");
  LLVM_DEBUG(llvm::dbgs() << "register builtin Sema handler `" << iter->first
                          << "`\n");
}

auto SemaImpl::lookupBuiltinHandler(std::string_view name) const
    -> const BuiltinHandler * {
  auto iter = _builtinHandlers.find(name);
  if (iter == _builtinHandlers.end())
    return nullptr;
  return &iter->second;
}

auto SemaImpl::semaToUInt8(CallExpr *node, const Type *expectedType)
    -> llvm::LogicalResult {
  auto &arguments = node->expressions();
  if (arguments.size() != 1) {
    auto diagnostic =
        formatNameSizeDiagnostic(diag::func_param, node->name(), 1);
    return emitError(node, diagnostic);
  }

  auto *parameterType =
      _typeContext.getBuiltinType(BuiltinTypeKind::UInt64);
  if (llvm::failed(sema(arguments.front().get(), parameterType)))
    return failure();
  if (!sameType(arguments.front()->type(), parameterType))
    return emitError(arguments.front().get(), diag::mismatch_type);

  auto *resultType = _typeContext.getBuiltinType(BuiltinTypeKind::UInt8);
  if (expectedType && !sameType(expectedType, resultType))
    return emitError(node, diag::mismatch_type);
  node->setType(resultType);
  return success();
}

auto SemaImpl::semaToUInt64(CallExpr *node, const Type *expectedType)
    -> llvm::LogicalResult {
  auto &arguments = node->expressions();
  if (arguments.size() != 1) {
    auto diagnostic =
        formatNameSizeDiagnostic(diag::func_param, node->name(), 1);
    return emitError(node, diagnostic);
  }

  auto *parameterType =
      _typeContext.getBuiltinType(BuiltinTypeKind::UInt8);
  if (llvm::failed(sema(arguments.front().get(), parameterType)))
    return failure();
  if (!sameType(arguments.front()->type(), parameterType))
    return emitError(arguments.front().get(), diag::mismatch_type);

  auto *resultType = _typeContext.getBuiltinType(BuiltinTypeKind::UInt64);
  if (expectedType && !sameType(expectedType, resultType))
    return emitError(node, diag::mismatch_type);
  node->setType(resultType);
  return success();
}

auto SemaImpl::semaToFloat32(CallExpr *node, const Type *expectedType)
    -> llvm::LogicalResult {
  auto &arguments = node->expressions();
  if (arguments.size() != 1) {
    auto diagnostic =
        formatNameSizeDiagnostic(diag::func_param, node->name(), 1);
    return emitError(node, diagnostic);
  }

  auto *parameterType =
      _typeContext.getBuiltinType(BuiltinTypeKind::UInt64);
  if (llvm::failed(sema(arguments.front().get(), parameterType)))
    return failure();
  if (!sameType(arguments.front()->type(), parameterType))
    return emitError(arguments.front().get(), diag::mismatch_type);

  auto *resultType = _typeContext.getBuiltinType(BuiltinTypeKind::Float32);
  if (expectedType && !sameType(expectedType, resultType))
    return emitError(node, diag::mismatch_type);
  node->setType(resultType);
  return success();
}

} // namespace mulberry
