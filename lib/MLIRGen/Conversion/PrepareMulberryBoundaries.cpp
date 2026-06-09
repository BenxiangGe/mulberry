//===--- PrepareMulberryBoundaries.cpp ------------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/Conversion/CherryPasses.h"
#include "cherry/MLIRGen/IR/MulberryOps.h"
#include "cherry/MLIRGen/IR/MulberryTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <optional>
#include <vector>

namespace mlir::cherry {

#define GEN_PASS_DEF_PREPAREMULBERRYBOUNDARIES
#include "cherry/MLIRGen/Conversion/CherryPasses.h.inc"

namespace {

static auto isTensorList(Type type) -> bool {
  auto listType = llvm::dyn_cast<mulberry::ListType>(type);
  return listType &&
         llvm::isa<mulberry::TensorType>(listType.getElementType());
}

static auto getTensorDescType(mulberry::TensorType tensorType)
    -> mulberry::TensorDescType {
  return mulberry::TensorDescType::get(
      tensorType.getContext(), tensorType.getShape(),
      tensorType.getElementType());
}

static auto getTensorHandleType(mulberry::TensorDescType descType)
    -> mulberry::TensorHandleType {
  return mulberry::TensorHandleType::get(
      descType.getContext(), descType.getShape(), descType.getElementType());
}

static auto getTensorListDescType(mulberry::ListType listType)
    -> mulberry::ListDescType {
  auto tensorType =
      llvm::cast<mulberry::TensorType>(listType.getElementType());
  return mulberry::ListDescType::get(listType.getContext(),
                                     getTensorDescType(tensorType));
}

static auto getBoundaryPreparedType(Type type) -> std::optional<Type> {
  auto listType = llvm::dyn_cast<mulberry::ListType>(type);
  if (listType && isTensorList(listType))
    return getTensorListDescType(listType);

  return std::nullopt;
}

enum class BoundaryKind {
  FunctionParameter,
  FunctionReturn,
  CallArgument,
  CallResult,
  ReturnValue,
};

static auto describeBoundaryKind(BoundaryKind kind) -> const char* {
  switch (kind) {
  case BoundaryKind::FunctionParameter:
    return "function parameter";
  case BoundaryKind::FunctionReturn:
    return "function return";
  case BoundaryKind::CallArgument:
    return "call argument";
  case BoundaryKind::CallResult:
    return "call result";
  case BoundaryKind::ReturnValue:
    return "return value";
  }

  llvm_unreachable("unknown boundary kind");
}

static auto checkNoSourceTensorList(Operation* op, TypeRange types,
                                    BoundaryKind kind)
    -> LogicalResult {
  for (auto type : types) {
    if (!isTensorList(type))
      continue;

    op->emitError()
        << "source-level List<Tensor> " << describeBoundaryKind(kind)
        << " boundary preparation is not implemented yet";
    return failure();
  }

  return success();
}

static auto hasTensorListType(TypeRange types) -> bool {
  for (auto type : types)
    if (isTensorList(type))
      return true;

  return false;
}

static auto hasSupportedBoundaryResults(func::FuncOp funcOp) -> bool {
  return !hasTensorListType(funcOp.getFunctionType().getResults());
}

static auto rewriteListSizeOp(mulberry::ListSizeOp op, Value desc)
    -> LogicalResult {
  OpBuilder builder(op);
  auto length = mulberry::ListDescLengthOp::create(
      builder, op.getLoc(), builder.getIndexType(), desc);
  auto cast = arith::IndexCastOp::create(builder, op.getLoc(),
                                         builder.getI64Type(), length);
  op.replaceAllUsesWith(cast.getResult());
  op.erase();
  return success();
}

static auto emitUnsupportedParameterUse(Operation* op) -> InFlightDiagnostic {
  return op->emitError()
         << "List<Tensor> function parameter boundary preparation only "
            "supports direct list.size/list.get uses for now";
}

static auto emitTensorHandleExposureError(Operation* op) -> InFlightDiagnostic {
  return op->emitError()
         << "List<Tensor> boundary rewrite cannot expose tensor_handle to "
            "downstream users yet";
}

static auto emitUnsupportedCallArgRewrite(Operation* op) -> InFlightDiagnostic {
  return op->emitError() << "unsupported function boundary argument rewrite";
}

static auto rewriteListGetOp(mulberry::ListGetOp op, Value desc)
    -> LogicalResult {
  if (!op.getResult().use_empty()) {
    emitTensorHandleExposureError(op);
    return failure();
  }

  auto tensorType = llvm::cast<mulberry::TensorType>(op.getResult().getType());
  auto descType = getTensorDescType(tensorType);
  auto handleType = getTensorHandleType(descType);
  OpBuilder builder(op);
  auto tensorDesc = mulberry::ListDescGetOp::create(
      builder, op.getLoc(), descType, desc, op.getIndex());
  mulberry::TensorHandleFromDescOp::create(builder, op.getLoc(), handleType,
                                           tensorDesc);
  op.erase();
  return success();
}

static auto checkParameterUses(BlockArgument argument) -> LogicalResult {
  for (auto& use : argument.getUses()) {
    auto user = use.getOwner();
    if (use.getOperandNumber() != 0) {
      return emitUnsupportedParameterUse(user);
    }

    if (llvm::isa<mulberry::ListSizeOp>(user))
      continue;

    if (auto getOp = llvm::dyn_cast<mulberry::ListGetOp>(user)) {
      if (!getOp.getResult().use_empty())
        return emitTensorHandleExposureError(getOp);
      continue;
    }

    return emitUnsupportedParameterUse(user);
  }

  return success();
}

static auto rewriteParameterUses(BlockArgument argument) -> LogicalResult {
  struct OriginalUse {
    Operation* user;
    unsigned operandNumber;
  };

  std::vector<OriginalUse> originalUses;
  for (auto& use : argument.getUses())
    originalUses.push_back({use.getOwner(), use.getOperandNumber()});

  for (auto use : originalUses) {
    auto user = use.user;
    if (use.operandNumber != 0) {
      emitUnsupportedParameterUse(user);
      return failure();
    }

    if (auto sizeOp = llvm::dyn_cast<mulberry::ListSizeOp>(user)) {
      if (failed(rewriteListSizeOp(sizeOp, argument)))
        return failure();
      continue;
    }

    if (auto getOp = llvm::dyn_cast<mulberry::ListGetOp>(user)) {
      if (failed(rewriteListGetOp(getOp, argument)))
        return failure();
      continue;
    }

    // Keep this skeleton honest: only direct list operations are rewritten.
    // Other uses would require a broader source-list-to-descriptor model.
    emitUnsupportedParameterUse(user);
    return failure();
  }

  return success();
}

static auto getPreparedInputTypes(func::FuncOp funcOp)
    -> std::optional<std::vector<Type>> {
  auto funcType = funcOp.getFunctionType();
  std::vector<Type> inputTypes;
  auto changed = false;

  for (auto type : funcType.getInputs()) {
    if (auto preparedType = getBoundaryPreparedType(type)) {
      inputTypes.push_back(*preparedType);
      changed = true;
      continue;
    }

    inputTypes.push_back(type);
  }

  if (!changed)
    return std::nullopt;

  return inputTypes;
}

static auto rewritePrivateFunctionParameters(func::FuncOp funcOp,
                                             ArrayRef<Type> inputTypes)
    -> LogicalResult {
  if (!funcOp.isPrivate() || funcOp.isExternal())
    return success();

  auto funcType = funcOp.getFunctionType();

  // Changing a function type is only safe when the entry block arguments and
  // all body uses are rewritten in the same step.
  auto& entryBlock = funcOp.front();
  for (auto [argument, inputType] :
       llvm::zip(entryBlock.getArguments(), inputTypes)) {
    if (argument.getType() == inputType)
      continue;

    if (failed(rewriteParameterUses(argument)))
      return failure();
    argument.setType(inputType);
  }

  auto newFuncType = FunctionType::get(funcOp.getContext(), inputTypes,
                                       funcType.getResults());
  funcOp.setType(newFuncType);
  return success();
}

static auto checkFunctionArgUses(func::FuncOp funcOp, ArrayRef<Type> inputTypes)
    -> LogicalResult {
  if (funcOp.isExternal())
    return success();

  auto& entryBlock = funcOp.front();
  for (auto [argument, inputType] :
       llvm::zip(entryBlock.getArguments(), inputTypes)) {
    if (argument.getType() == inputType)
      continue;

    if (failed(checkParameterUses(argument)))
      return failure();
  }

  return success();
}

static auto checkCallRewrite(func::CallOp callOp, ArrayRef<Type> inputTypes)
    -> LogicalResult {
  if (hasTensorListType(callOp.getResultTypes()))
    return callOp.emitError()
           << "List<Tensor> call-site boundary preparation with list results "
              "is not implemented yet";

  if (callOp.getArgAttrsAttr() || callOp.getResAttrsAttr())
    return callOp.emitError()
           << "List<Tensor> call-site boundary preparation with call "
              "argument/result attributes is not implemented yet";

  for (auto [operand, inputType] :
       llvm::zip(callOp.getArgOperands(), inputTypes)) {
    if (operand.getType() == inputType)
      continue;

    if (!llvm::isa<mulberry::ListDescType>(inputType) ||
        !isTensorList(operand.getType()))
      return emitUnsupportedCallArgRewrite(callOp);
  }

  return success();
}

static auto rewriteCallArguments(func::CallOp callOp, ArrayRef<Type> inputTypes)
    -> LogicalResult {
  std::vector<Value> operands;
  auto changed = false;

  for (auto [operand, inputType] :
       llvm::zip(callOp.getArgOperands(), inputTypes)) {
    if (operand.getType() == inputType) {
      operands.push_back(operand);
      continue;
    }

    auto descType = llvm::dyn_cast<mulberry::ListDescType>(inputType);
    if (!descType)
      return emitUnsupportedCallArgRewrite(callOp);

    OpBuilder builder(callOp);
    auto desc = mulberry::ListToDescOp::create(builder, callOp.getLoc(),
                                               descType, operand);
    operands.push_back(desc);
    changed = true;
  }

  if (!changed)
    return success();

  OpBuilder builder(callOp);
  auto newCallOp = func::CallOp::create(builder, callOp.getLoc(),
                                        callOp.getCallee(),
                                        callOp.getResultTypes(), operands);
  newCallOp.setNoInlineAttr(callOp.getNoInlineAttr());
  for (auto [oldResult, newResult] :
       llvm::zip(callOp.getResults(), newCallOp.getResults()))
    oldResult.replaceAllUsesWith(newResult);
  callOp.erase();
  return success();
}

static auto rewritePrivateFunctionBoundary(func::FuncOp funcOp)
    -> LogicalResult {
  if (!funcOp.isPrivate())
    return success();

  if (!hasSupportedBoundaryResults(funcOp))
    return success();

  auto inputTypes = getPreparedInputTypes(funcOp);
  if (!inputTypes)
    return success();

  auto* moduleOp = funcOp->getParentOp();
  auto uses = SymbolTable::getSymbolUses(funcOp.getOperation(), moduleOp);
  if (!uses)
    return funcOp.emitError()
           << "cannot prove List<Tensor> function boundary symbol uses";

  std::vector<func::CallOp> callOps;
  for (auto use : *uses) {
    auto callOp = llvm::dyn_cast<func::CallOp>(use.getUser());
    if (!callOp)
      return use.getUser()->emitError()
             << "List<Tensor> function boundary preparation only supports "
                "direct func.call users";
    callOps.push_back(callOp);
  }

  // Rewrite all direct calls before changing the callee signature. Rebuilding
  // call ops while iterating symbol uses would invalidate the use list.
  for (auto callOp : callOps) {
    if (failed(checkCallRewrite(callOp, *inputTypes)))
      return failure();
  }

  if (failed(checkFunctionArgUses(funcOp, *inputTypes)))
    return failure();

  for (auto callOp : callOps) {
    if (failed(rewriteCallArguments(callOp, *inputTypes)))
      return failure();
  }

  if (funcOp.isExternal()) {
    auto newFuncType = FunctionType::get(
        funcOp.getContext(), *inputTypes, funcOp.getFunctionType().getResults());
    funcOp.setType(newFuncType);
    return success();
  }

  return rewritePrivateFunctionParameters(funcOp, *inputTypes);
}

struct PrepareMulberryBoundaries
    : public impl::PrepareMulberryBoundariesBase<PrepareMulberryBoundaries> {
  using impl::PrepareMulberryBoundariesBase<
      PrepareMulberryBoundaries>::PrepareMulberryBoundariesBase;

  auto runOnOperation() -> void final {
    auto hasError = false;

    getOperation()->walk([&](Operation* op) {
      if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {
        if (failed(rewritePrivateFunctionBoundary(funcOp))) {
          hasError = true;
          return WalkResult::interrupt();
        }
      }

      return WalkResult::advance();
    });

    if (hasError) {
      signalPassFailure();
      return;
    }

    getOperation()->walk([&](Operation* op) {
      if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {

        auto funcType = funcOp.getFunctionType();
        // TODO: Rewrite List<Tensor> boundaries only when the pass can update
        // the callee signature, entry block arguments, call sites, and body
        // uses together. Rewriting just one side would create half-converted IR.
        if (failed(checkNoSourceTensorList(
                op, funcType.getInputs(), BoundaryKind::FunctionParameter)) ||
            failed(checkNoSourceTensorList(
                op, funcType.getResults(), BoundaryKind::FunctionReturn))) {
          hasError = true;
          return WalkResult::interrupt();
        }
      }

      if (llvm::isa<func::CallOp>(op)) {
        if (failed(checkNoSourceTensorList(
                op, op->getOperandTypes(), BoundaryKind::CallArgument)) ||
            failed(checkNoSourceTensorList(
                op, op->getResultTypes(), BoundaryKind::CallResult))) {
          hasError = true;
          return WalkResult::interrupt();
        }
      }

      if (llvm::isa<func::ReturnOp>(op)) {
        if (failed(checkNoSourceTensorList(
                op, op->getOperandTypes(), BoundaryKind::ReturnValue))) {
          hasError = true;
          return WalkResult::interrupt();
        }
      }

      return WalkResult::advance();
    });

    if (hasError)
      signalPassFailure();
  }
};

} // end namespace
} // namespace mlir::cherry
