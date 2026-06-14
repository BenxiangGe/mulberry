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

static auto isScalarListElement(Type type) -> bool {
  return type.isIndex() || llvm::isa<IntegerType, FloatType>(type);
}

static auto getTensorDescType(mulberry::TensorType tensorType)
    -> mulberry::TensorDescType {
  return mulberry::TensorDescType::get(tensorType.getContext(),
                                       tensorType.getShape(),
                                       tensorType.getElementType());
}

static auto getListDescElementType(mulberry::ListType listType) -> Type {
  auto elementType = listType.getElementType();
  if (auto tensorType = llvm::dyn_cast<mulberry::TensorType>(elementType))
    return getTensorDescType(tensorType);

  if (isScalarListElement(elementType))
    return elementType;

  return {};
}

static auto isBoundaryList(Type type) -> bool {
  auto listType = llvm::dyn_cast<mulberry::ListType>(type);
  return listType && getListDescElementType(listType);
}

static auto getBoundaryPreparedType(Type type) -> std::optional<Type> {
  auto listType = llvm::dyn_cast<mulberry::ListType>(type);
  if (!listType)
    return std::nullopt;

  auto elementType = getListDescElementType(listType);
  if (elementType)
    return mulberry::ListDescType::get(listType.getContext(), elementType);

  return std::nullopt;
}

enum class BoundaryKind {
  FunctionParameter,
  FunctionReturn,
  CallArgument,
  CallResult,
  ReturnValue,
};

static auto describeBoundaryKind(BoundaryKind kind) -> const char * {
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

static auto checkNoSourceBoundaryList(Operation *op, TypeRange types,
                                      BoundaryKind kind) -> LogicalResult {
  for (auto type : types) {
    if (!isBoundaryList(type))
      continue;

    op->emitError() << "source-level List<T> "
                    << describeBoundaryKind(kind)
                    << " boundary preparation is not implemented yet";
    return failure();
  }

  return success();
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

static auto emitUnsupportedListUse(Operation *op) -> InFlightDiagnostic {
  return op->emitError()
         << "List<T> boundary preparation only supports direct "
            "list.size/list.get uses for now";
}

static auto rewriteListGetOp(mulberry::ListGetOp op, Value desc)
    -> LogicalResult {
  auto descType = llvm::cast<mulberry::ListDescType>(desc.getType());
  OpBuilder builder(op);
  auto element = mulberry::ListDescGetOp::create(
      builder, op.getLoc(), descType.getElementType(), desc, op.getIndex());
  auto tensorType = llvm::dyn_cast<mulberry::TensorType>(
      op.getResult().getType());
  if (!tensorType) {
    op.replaceAllUsesWith(element.getResult());
    op.erase();
    return success();
  }

  auto tensor = mulberry::TensorDescUnpackOp::create(
      builder, op.getLoc(), tensorType, element);
  op.replaceAllUsesWith(tensor.getResult());
  op.erase();
  return success();
}

static auto checkListUses(Value list) -> LogicalResult {
  for (auto &use : list.getUses()) {
    auto user = use.getOwner();
    if (use.getOperandNumber() != 0) {
      return emitUnsupportedListUse(user);
    }

    if (llvm::isa<mulberry::ListSizeOp>(user))
      continue;

    if (llvm::isa<mulberry::ListGetOp>(user))
      continue;

    return emitUnsupportedListUse(user);
  }

  return success();
}

static auto rewriteListUses(Value list, Value desc) -> LogicalResult {
  struct OriginalUse {
    Operation *user;
    unsigned operandNumber;
  };

  std::vector<OriginalUse> originalUses;
  for (auto &use : list.getUses())
    originalUses.push_back({use.getOwner(), use.getOperandNumber()});

  for (auto use : originalUses) {
    auto user = use.user;
    if (use.operandNumber != 0) {
      emitUnsupportedListUse(user);
      return failure();
    }

    if (auto sizeOp = llvm::dyn_cast<mulberry::ListSizeOp>(user)) {
      if (failed(rewriteListSizeOp(sizeOp, desc)))
        return failure();
      continue;
    }

    if (auto getOp = llvm::dyn_cast<mulberry::ListGetOp>(user)) {
      if (failed(rewriteListGetOp(getOp, desc)))
        return failure();
      continue;
    }

    // Keep this skeleton honest: only direct list operations are rewritten.
    // Other uses would require a broader source-list-to-descriptor model.
    emitUnsupportedListUse(user);
    return failure();
  }

  return success();
}

static auto getPreparedTypes(TypeRange types)
    -> std::optional<std::vector<Type>> {
  std::vector<Type> preparedTypes;
  auto changed = false;

  for (auto type : types) {
    if (auto preparedType = getBoundaryPreparedType(type)) {
      preparedTypes.push_back(*preparedType);
      changed = true;
      continue;
    }

    preparedTypes.push_back(type);
  }

  if (!changed)
    return std::nullopt;

  return preparedTypes;
}

static auto keepTypes(TypeRange types) -> std::vector<Type> {
  std::vector<Type> result;
  for (auto type : types)
    result.push_back(type);
  return result;
}

static auto rewriteFunctionParameters(func::FuncOp funcOp,
                                      ArrayRef<Type> inputTypes,
                                      ArrayRef<Type> resultTypes)
    -> LogicalResult {
  // Changing a function type is only safe when the entry block arguments and
  // all body uses are rewritten in the same step.
  auto &entryBlock = funcOp.front();
  for (auto [argument, inputType] :
       llvm::zip(entryBlock.getArguments(), inputTypes)) {
    if (argument.getType() == inputType)
      continue;

    argument.setType(inputType);
    if (failed(rewriteListUses(argument, argument)))
      return failure();
  }

  auto newFuncType =
      FunctionType::get(funcOp.getContext(), inputTypes, resultTypes);
  funcOp.setType(newFuncType);
  return success();
}

static auto checkFunctionArgUses(func::FuncOp funcOp, ArrayRef<Type> inputTypes)
    -> LogicalResult {
  if (funcOp.isExternal())
    return success();

  auto &entryBlock = funcOp.front();
  for (auto [argument, inputType] :
       llvm::zip(entryBlock.getArguments(), inputTypes)) {
    if (argument.getType() == inputType)
      continue;

    if (failed(checkListUses(argument)))
      return failure();
  }

  return success();
}

static auto emitUnsupportedCallArgRewrite(Operation *op) -> InFlightDiagnostic {
  return op->emitError() << "unsupported function boundary argument rewrite";
}

static auto emitUnsupportedCallResultRewrite(Operation *op)
    -> InFlightDiagnostic {
  return op->emitError() << "unsupported function boundary result rewrite";
}

static auto checkCallRewrite(func::CallOp callOp, ArrayRef<Type> inputTypes,
                             ArrayRef<Type> resultTypes) -> LogicalResult {
  if (callOp.getArgAttrsAttr() || callOp.getResAttrsAttr())
    return callOp.emitError()
           << "List<T> call-site boundary preparation with call "
              "argument/result attributes is not implemented yet";

  for (auto [operand, inputType] :
       llvm::zip(callOp.getArgOperands(), inputTypes)) {
    if (operand.getType() == inputType)
      continue;

    if (!llvm::isa<mulberry::ListDescType>(inputType) ||
        !isBoundaryList(operand.getType()))
      return emitUnsupportedCallArgRewrite(callOp);
  }

  for (auto [result, resultType] :
       llvm::zip(callOp.getResults(), resultTypes)) {
    if (result.getType() == resultType)
      continue;

    if (!llvm::isa<mulberry::ListDescType>(resultType) ||
        !isBoundaryList(result.getType()))
      return emitUnsupportedCallResultRewrite(callOp);

    if (failed(checkListUses(result)))
      return failure();
  }

  return success();
}

static auto rewriteCall(func::CallOp callOp, ArrayRef<Type> inputTypes,
                        ArrayRef<Type> resultTypes) -> LogicalResult {
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

  for (auto [result, resultType] : llvm::zip(callOp.getResults(), resultTypes))
    if (result.getType() != resultType)
      changed = true;

  if (!changed)
    return success();

  OpBuilder builder(callOp);
  auto newCallOp = func::CallOp::create(
      builder, callOp.getLoc(), callOp.getCallee(), resultTypes, operands);
  newCallOp.setNoInlineAttr(callOp.getNoInlineAttr());

  for (auto [oldResult, newResult] :
       llvm::zip(callOp.getResults(), newCallOp.getResults())) {
    if (oldResult.getType() == newResult.getType()) {
      oldResult.replaceAllUsesWith(newResult);
      continue;
    }

    if (failed(rewriteListUses(oldResult, newResult)))
      return failure();
  }

  callOp.erase();
  return success();
}

static auto buildEscapedDesc(mulberry::ListCreateOp createOp,
                             mulberry::ListDescType descType,
                             OpBuilder &builder) -> Value {
  auto descElementType = descType.getElementType();
  auto storageType =
      mulberry::ListStorageType::get(descType.getContext(), descElementType);
  auto length = arith::ConstantIndexOp::create(builder, createOp.getLoc(),
                                               createOp.getElements().size());
  auto storage = mulberry::ListAllocOp::create(builder, createOp.getLoc(),
                                               storageType, length.getResult());

  for (auto element : llvm::enumerate(createOp.getElements())) {
    auto index = arith::ConstantIndexOp::create(builder, createOp.getLoc(),
                                                element.index());
    Value value = element.value();
    if (auto tensorDescType =
            llvm::dyn_cast<mulberry::TensorDescType>(descElementType)) {
      auto tensorDesc = mulberry::TensorDescPackOp::create(
          builder, createOp.getLoc(), tensorDescType, element.value());
      value = tensorDesc.getResult();
    }
    mulberry::ListStoreOp::create(builder, createOp.getLoc(), value,
                                  storage.getResult(), index.getResult());
  }

  // Returning a list copies local storage to Boehm-managed ABI storage. There is
  // no explicit descriptor dealloc op in the Boehm-only ownership model.
  auto escaped = mulberry::ListEscapeStorageOp::create(
      builder, createOp.getLoc(), storageType, storage.getResult(),
      length.getResult());
  Value data = escaped.getResult();

  auto desc =
      mulberry::ListDescPackOp::create(builder, createOp.getLoc(), descType,
                                       length.getResult(), data);
  return desc.getResult();
}

static auto emitUnsupportedReturnRewrite(Operation *op) -> InFlightDiagnostic {
  return op->emitError()
         << "List<T> return rewrite only supports returning a local "
            "list.create for now";
}

static auto checkReturnOp(func::ReturnOp returnOp, ArrayRef<Type> resultTypes)
    -> LogicalResult {
  for (auto [operand, resultType] :
       llvm::zip(returnOp.getOperands(), resultTypes)) {
    if (operand.getType() == resultType)
      continue;

    if (!llvm::isa<mulberry::ListDescType>(resultType) ||
        !isBoundaryList(operand.getType()))
      return emitUnsupportedReturnRewrite(returnOp);

    if (!operand.getDefiningOp<mulberry::ListCreateOp>())
      return emitUnsupportedReturnRewrite(returnOp);
  }

  return success();
}

static auto checkReturnOps(func::FuncOp funcOp, ArrayRef<Type> resultTypes)
    -> LogicalResult {
  auto failedRewrite = false;
  funcOp.walk([&](func::ReturnOp returnOp) {
    if (failed(checkReturnOp(returnOp, resultTypes))) {
      failedRewrite = true;
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  return failure(failedRewrite);
}

static auto rewriteReturnOp(func::ReturnOp returnOp, ArrayRef<Type> resultTypes)
    -> LogicalResult {
  std::vector<Value> operands;
  auto changed = false;
  std::vector<mulberry::ListCreateOp> oldCreateOps;

  OpBuilder builder(returnOp);
  for (auto [operand, resultType] :
       llvm::zip(returnOp.getOperands(), resultTypes)) {
    if (operand.getType() == resultType) {
      operands.push_back(operand);
      continue;
    }

    auto descType = llvm::dyn_cast<mulberry::ListDescType>(resultType);
    auto createOp = operand.getDefiningOp<mulberry::ListCreateOp>();
    if (!descType || !createOp)
      return emitUnsupportedReturnRewrite(returnOp);

    operands.push_back(buildEscapedDesc(createOp, descType, builder));
    oldCreateOps.push_back(createOp);
    changed = true;
  }

  if (!changed)
    return success();

  returnOp->setOperands(operands);
  for (auto createOp : oldCreateOps)
    if (createOp->use_empty())
      createOp.erase();

  return success();
}

static auto rewriteReturnOps(func::FuncOp funcOp, ArrayRef<Type> resultTypes)
    -> LogicalResult {
  auto failedRewrite = false;
  funcOp.walk([&](func::ReturnOp returnOp) {
    if (failed(rewriteReturnOp(returnOp, resultTypes))) {
      failedRewrite = true;
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  return failure(failedRewrite);
}

static auto rewriteFunctionBoundary(func::FuncOp funcOp) -> LogicalResult {
  if (funcOp.isExternal())
    return success();

  auto funcType = funcOp.getFunctionType();
  auto inputTypes = getPreparedTypes(funcType.getInputs());
  auto resultTypes = getPreparedTypes(funcType.getResults());
  if (!inputTypes && !resultTypes)
    return success();

  auto preparedInputs =
      inputTypes ? *inputTypes : keepTypes(funcType.getInputs());
  auto preparedResults =
      resultTypes ? *resultTypes : keepTypes(funcType.getResults());

  auto *moduleOp = funcOp->getParentOp();
  auto uses = SymbolTable::getSymbolUses(funcOp.getOperation(), moduleOp);
  if (!uses)
    return funcOp.emitError()
           << "cannot prove List<T> function boundary symbol uses";

  // Source functions currently have no export syntax. A non-external function
  // can be boundary-rewritten when every known symbol use is a direct
  // func.call, because the callee signature and all callers are rewritten
  // together in this module.
  std::vector<func::CallOp> callOps;
  for (auto use : *uses) {
    auto callOp = llvm::dyn_cast<func::CallOp>(use.getUser());
    if (!callOp)
      return use.getUser()->emitError()
             << "List<T> function boundary preparation only supports "
                "direct func.call users";
    callOps.push_back(callOp);
  }

  // Rewrite all direct calls before changing the callee signature. Rebuilding
  // call ops while iterating symbol uses would invalidate the use list.
  for (auto callOp : callOps) {
    if (failed(checkCallRewrite(callOp, preparedInputs, preparedResults)))
      return failure();
  }

  if (failed(checkFunctionArgUses(funcOp, preparedInputs)))
    return failure();

  if (failed(checkReturnOps(funcOp, preparedResults)))
    return failure();

  for (auto callOp : callOps) {
    if (failed(rewriteCall(callOp, preparedInputs, preparedResults)))
      return failure();
  }

  if (failed(rewriteReturnOps(funcOp, preparedResults)))
    return failure();

  return rewriteFunctionParameters(funcOp, preparedInputs, preparedResults);
}

struct PrepareMulberryBoundaries
    : public impl::PrepareMulberryBoundariesBase<PrepareMulberryBoundaries> {
  using impl::PrepareMulberryBoundariesBase<
      PrepareMulberryBoundaries>::PrepareMulberryBoundariesBase;

  auto runOnOperation() -> void final {
    auto hasError = false;

    getOperation()->walk([&](Operation *op) {
      if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {
        if (failed(rewriteFunctionBoundary(funcOp))) {
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

    getOperation()->walk([&](Operation *op) {
      if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {

        auto funcType = funcOp.getFunctionType();
        if (failed(checkNoSourceBoundaryList(
                op, funcType.getInputs(), BoundaryKind::FunctionParameter)) ||
            failed(checkNoSourceBoundaryList(
                op, funcType.getResults(), BoundaryKind::FunctionReturn))) {
          hasError = true;
          return WalkResult::interrupt();
        }
      }

      if (llvm::isa<func::CallOp>(op)) {
        if (failed(checkNoSourceBoundaryList(
                op, op->getOperandTypes(), BoundaryKind::CallArgument)) ||
            failed(checkNoSourceBoundaryList(
                op, op->getResultTypes(), BoundaryKind::CallResult))) {
          hasError = true;
          return WalkResult::interrupt();
        }
      }

      if (llvm::isa<func::ReturnOp>(op)) {
        if (failed(checkNoSourceBoundaryList(
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
