//===--- LowerExternABI.cpp ----------------------------------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "mulberry/MLIRGen/Conversion/MulberryPasses.h"
#include "mulberry/MLIRGen/IR/MulberryOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/Debug.h"

#include <vector>

namespace mlir::mulberry_core {

#define GEN_PASS_DEF_LOWEREXTERNABI
#include "mulberry/MLIRGen/Conversion/MulberryPasses.h.inc"

#undef DEBUG_TYPE
#define DEBUG_TYPE "lower-extern-abi"

namespace {

struct ExternFunctionABI {
  func::FuncOp function;
  FunctionType sourceType;
  FunctionType layoutType;
};

auto isSourceObjectReference(Type type) -> bool {
  auto pointerType = llvm::dyn_cast<PtrType>(type);
  if (!pointerType)
    return false;
  auto pointeeType = pointerType.getPointeeType();
  return llvm::isa<RecordType, DataType>(pointeeType);
}

auto isLayoutObjectValue(Type type) -> bool {
  return llvm::isa<RecordType, DataType>(type);
}

auto lowerExternCall(func::CallOp call, const ExternFunctionABI &function,
                     IRRewriter &rewriter) -> LogicalResult {
  auto sourceType = call.getCalleeType();
  auto layoutType = function.layoutType;
  if (sourceType.getNumInputs() != layoutType.getNumInputs() ||
      sourceType.getNumResults() != layoutType.getNumResults())
    return call.emitOpError("extern source and layout signatures disagree");

  rewriter.setInsertionPoint(call);
  std::vector<Value> operands;
  for (unsigned index = 0; index < sourceType.getNumInputs(); ++index) {
    auto sourceInput = sourceType.getInput(index);
    auto layoutInput = layoutType.getInput(index);
    auto operand = call.getOperand(index);
    if (isSourceObjectReference(sourceInput)) {
      if (!isLayoutObjectValue(layoutInput))
        return call.emitOpError()
               << "extern object argument has non-object layout type at "
               << index;
      operands.push_back(
          LoadOp::create(rewriter, call.getLoc(), layoutInput, operand));
      continue;
    }

    if (sourceInput != layoutInput)
      return call.emitOpError()
             << "extern non-object argument changes type at " << index;
    operands.push_back(operand);
  }

  auto callee = call->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!callee)
    return call.emitOpError("requires a direct callee symbol");

  auto loweredCall = func::CallOp::create(
      rewriter, call.getLoc(), callee, layoutType.getResults(), operands);

  if (sourceType.getNumResults() == 0) {
    rewriter.eraseOp(call);
    return success();
  }

  auto sourceResult = sourceType.getResult(0);
  auto layoutResult = layoutType.getResult(0);
  if (isSourceObjectReference(sourceResult)) {
    if (!isLayoutObjectValue(layoutResult))
      return call.emitOpError(
          "extern object result has non-object layout type");

    // Runtime declarations return the record value directly. Recreate the
    // source object reference only after the ABI call, so the high-level
    // object model never has to expose the runtime record representation.
    rewriter.setInsertionPointAfter(loweredCall);
    auto count = arith::ConstantIntOp::create(rewriter, call.getLoc(), 1, 64);
    auto objectType = PtrType::get(call.getContext(), layoutResult);
    auto object = HeapAllocOp::create(rewriter, call.getLoc(), objectType,
                                      layoutResult, count);
    StoreOp::create(rewriter, call.getLoc(), loweredCall.getResult(0),
                    object.getResult());
    call.getResult(0).replaceAllUsesWith(object.getResult());
  } else {
    if (sourceResult != layoutResult)
      return call.emitOpError("extern non-object result changes type");
    call.getResult(0).replaceAllUsesWith(loweredCall.getResult(0));
  }

  rewriter.eraseOp(call);
  LLVM_DEBUG(llvm::dbgs() << "lower extern call `" << callee.getValue()
                          << "` to layout ABI\n");
  return success();
}

struct LowerExternABI : public impl::LowerExternABIBase<LowerExternABI> {
  using impl::LowerExternABIBase<LowerExternABI>::LowerExternABIBase;

  auto runOnOperation() -> void final {
    std::vector<ExternFunctionABI> functions;
    bool invalidDeclaration = false;

    getOperation().walk([&](func::FuncOp function) {
      auto layoutAttr =
          function->getAttrOfType<TypeAttr>(kExternLayoutTypeAttr);
      if (!layoutAttr)
        return;

      auto layoutType = llvm::dyn_cast<FunctionType>(layoutAttr.getValue());
      if (!layoutType) {
        function.emitError("extern layout attribute must be a FunctionType");
        invalidDeclaration = true;
        return;
      }

      auto sourceType = function.getFunctionType();
      if (sourceType.getNumInputs() != layoutType.getNumInputs() ||
          sourceType.getNumResults() != layoutType.getNumResults()) {
        function.emitError("extern source and layout signatures disagree");
        invalidDeclaration = true;
        return;
      }

      functions.push_back({function, sourceType, layoutType});
      function->setAttr("function_type", TypeAttr::get(layoutType));
      LLVM_DEBUG(llvm::dbgs() << "lower extern declaration `"
                              << function.getSymName() << "`\n");
    });

    if (invalidDeclaration) {
      signalPassFailure();
      return;
    }

    IRRewriter rewriter(&getContext());
    for (const auto &function : functions) {
      std::vector<func::CallOp> calls;
      auto functionName = function.function->getAttrOfType<StringAttr>(
          SymbolTable::getSymbolAttrName());
      getOperation().walk([&](func::CallOp call) {
        auto callee = call->getAttrOfType<FlatSymbolRefAttr>("callee");
        if (callee && functionName &&
            callee.getValue() == functionName.getValue())
          calls.push_back(call);
      });

      for (auto call : calls) {
        if (failed(lowerExternCall(call, function, rewriter))) {
          signalPassFailure();
          return;
        }
      }

      function.function->removeAttr(kExternLayoutTypeAttr);
    }
  }
};

} // namespace
} // namespace mlir::mulberry_core
