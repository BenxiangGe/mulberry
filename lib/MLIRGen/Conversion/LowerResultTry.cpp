//===--- LowerResultTry.cpp -----------------------------------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "mulberry/MLIRGen/Conversion/MulberryPasses.h"
#include "LowerMulberrySupport.h"
#include "mulberry/MLIRGen/IR/MulberryDialect.h"
#include "mulberry/MLIRGen/IR/MulberryOps.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <iterator>
#include <vector>

namespace mlir::mulberry_core {

#define GEN_PASS_DEF_LOWERRESULTTRY
#include "mulberry/MLIRGen/Conversion/MulberryPasses.h.inc"

#undef DEBUG_TYPE
#define DEBUG_TYPE "lower-result-try"

namespace {

auto createDataLayout(MLIRContext *context, ArrayRef<Type> payloadTypes)
    -> LLVM::LLVMStructType {
  std::vector<Type> fields{IntegerType::get(context, 64)};
  fields.insert(fields.end(), payloadTypes.begin(), payloadTypes.end());
  return LLVM::LLVMStructType::getLiteral(context, fields);
}

auto unpackDataPayload(Location location, IRRewriter &rewriter, Value value,
                       ArrayRef<Type> payloadTypes)
    -> FailureOr<std::vector<Value>> {
  auto ptrType = llvm::dyn_cast<LLVM::LLVMPointerType>(value.getType());
  if (!ptrType)
    return failure();

  auto layout = createDataLayout(rewriter.getContext(), payloadTypes);
  std::vector<Value> payloads;
  for (size_t index = 0; index < payloadTypes.size(); ++index) {
    auto field = LLVM::GEPOp::create(
        rewriter, location, ptrType, layout, value,
        ArrayRef<LLVM::GEPArg>{0, static_cast<int32_t>(index + 1)});
    payloads.push_back(LLVM::LoadOp::create(rewriter, location,
                                            payloadTypes[index], field));
  }
  return payloads;
}

auto constructDataValue(Location location, IRRewriter &rewriter,
                        Operation *anchor, ArrayRef<Type> payloadTypes,
                        ArrayRef<Value> payloads, int64_t tagValue)
    -> FailureOr<Value> {
  auto layout = createDataLayout(rewriter.getContext(), payloadTypes);
  auto size = detail::createSizeOf(location, rewriter, layout);
  auto object = detail::callBoehmMalloc(location, rewriter, anchor, size);
  if (failed(object))
    return failure();

  auto tag = LLVM::ConstantOp::create(
      rewriter, location, IntegerType::get(rewriter.getContext(), 64),
      rewriter.getI64IntegerAttr(tagValue));
  Value value = LLVM::UndefOp::create(rewriter, location, layout);
  value = LLVM::InsertValueOp::create(rewriter, location, value, tag,
                                      ArrayRef<int64_t>{0});
  for (auto indexedPayload : llvm::enumerate(payloads))
    value = LLVM::InsertValueOp::create(
        rewriter, location, value, indexedPayload.value(),
        ArrayRef<int64_t>{static_cast<int64_t>(indexedPayload.index() + 1)});
  LLVM::StoreOp::create(rewriter, location, value, *object);
  return *object;
}

auto containsResultTry(Operation *operation) -> bool {
  auto containsTry = false;
  operation->walk([&](ResultTryOp) { containsTry = true; });
  return containsTry;
}

auto lowerContainingSCF(func::FuncOp function) -> LogicalResult {
  RewritePatternSet patterns(function.getContext());
  populateSCFToControlFlowConversionPatterns(patterns);

  ConversionTarget target(*function.getContext());
  // Tensor bufferization still needs unrelated loops in structured form. Only
  // the SCF ancestors that must carry an Err path to the function exit need CFG.
  target.addDynamicallyLegalOp<
      scf::ForallOp, scf::ForOp, scf::IfOp, scf::IndexSwitchOp,
      scf::ParallelOp, scf::WhileOp, scf::ExecuteRegionOp>(
      [](Operation *operation) { return !containsResultTry(operation); });
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  return applyPartialConversion(function, target, std::move(patterns));
}

auto lowerResultTry(ResultTryOp op, IRRewriter &rewriter) -> LogicalResult {
  auto function = op->getParentOfType<func::FuncOp>();
  if (!function || function.getFunctionType().getNumResults() != 1)
    return op.emitOpError("needs an enclosing single-result function");

  auto location = op.getLoc();
  auto input = op.getInput();
  std::vector<Type> valueTypes(op.getResultTypes().begin(),
                               op.getResultTypes().end());
  auto sourceErrorType = op.getSourceErrorType();
  auto targetErrorType = op.getTargetErrorType();
  auto functionResultType = function.getFunctionType().getResult(0);

  auto *currentBlock = op->getBlock();
  auto *continuationBlock = rewriter.splitBlock(
      currentBlock, std::next(Block::iterator(op.getOperation())));
  if (!valueTypes.empty()) {
    auto continuationValue =
        continuationBlock->addArgument(valueTypes.front(), location);
    op.getResult(0).replaceAllUsesWith(continuationValue);
  }

  auto *region = currentBlock->getParent();
  auto *okBlock = rewriter.createBlock(
      region, Region::iterator(continuationBlock));
  auto *errorBlock = rewriter.createBlock(
      region, Region::iterator(continuationBlock));

  auto inputPtrType = llvm::dyn_cast<LLVM::LLVMPointerType>(input.getType());
  if (!inputPtrType || !llvm::isa<LLVM::LLVMPointerType>(functionResultType))
    return op.emitOpError("must be lowered to storage pointer types first");

  rewriter.setInsertionPoint(op);
  auto tag = LLVM::LoadOp::create(
      rewriter, location, IntegerType::get(rewriter.getContext(), 64), input);
  auto okTag = arith::ConstantIntOp::create(
      rewriter, location, 0, 64);
  auto isOk = arith::CmpIOp::create(
      rewriter, location, arith::CmpIPredicate::eq, tag, okTag);
  cf::CondBranchOp::create(
      rewriter, location, isOk, okBlock, ValueRange{}, errorBlock,
      ValueRange{});

  rewriter.setInsertionPointToEnd(okBlock);
  auto okPayloads = unpackDataPayload(location, rewriter, input, valueTypes);
  if (failed(okPayloads))
    return op.emitOpError("cannot unpack the Ok payload");
  cf::BranchOp::create(rewriter, location, continuationBlock,
                       *okPayloads);

  rewriter.setInsertionPointToEnd(errorBlock);
  std::vector<Type> errorTypes;
  if (!llvm::isa<NoneType>(sourceErrorType))
    errorTypes.push_back(sourceErrorType);
  auto errorPayloads =
      unpackDataPayload(location, rewriter, input, errorTypes);
  if (failed(errorPayloads))
    return op.emitOpError("cannot unpack the Err payload");
  std::vector<Value> propagatedErrors;
  if (op.getErrorConverterAttr()) {
    auto sourceIsUnit = llvm::isa<NoneType>(sourceErrorType);
    if (errorPayloads->empty() != sourceIsUnit)
      return op.emitOpError("converter requires a source error payload");
    std::vector<Type> converterResultTypes;
    if (!llvm::isa<NoneType>(targetErrorType))
      converterResultTypes.push_back(targetErrorType);
    auto converted = func::CallOp::create(
        rewriter, location, op.getErrorConverterAttr().getValue(),
        converterResultTypes, *errorPayloads);
    if (!converterResultTypes.empty())
      propagatedErrors.push_back(converted.getResult(0));
    LLVM_DEBUG(llvm::dbgs() << "apply Result error converter `"
                            << op.getErrorConverterAttr().getValue()
                            << "` in function `" << function.getSymName()
                            << "`\n");
  } else {
    for (auto payload : *errorPayloads)
      propagatedErrors.push_back(payload);
  }
  std::vector<Type> propagatedErrorTypes;
  for (auto payload : propagatedErrors)
    propagatedErrorTypes.push_back(payload.getType());
  auto propagated = constructDataValue(
      location, rewriter, op, propagatedErrorTypes, propagatedErrors, 1);
  if (failed(propagated))
    return op.emitOpError("cannot construct the propagated Err value");
  // Native Tensor storage has a GC finalizer, so locals skipped by this direct
  // error return safely fall back to GC-managed release.
  func::ReturnOp::create(rewriter, location, *propagated);

  LLVM_DEBUG(llvm::dbgs() << "lower `?` in function `"
                          << function.getSymName() << "`\n");
  rewriter.eraseOp(op);
  return success();
}

struct LowerResultTry : public impl::LowerResultTryBase<LowerResultTry> {
  using impl::LowerResultTryBase<LowerResultTry>::LowerResultTryBase;

  auto runOnOperation() -> void final {
    std::vector<func::FuncOp> functions;
    getOperation().walk([&](ResultTryOp op) {
      auto function = op->getParentOfType<func::FuncOp>();
      if (std::find(functions.begin(), functions.end(), function) ==
          functions.end())
        functions.push_back(function);
    });

    for (auto function : functions) {
      if (failed(lowerContainingSCF(function))) {
        signalPassFailure();
        return;
      }

      std::vector<ResultTryOp> tryOps;
      function.walk([&](ResultTryOp op) { tryOps.push_back(op); });
      IRRewriter rewriter(&getContext());
      for (auto op : tryOps) {
        if (failed(lowerResultTry(op, rewriter))) {
          signalPassFailure();
          return;
        }
      }
    }
  }
};

} // namespace
} // namespace mlir::mulberry_core
