//===--- LowerMulberry.cpp -----------------------------------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "LowerMulberrySupport.h"

#include "mulberry/BigInt/BigIntOps.h"
#include "mulberry/MLIRGen/Conversion/MulberryPasses.h"
#include "mulberry/MLIRGen/IR/MulberryDialect.h"
#include "mulberry/MLIRGen/IR/MulberryOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/ControlFlow/Transforms/StructuralTypeConversions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <utility>

namespace mlir::mulberry_core {

namespace bigint = ::mlir::bigint;

#define GEN_PASS_DEF_FINALIZEMULBERRYTENSORSTORAGE
#define GEN_PASS_DEF_LOWERMULBERRY
#include "mulberry/MLIRGen/Conversion/MulberryPasses.h.inc"

namespace {

struct FinalizeMulberryTensorStorage
    : public impl::FinalizeMulberryTensorStorageBase<
          FinalizeMulberryTensorStorage> {
  using impl::FinalizeMulberryTensorStorageBase<
      FinalizeMulberryTensorStorage>::FinalizeMulberryTensorStorageBase;

  auto runOnOperation() -> void final {
    RewritePatternSet patterns(&getContext());
    detail::populateFinalizeMulberryTensorStoragePatterns(
        patterns, &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

struct LowerMulberry : public impl::LowerMulberryBase<LowerMulberry> {
  using impl::LowerMulberryBase<LowerMulberry>::LowerMulberryBase;

  auto runOnOperation() -> void final {
    detail::MulberryTypeConverter typeConverter;

    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, cf::ControlFlowDialect,
                           func::FuncDialect, linalg::LinalgDialect,
                           LLVM::LLVMDialect, math::MathDialect,
                           memref::MemRefDialect, scf::SCFDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addLegalOp<TensorStorageAllocLoweredOp>();
    target.addIllegalDialect<MulberryDialect>();
    target.addIllegalDialect<bigint::BigIntDialect>();
    target.addDynamicallyLegalOp<func::FuncOp>(
        [&](func::FuncOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType());
        });
    target.addDynamicallyLegalOp<func::CallOp, func::ReturnOp>(
        [&](Operation* op) {
          return typeConverter.isLegal(op);
        });
    target.addDynamicallyLegalOp<func::ConstantOp, func::CallIndirectOp>(
        [&](Operation* op) {
          return typeConverter.isLegal(op);
        });

    RewritePatternSet patterns(&getContext());
    detail::populateMulberryCoreLoweringPatterns(
        patterns, typeConverter, &getContext());
    detail::populateMulberryBigIntLoweringPatterns(
        patterns, typeConverter, &getContext());
    detail::populateMulberryTensorLoweringPatterns(
        patterns, typeConverter, &getContext());
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    cf::populateCFStructuralTypeConversionsAndLegality(
        typeConverter, patterns, target);
    scf::populateSCFStructuralTypeConversionsAndLegality(
        typeConverter, patterns, target);

    // Keep unresolved Mulberry ops fail-fast until each one has an explicit
    // lowering. A silent no-op pass would make backend tests look lower than
    // they are.
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPartialConversion(getOperation(), target, patternSet)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::mulberry_core
