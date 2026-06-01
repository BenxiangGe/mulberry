//===--- ConvertCherryToLLVM.cpp ------------------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/Conversion/CherryPasses.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

namespace mlir::cherry {

#define GEN_PASS_DEF_CONVERTCHERRYTOLLVM
#include "cherry/MLIRGen/Conversion/CherryPasses.h.inc"

namespace {

auto isCIRToMLIRScalarBridge(Type resultType, Type originalType) -> bool {
  if (auto originalIntType = llvm::dyn_cast<cir::IntType>(originalType)) {
    auto resultIntType = llvm::dyn_cast<IntegerType>(resultType);
    return resultIntType &&
           resultIntType.getWidth() == originalIntType.getWidth();
  }

  if (llvm::isa<cir::BoolType>(originalType)) {
    auto resultIntType = llvm::dyn_cast<IntegerType>(resultType);
    return resultIntType && resultIntType.getWidth() == 1;
  }

  return llvm::isa<cir::SingleType>(originalType) &&
         llvm::isa<Float32Type>(resultType);
}

class CastOpLowering : public OpConversionPattern<cherry::CastOp> {
public:
  explicit CastOpLowering(LLVMTypeConverter &typeConverter,
                          MLIRContext *context)
      : OpConversionPattern<cherry::CastOp>(typeConverter, context) {}

  auto matchAndRewrite(cherry::CastOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult final {
    auto loc = op->getLoc();
    auto operand = op.getInput();
    Value newOperand = llvm::isa<MemRefType>(operand.getType())
                           ? memref::LoadOp::create(rewriter, loc, operand)
                           : adaptor.getInput();

    auto cast =
        LLVM::ZExtOp::create(rewriter, loc, rewriter.getI64Type(), newOperand);
    rewriter.replaceOp(op, cast.getRes());
    return llvm::success();
  }
};

struct ConvertCherryToLLVM
    : public impl::ConvertCherryToLLVMBase<ConvertCherryToLLVM> {

  using impl::ConvertCherryToLLVMBase<
      ConvertCherryToLLVM>::ConvertCherryToLLVMBase;

  auto runOnOperation() -> void final {
    // Target
    LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();
    // The driver runs this before CIR lowering when Tensor ABI functions use
    // func.func. Leave the remaining CIR island legal here; ClangIR's own
    // lowering pass handles it immediately afterwards.
    target.addLegalDialect<cir::CIRDialect>();
    target.addLegalOp<BridgeCastOp>();
    target.addLegalOp<PrintOp>();

    // Types conversions
    LLVMTypeConverter typeConverter(&getContext());

    typeConverter.addConversion([&](cir::IntType type) -> Type {
      return IntegerType::get(type.getContext(), type.getWidth());
    });
    typeConverter.addConversion([&](cir::BoolType type) -> Type {
      return IntegerType::get(type.getContext(), 1);
    });
    typeConverter.addConversion([&](cir::SingleType type) -> Type {
      return Float32Type::get(type.getContext());
    });
    typeConverter.addConversion([&](cir::VoidType type) -> Type {
      return LLVM::LLVMVoidType::get(type.getContext());
    });
    typeConverter.addTargetMaterialization(
        [&](OpBuilder &builder, Type resultType, ValueRange inputs,
            Location loc, Type originalType) -> Value {
          if (inputs.size() != 1 ||
              !isCIRToMLIRScalarBridge(resultType, originalType))
            return {};

          return BridgeCastOp::create(builder, loc, resultType, inputs.front());
        });

    // Patterns
    RewritePatternSet patterns(&getContext());
    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    populateSCFToControlFlowConversionPatterns(patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    populateMathToLLVMConversionPatterns(typeConverter, patterns);
    patterns.add<CastOpLowering>(typeConverter, &getContext());

    // Conversion
    auto module = getOperation();
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyFullConversion(module, target, patternSet)))
      signalPassFailure();
  }
};

} // end namespace
} // namespace mlir::cherry
