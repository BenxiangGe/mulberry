//===--- ConvertMulberryToLLVM.cpp - Mulberry-to-LLVM lowering -----------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "mulberry/MLIRGen/Conversion/MulberryPasses.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMInterfaces.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::mulberry {

#define GEN_PASS_DEF_CONVERTMULBERRYTOLLVM
#include "mulberry/MLIRGen/Conversion/MulberryPasses.h.inc"

namespace {

struct ConvertMulberryToLLVM
    : public impl::ConvertMulberryToLLVMBase<ConvertMulberryToLLVM> {
  using impl::ConvertMulberryToLLVMBase<
      ConvertMulberryToLLVM>::ConvertMulberryToLLVMBase;

  auto runOnOperation() -> void final {
    auto* context = &getContext();
    auto operation = getOperation();

    const auto& dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
    const auto& dataLayout = dataLayoutAnalysis.getAtOrAbove(operation);
    LowerToLLVMOptions options(context, dataLayout);
    LLVMTypeConverter typeConverter(context, options, &dataLayoutAnalysis);

    // `ptr.ptr` requires a MemorySpaceAttrInterface such as
    // `#llvm.address_space<0>`, while memref-to-LLVM needs an integer address
    // space. Teach the shared LLVMTypeConverter that both spellings represent
    // the same backend address space before memref/ptr patterns run.
    typeConverter.addTypeAttributeConversion(
        [](PtrLikeTypeInterface type,
           LLVM::LLVMAddrSpaceAttrInterface memorySpace)
            -> TypeConverter::AttributeConversionResult {
          if (type.getMemorySpace() != memorySpace)
            return TypeConverter::AttributeConversionResult::na();

          return IntegerAttr::get(IntegerType::get(type.getContext(), 32),
                                  memorySpace.getAddressSpace());
        });

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalDialect<LLVM::LLVMDialect>();
    populateConversionTargetFromOperation(operation, target, typeConverter,
                                          patterns);
    populateOpConvertToLLVMConversionPatterns(operation, target, typeConverter,
                                              patterns);

    if (failed(applyPartialConversion(operation, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::mulberry
