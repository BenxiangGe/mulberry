//===--- ConvertCherryToLLVM.cpp ------------------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/Conversion/CherryPasses.h"
#include "cherry/MLIRGen/IR/CherryNNOps.h"
#include "cherry/MLIRGen/IR/MulberryOps.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

#include <cstdint>
#include <limits>
#include <vector>

namespace mlir::cherry {

#define GEN_PASS_DEF_CONVERTCHERRYTOLLVM
#include "cherry/MLIRGen/Conversion/CherryPasses.h.inc"

namespace {

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
                           ? rewriter.create<memref::LoadOp>(loc, operand)
                           : adaptor.getInput();

    auto cast =
        rewriter.create<LLVM::ZExtOp>(loc, rewriter.getI64Type(), newOperand);
    rewriter.replaceOp(op, cast.getRes());
    return llvm::success();
  }
};

class NNCastOpLowering : public OpConversionPattern<cherry_nn::CastOp> {
public:
  explicit NNCastOpLowering(LLVMTypeConverter &typeConverter,
                            MLIRContext *context)
      : OpConversionPattern<cherry_nn::CastOp>(typeConverter, context) {}

  auto matchAndRewrite(cherry_nn::CastOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult final {
    if (!((llvm::isa<IntegerType>(op.getInput().getType()) &&
           llvm::isa<cir::IntType>(op.getResult().getType())) ||
          (llvm::isa<cir::IntType>(op.getInput().getType()) &&
           llvm::isa<IntegerType>(op.getResult().getType()))))
      return failure();

    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

class PrintOpLowering : public OpConversionPattern<cherry::PrintOp> {
public:
  explicit PrintOpLowering(LLVMTypeConverter &typeConverter,
                           MLIRContext *context)
      : OpConversionPattern<cherry::PrintOp>(typeConverter, context) {}

  auto matchAndRewrite(cherry::PrintOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult final {
    auto loc = op->getLoc();

    // Get a symbol reference to the printf function, inserting it if necessary.
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto printfRef = getOrInsertPrintf(rewriter, parentModule);
    Value formatSpecifierCst = getOrCreateGlobalString(
        loc, rewriter, "frmt_spec", StringRef("%llu\n\0", 6), parentModule);

    auto operand = op.getInput();
    Value newOperand = llvm::isa<MemRefType>(operand.getType())
                           ? rewriter.create<memref::LoadOp>(loc, operand)
                           : adaptor.getInput();

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, getPrintfType(rewriter.getContext()), printfRef,
        ArrayRef<Value>({formatSpecifierCst, newOperand}));
    return success();
  }

private:
  static LLVM::LLVMFunctionType getPrintfType(MLIRContext *context) {
    auto llvmI32Ty = IntegerType::get(context, 32);
    auto llvmPtrTy = LLVM::LLVMPointerType::get(context);
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmPtrTy,
                                                  /*isVarArg=*/true);
    return llvmFnType;
  }

  static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                             ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
      return SymbolRefAttr::get(context, "printf");

    // Insert the printf function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf",
                                      getPrintfType(context));
    return SymbolRefAttr::get(context, "printf");
  }

  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
  static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                       StringRef name, StringRef value,
                                       ModuleOp module) {
    // Create the global at the entry of the module.
    LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
      OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = LLVM::LLVMArrayType::get(
          IntegerType::get(builder.getContext(), 8), value.size());
      global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                              LLVM::Linkage::Internal, name,
                                              builder.getStringAttr(value));
    }

    // Get the pointer to the first character in the global string.
    Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
    Value cst0 = builder.create<LLVM::ConstantOp>(
        loc, IntegerType::get(builder.getContext(), 64),
        builder.getIntegerAttr(builder.getIndexType(), 0));
    return builder.create<LLVM::GEPOp>(
        loc, LLVM::LLVMPointerType::get(builder.getContext()), global.getType(),
        globalPtr, ArrayRef<Value>({cst0, cst0}));
  }
};

class MulberryAllocaOpLowering
    : public OpConversionPattern<mulberry::AllocaOp> {
public:
  using OpConversionPattern<mulberry::AllocaOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::AllocaOp op, OpAdaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto ptrType = getTypeConverter()->convertType(op.getResult().getType());
    auto elementType = getTypeConverter()->convertType(op.getElementType());
    if (!ptrType || !elementType)
      return failure();

    auto one = LLVM::ConstantOp::create(rewriter, op.getLoc(),
                                        rewriter.getI64Type(),
                                        rewriter.getI64IntegerAttr(1));
    rewriter.replaceOpWithNewOp<LLVM::AllocaOp>(
        op, ptrType, elementType, one, /*alignment=*/0);
    return success();
  }
};

class MulberryLoadOpLowering : public OpConversionPattern<mulberry::LoadOp> {
public:
  using OpConversionPattern<mulberry::LoadOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::LoadOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return failure();

    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, resultType,
                                              adaptor.getPtr());
    return success();
  }
};

class MulberryStoreOpLowering : public OpConversionPattern<mulberry::StoreOp> {
public:
  using OpConversionPattern<mulberry::StoreOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::StoreOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, adaptor.getValue(),
                                               adaptor.getPtr());
    return success();
  }
};

class MulberryRecordGetFieldOpLowering
    : public OpConversionPattern<mulberry::RecordGetFieldOp> {
public:
  using OpConversionPattern<mulberry::RecordGetFieldOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::RecordGetFieldOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto sourcePtrType =
        llvm::dyn_cast<mulberry::PtrType>(op.getRecord().getType());
    if (!sourcePtrType)
      return failure();

    auto sourceRecordType =
        llvm::dyn_cast<mulberry::RecordType>(sourcePtrType.getElementType());
    if (!sourceRecordType)
      return failure();

    auto fieldIndex = sourceRecordType.getFieldIndex(op.getField());
    if (fieldIndex == std::numeric_limits<unsigned>::max() ||
        fieldIndex > static_cast<unsigned>(std::numeric_limits<int32_t>::max()))
      return failure();

    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    auto recordType = getTypeConverter()->convertType(sourceRecordType);
    if (!resultType || !recordType)
      return failure();

    std::vector<LLVM::GEPArg> indices = {0, static_cast<int32_t>(fieldIndex)};
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
        op, resultType, recordType, adaptor.getRecord(), indices);
    return success();
  }
};

class MulberryRecordCreateOpLowering
    : public OpConversionPattern<mulberry::RecordCreateOp> {
public:
  using OpConversionPattern<mulberry::RecordCreateOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::RecordCreateOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return failure();

    Value result = LLVM::UndefOp::create(rewriter, op.getLoc(), resultType);
    for (auto field : llvm::enumerate(adaptor.getFields()))
      result = LLVM::InsertValueOp::create(rewriter, op.getLoc(), result,
                                           field.value(), field.index());

    rewriter.replaceOp(op, result);
    return success();
  }
};

class MulberryTensorPackOpLowering
    : public OpConversionPattern<mulberry::TensorPackOp> {
public:
  using OpConversionPattern<mulberry::TensorPackOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::TensorPackOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto descriptorType =
        llvm::dyn_cast<mulberry::RecordType>(op.getResult().getType());
    auto memRefType = llvm::dyn_cast<MemRefType>(op.getTensor().getType());
    if (!descriptorType || !memRefType)
      return failure();

    auto resultType = getTypeConverter()->convertType(descriptorType);
    auto sizesType =
        getTypeConverter()->convertType(descriptorType.getFieldType("sizes"));
    auto stridesType =
        getTypeConverter()->convertType(descriptorType.getFieldType("strides"));
    if (!resultType || !sizesType || !stridesType)
      return failure();

    auto loc = op.getLoc();
    auto tensor = adaptor.getTensor();
    auto allocated = LLVM::ExtractValueOp::create(rewriter, loc, tensor, 0);
    auto aligned = LLVM::ExtractValueOp::create(rewriter, loc, tensor, 1);
    auto offset = LLVM::ExtractValueOp::create(rewriter, loc, tensor, 2);

    Value sizes = LLVM::UndefOp::create(rewriter, loc, sizesType);
    Value strides = LLVM::UndefOp::create(rewriter, loc, stridesType);
    for (int64_t i = 0; i < memRefType.getRank(); ++i) {
      // MemRef-to-LLVM lowers ranked memrefs to:
      // {allocated, aligned, offset, sizes[rank], strides[rank]}.
      // See mlir/lib/Conversion/LLVMCommon/MemRefDescriptor.h.
      auto size = LLVM::ExtractValueOp::create(
          rewriter, loc, tensor, ArrayRef<int64_t>{3, i});
      sizes = LLVM::InsertValueOp::create(rewriter, loc, sizes, size, i);

      auto stride = LLVM::ExtractValueOp::create(
          rewriter, loc, tensor, ArrayRef<int64_t>{4, i});
      strides = LLVM::InsertValueOp::create(rewriter, loc, strides, stride, i);
    }

    Value descriptor = LLVM::UndefOp::create(rewriter, loc, resultType);
    descriptor = LLVM::InsertValueOp::create(
        rewriter, loc, descriptor, allocated, ArrayRef<int64_t>{0});
    descriptor = LLVM::InsertValueOp::create(
        rewriter, loc, descriptor, aligned, ArrayRef<int64_t>{1});
    descriptor = LLVM::InsertValueOp::create(
        rewriter, loc, descriptor, offset, ArrayRef<int64_t>{2});
    descriptor = LLVM::InsertValueOp::create(
        rewriter, loc, descriptor, sizes, ArrayRef<int64_t>{3});
    descriptor = LLVM::InsertValueOp::create(
        rewriter, loc, descriptor, strides, ArrayRef<int64_t>{4});

    rewriter.replaceOp(op, descriptor);
    return success();
  }
};

class MulberryTensorUnpackOpLowering
    : public OpConversionPattern<mulberry::TensorUnpackOp> {
public:
  using OpConversionPattern<mulberry::TensorUnpackOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::TensorUnpackOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto descriptorType =
        llvm::dyn_cast<mulberry::RecordType>(op.getTensor().getType());
    auto memRefType = llvm::dyn_cast<MemRefType>(op.getResult().getType());
    if (!descriptorType || !memRefType)
      return failure();

    auto sizesType =
        llvm::dyn_cast_if_present<mulberry::RecordType>(
            descriptorType.getFieldType("sizes"));
    auto stridesType =
        llvm::dyn_cast_if_present<mulberry::RecordType>(
            descriptorType.getFieldType("strides"));
    if (!sizesType || !stridesType)
      return failure();

    auto loc = op.getLoc();
    auto tensor = adaptor.getTensor();
    auto allocated = LLVM::ExtractValueOp::create(
        rewriter, loc, tensor, descriptorType.getFieldIndex("allocated"));
    auto aligned = LLVM::ExtractValueOp::create(
        rewriter, loc, tensor, descriptorType.getFieldIndex("aligned"));
    auto offset = LLVM::ExtractValueOp::create(
        rewriter, loc, tensor, descriptorType.getFieldIndex("offset"));
    auto sizes = LLVM::ExtractValueOp::create(
        rewriter, loc, tensor, descriptorType.getFieldIndex("sizes"));
    auto strides = LLVM::ExtractValueOp::create(
        rewriter, loc, tensor, descriptorType.getFieldIndex("strides"));

    // Rebuild the LLVM memref descriptor from Mulberry's named descriptor.
    // MLIR represents ranked memrefs as:
    // {allocated, aligned, offset, sizes[rank], strides[rank]}.
    // See mlir/lib/Conversion/LLVMCommon/MemRefDescriptor.h.
    std::vector<Value> descriptorFields = {allocated, aligned, offset};
    for (unsigned i = 0; i < sizesType.getNumFields(); ++i) {
      descriptorFields.push_back(
          LLVM::ExtractValueOp::create(rewriter, loc, sizes, i));
    }
    for (unsigned i = 0; i < stridesType.getNumFields(); ++i) {
      descriptorFields.push_back(
          LLVM::ExtractValueOp::create(rewriter, loc, strides, i));
    }

    auto memRefDescriptor = MemRefDescriptor::pack(
        rewriter, loc, *static_cast<const LLVMTypeConverter *>(
                           getTypeConverter()),
        memRefType, descriptorFields);
    rewriter.replaceOp(op, memRefDescriptor);
    return success();
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
    target.addIllegalDialect<mulberry::MulberryDialect>();

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
    typeConverter.addConversion([&](mulberry::PtrType type) -> Type {
      return LLVM::LLVMPointerType::get(type.getContext());
    });
    typeConverter.addConversion([&](mulberry::RecordType type) -> Type {
      std::vector<Type> fields;
      for (const auto& field : type.getFields()) {
        auto fieldType = typeConverter.convertType(field.type);
        if (!fieldType)
          return {};
        fields.push_back(fieldType);
      }

      // This direct LLVM path is a foundation path for getting rid of the CIR
      // record bridge later. Keep ABI naming out of it until function lowering
      // is ready to own named LLVM structs.
      return LLVM::LLVMStructType::getLiteral(type.getContext(), fields);
    });

    // Patterns
    RewritePatternSet patterns(&getContext());
    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    populateSCFToControlFlowConversionPatterns(patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    populateMathToLLVMConversionPatterns(typeConverter, patterns);
    patterns
        .add<CastOpLowering, NNCastOpLowering, PrintOpLowering,
             MulberryAllocaOpLowering, MulberryLoadOpLowering,
             MulberryStoreOpLowering, MulberryRecordGetFieldOpLowering,
             MulberryRecordCreateOpLowering, MulberryTensorPackOpLowering,
             MulberryTensorUnpackOpLowering>(
            typeConverter, &getContext());

    // Conversion
    auto module = getOperation();
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyFullConversion(module, target, patternSet)))
      signalPassFailure();
  }
};

} // end namespace
} // namespace mlir::cherry
