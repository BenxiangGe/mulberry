//===--- ConvertMulberryRecordToCIR.cpp ----------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/Conversion/CherryPasses.h"
#include "cherry/MLIRGen/IR/MulberryOps.h"
#include "cherry/MLIRGen/IR/MulberryTypeUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

#include <limits>
#include <set>
#include <vector>

namespace mlir::cherry {

#define GEN_PASS_DEF_CONVERTMULBERRYRECORDTOCIR
#include "cherry/MLIRGen/Conversion/CherryPasses.h.inc"

namespace {

auto operationUsesMulberryRecordType(Operation *op) -> bool {
  return containsMulberryRecordType(op->getOperandTypes()) ||
         containsMulberryRecordType(op->getResultTypes());
}

auto bodyUsesMulberryRecordType(func::FuncOp op) -> bool {
  // A func.func with a plain scalar signature can still contain Mulberry record
  // ops in its body. If we leave the func boundary alone, this pass creates a
  // mixed func.func + CIR body that the downstream CIR-to-LLVM pass rejects.
  auto result = op.walk([&](Operation *nestedOp) {
    if (nestedOp == op.getOperation())
      return WalkResult::advance();

    if (operationUsesMulberryRecordType(nestedOp))
      return WalkResult::interrupt();

    return WalkResult::advance();
  });

  return result.wasInterrupted();
}

auto functionUsesMulberryRecordType(func::FuncOp op) -> bool {
  return containsMulberryRecordType(op.getFunctionType()) ||
         bodyUsesMulberryRecordType(op);
}

class MulberryRecordTypeConverter : public TypeConverter {
public:
  MulberryRecordTypeConverter() {
    addConversion([](Type type) -> Type { return type; });
    addConversion([this](mulberry::PtrType type) -> Type {
      auto elementType = convertType(type.getElementType());
      if (!elementType)
        return {};
      return cir::PointerType::get(elementType);
    });
    addConversion([this](mulberry::RecordType type) -> Type {
      std::vector<Type> fields;
      for (const auto& field : type.getFields()) {
        auto fieldType = convertType(field.type);
        if (!fieldType)
          return {};
        fields.push_back(fieldType);
      }

      return cir::RecordType::get(type.getContext(), fields,
                                  StringAttr::get(type.getContext(),
                                                  type.getName()),
                                  /*packed=*/false, /*padded=*/false,
                                  cir::RecordType::Struct);
    });
  }
};

class AllocaOpLowering : public OpConversionPattern<mulberry::AllocaOp> {
public:
  using OpConversionPattern<mulberry::AllocaOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::AllocaOp op, OpAdaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto addrType = getTypeConverter()->convertType(op.getResult().getType());
    auto elementType = getTypeConverter()->convertType(op.getElementType());
    if (!addrType || !elementType)
      return failure();

    // Mulberry alloca has no alignment attribute yet. Keep the old codegen
    // behavior by emitting the weakest valid CIR alignment.
    rewriter.replaceOpWithNewOp<cir::AllocaOp>(
        op, addrType, elementType, "", rewriter.getI64IntegerAttr(1));
    return success();
  }
};

class LoadOpLowering : public OpConversionPattern<mulberry::LoadOp> {
public:
  using OpConversionPattern<mulberry::LoadOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::LoadOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    if (!llvm::isa<cir::PointerType>(adaptor.getPtr().getType()))
      return failure();

    rewriter.replaceOpWithNewOp<cir::LoadOp>(op, adaptor.getPtr());
    return success();
  }
};

class StoreOpLowering : public OpConversionPattern<mulberry::StoreOp> {
public:
  using OpConversionPattern<mulberry::StoreOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::StoreOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    if (!llvm::isa<cir::PointerType>(adaptor.getPtr().getType()))
      return failure();

    rewriter.replaceOpWithNewOp<cir::StoreOp>(
        op, adaptor.getValue(), adaptor.getPtr(), /*isVolatile=*/false,
        /*alignment=*/IntegerAttr{}, /*sync_scope=*/cir::SyncScopeKindAttr(),
        /*mem-order=*/cir::MemOrderAttr());
    return success();
  }
};

class RecordGetFieldOpLowering
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
    if (fieldIndex == std::numeric_limits<unsigned>::max())
      return failure();

    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType || !llvm::isa<cir::PointerType>(resultType))
      return failure();

    rewriter.replaceOpWithNewOp<cir::GetMemberOp>(
        op, resultType, adaptor.getRecord(), op.getField(), fieldIndex);
    return success();
  }
};

class RecordCreateOpLowering
    : public OpConversionPattern<mulberry::RecordCreateOp> {
public:
  using OpConversionPattern<mulberry::RecordCreateOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::RecordCreateOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto recordType = llvm::dyn_cast<mulberry::RecordType>(
        op.getResult().getType());
    if (!recordType)
      return failure();

    auto convertedRecordType = getTypeConverter()->convertType(recordType);
    if (!convertedRecordType)
      return failure();

    auto ptrType = cir::PointerType::get(convertedRecordType);
    // CIR has no aggregate value builder op, so construct the value through a
    // temporary record slot and load it back. Direct LLVM lowering uses
    // insertvalue instead.
    auto recordAlloca = cir::AllocaOp::create(
        rewriter, op.getLoc(), ptrType, convertedRecordType, "",
        rewriter.getI64IntegerAttr(1));
    auto recordPtr = recordAlloca.getAddr();

    for (auto field : llvm::enumerate(recordType.getFields())) {
      auto fieldType = getTypeConverter()->convertType(field.value().type);
      if (!fieldType)
        return failure();
      auto fieldPtrType = cir::PointerType::get(fieldType);
      auto fieldPtr = cir::GetMemberOp::create(
          rewriter, op.getLoc(), fieldPtrType, recordPtr,
          field.value().name, field.index());
      cir::StoreOp::create(
          rewriter, op.getLoc(), adaptor.getFields()[field.index()], fieldPtr,
          /*isVolatile=*/false, /*alignment=*/IntegerAttr{},
          /*sync_scope=*/cir::SyncScopeKindAttr(),
          /*mem-order=*/cir::MemOrderAttr());
    }

    auto loadedRecord = cir::LoadOp::create(
        rewriter, op.getLoc(), convertedRecordType, recordPtr,
        /*isDeref=*/false, /*isVolatile=*/false, /*alignment=*/IntegerAttr{},
        /*sync_scope=*/cir::SyncScopeKindAttr(),
        /*mem-order=*/cir::MemOrderAttr());
    rewriter.replaceOp(op, loadedRecord);
    return success();
  }
};

class FuncOpLowering : public OpConversionPattern<func::FuncOp> {
public:
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  auto matchAndRewrite(func::FuncOp op, OpAdaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    std::vector<Type> inputs;
    for (auto input : op.getFunctionType().getInputs()) {
      auto convertedInput = getTypeConverter()->convertType(input);
      if (!convertedInput)
        return failure();
      inputs.push_back(convertedInput);
    }

    std::vector<Type> results;
    for (auto result : op.getFunctionType().getResults()) {
      auto convertedResult = getTypeConverter()->convertType(result);
      if (!convertedResult)
        return failure();
      results.push_back(convertedResult);
    }

    if (results.size() > 1)
      return op.emitError("CIR functions support at most one result");

    auto returnType = results.empty()
                          ? cir::VoidType::get(rewriter.getContext())
                          : results.front();
    auto funcType = cir::FuncType::get(inputs, returnType);
    auto newFunc =
        cir::FuncOp::create(rewriter, op.getLoc(), op.getName(), funcType);

    rewriter.inlineRegionBefore(op.getBody(), newFunc.getBody(),
                                newFunc.end());
    if (!newFunc.getBody().empty()) {
      TypeConverter::SignatureConversion signatureConversion(
          op.getNumArguments());
      for (auto input : llvm::enumerate(inputs))
        signatureConversion.addInputs(input.index(), input.value());
      rewriter.applySignatureConversion(&newFunc.getBody().front(),
                                        signatureConversion,
                                        getTypeConverter());
    }

    rewriter.eraseOp(op);
    return success();
  }
};

class FuncCallOpLowering : public OpConversionPattern<func::CallOp> {
public:
  using OpConversionPattern<func::CallOp>::OpConversionPattern;

  auto matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    Type resultType;
    if (op.getNumResults() > 1)
      return op.emitError("CIR calls support at most one result");
    if (op.getNumResults() == 1) {
      resultType = getTypeConverter()->convertType(op.getResultTypes().front());
      if (!resultType)
        return failure();
    }

    auto callee = SymbolRefAttr::get(rewriter.getContext(), op.getCallee());
    rewriter.replaceOpWithNewOp<cir::CallOp>(op, callee, resultType,
                                             adaptor.getOperands());
    return success();
  }
};

class FuncReturnOpLowering : public OpConversionPattern<func::ReturnOp> {
public:
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  auto matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    rewriter.replaceOpWithNewOp<cir::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

struct ConvertMulberryRecordToCIR
    : public impl::ConvertMulberryRecordToCIRBase<
          ConvertMulberryRecordToCIR> {

  using impl::ConvertMulberryRecordToCIRBase<
      ConvertMulberryRecordToCIR>::ConvertMulberryRecordToCIRBase;

  auto runOnOperation() -> void final {
    MulberryRecordTypeConverter typeConverter;

    // Dynamic legality is queried after nested ops may already have been
    // rewritten, so remember the original func.func boundary decisions first.
    std::set<Operation *> funcsToLower;
    getOperation()->walk([&](func::FuncOp op) {
      if (functionUsesMulberryRecordType(op))
        funcsToLower.insert(op.getOperation());
    });
    auto needsCIRFunctionLowering = [&](func::FuncOp op) {
      return funcsToLower.find(op.getOperation()) != funcsToLower.end();
    };

    ConversionTarget target(getContext());
    target.addLegalDialect<cir::CIRDialect, func::FuncDialect>();
    target.addIllegalDialect<mulberry::MulberryDialect>();
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return !needsCIRFunctionLowering(op);
    });
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      if (auto funcOp = op->getParentOfType<func::FuncOp>())
        if (needsCIRFunctionLowering(funcOp))
          return false;
      return !containsMulberryRecordType(op.getOperandTypes()) &&
             !containsMulberryRecordType(op.getResultTypes());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      auto funcOp = op->getParentOfType<func::FuncOp>();
      return funcOp && !needsCIRFunctionLowering(funcOp);
    });
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(&getContext());
    patterns.add<AllocaOpLowering, LoadOpLowering, StoreOpLowering,
                 RecordGetFieldOpLowering, RecordCreateOpLowering,
                 FuncOpLowering, FuncCallOpLowering, FuncReturnOpLowering>(
        typeConverter, &getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::cherry
