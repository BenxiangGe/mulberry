//===--- LowerMulberryBigInt.cpp ------------------------------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "LowerMulberrySupport.h"

#include "mulberry/BigInt/BigIntOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

#include <string>

namespace mlir::mulberry_core::detail {

namespace bigint = ::mlir::bigint;

namespace {

auto createBigIntLiteral(Location location,
                         ConversionPatternRewriter& rewriter,
                         Operation* op, llvm::StringRef spelling)
    -> FailureOr<Value> {
  auto moduleOp = op->getParentOfType<ModuleOp>();
  if (!moduleOp)
    return failure();

  std::string globalName = "__mulberry_bigint_literal";
  for (size_t suffix = 0; moduleOp.lookupSymbol(globalName); ++suffix)
    globalName = "__mulberry_bigint_literal_" +
                 std::to_string(suffix + 1);

  auto spellingAddress = LLVM::createGlobalString(
      location, rewriter, globalName, spelling, LLVM::Linkage::Private);
  auto length = LLVM::ConstantOp::create(
      rewriter, location, getI64Type(rewriter.getContext()),
      rewriter.getI64IntegerAttr(static_cast<int64_t>(spelling.size())));
  return callRuntimeValue(location, rewriter, op,
                          "mulberry_bigint_from_literal",
                          getPtrType(rewriter.getContext()),
                          ValueRange{spellingAddress, length.getResult()});
}

class BigIntConstantOpLowering
    : public OpConversionPattern<bigint::ConstantOp> {
public:
  using OpConversionPattern<bigint::ConstantOp>::OpConversionPattern;

  auto matchAndRewrite(bigint::ConstantOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!llvm::isa_and_present<LLVM::LLVMPointerType>(resultType))
      return rewriter.notifyMatchFailure(
          op, "BigInt constant result must lower to an opaque pointer");

    auto value = createBigIntLiteral(op.getLoc(), rewriter, op, op.getValue());
    if (failed(value))
      return rewriter.notifyMatchFailure(
          op, "could not create BigInt literal runtime call");
    rewriter.replaceOp(op, *value);
    return success();
  }
};

class BigIntFromUInt64OpLowering
    : public OpConversionPattern<bigint::FromUInt64Op> {
public:
  using OpConversionPattern<bigint::FromUInt64Op>::OpConversionPattern;

  auto matchAndRewrite(bigint::FromUInt64Op op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto value = callRuntimeValue(
        op.getLoc(), rewriter, op, "mulberry_bigint_from_uint64",
        getPtrType(op.getContext()), ValueRange{adaptor.getValue()});
    if (failed(value))
      return rewriter.notifyMatchFailure(
          op, "could not create BigInt widening runtime call");
    rewriter.replaceOp(op, *value);
    return success();
  }
};

class BigIntFromInt64OpLowering
    : public OpConversionPattern<bigint::FromInt64Op> {
public:
  using OpConversionPattern<bigint::FromInt64Op>::OpConversionPattern;

  auto matchAndRewrite(bigint::FromInt64Op op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto value = callRuntimeValue(
        op.getLoc(), rewriter, op, "mulberry_bigint_from_int64",
        getPtrType(op.getContext()), ValueRange{adaptor.getValue()});
    if (failed(value))
      return rewriter.notifyMatchFailure(
          op, "could not create signed BigInt widening runtime call");
    rewriter.replaceOp(op, *value);
    return success();
  }
};

class BigIntNegOpLowering : public OpConversionPattern<bigint::NegOp> {
public:
  using OpConversionPattern<bigint::NegOp>::OpConversionPattern;

  auto matchAndRewrite(bigint::NegOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto value = callRuntimeValue(op.getLoc(), rewriter, op,
                                  "mulberry_bigint_neg",
                                  getPtrType(op.getContext()),
                                  ValueRange{adaptor.getInput()});
    if (failed(value))
      return rewriter.notifyMatchFailure(
          op, "could not create BigInt negation runtime call");
    rewriter.replaceOp(op, *value);
    return success();
  }
};

class BigIntAddOpLowering : public OpConversionPattern<bigint::AddOp> {
public:
  using OpConversionPattern<bigint::AddOp>::OpConversionPattern;

  auto matchAndRewrite(bigint::AddOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto value = callRuntimeValue(op.getLoc(), rewriter, op,
                                  "mulberry_bigint_add",
                                  getPtrType(op.getContext()),
                                  ValueRange{adaptor.getLhs(), adaptor.getRhs()});
    if (failed(value))
      return rewriter.notifyMatchFailure(
          op, "could not create BigInt add runtime call");
    rewriter.replaceOp(op, *value);
    return success();
  }
};

class BigIntSubOpLowering : public OpConversionPattern<bigint::SubOp> {
public:
  using OpConversionPattern<bigint::SubOp>::OpConversionPattern;

  auto matchAndRewrite(bigint::SubOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto value = callRuntimeValue(op.getLoc(), rewriter, op,
                                  "mulberry_bigint_sub",
                                  getPtrType(op.getContext()),
                                  ValueRange{adaptor.getLhs(), adaptor.getRhs()});
    if (failed(value))
      return rewriter.notifyMatchFailure(
          op, "could not create BigInt subtract runtime call");
    rewriter.replaceOp(op, *value);
    return success();
  }
};

class BigIntMulOpLowering : public OpConversionPattern<bigint::MulOp> {
public:
  using OpConversionPattern<bigint::MulOp>::OpConversionPattern;

  auto matchAndRewrite(bigint::MulOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto value = callRuntimeValue(op.getLoc(), rewriter, op,
                                  "mulberry_bigint_mul",
                                  getPtrType(op.getContext()),
                                  ValueRange{adaptor.getLhs(), adaptor.getRhs()});
    if (failed(value))
      return rewriter.notifyMatchFailure(
          op, "could not create BigInt multiply runtime call");
    rewriter.replaceOp(op, *value);
    return success();
  }
};

class BigIntAndOpLowering : public OpConversionPattern<bigint::AndOp> {
public:
  using OpConversionPattern<bigint::AndOp>::OpConversionPattern;

  auto matchAndRewrite(bigint::AndOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto value = callRuntimeValue(
        op.getLoc(), rewriter, op, "mulberry_bigint_bit_and",
        getPtrType(op.getContext()),
        ValueRange{adaptor.getLhs(), adaptor.getRhs()});
    if (failed(value))
      return rewriter.notifyMatchFailure(
          op, "could not create BigInt bitwise-and runtime call");
    rewriter.replaceOp(op, *value);
    return success();
  }
};

class BigIntOrOpLowering : public OpConversionPattern<bigint::OrOp> {
public:
  using OpConversionPattern<bigint::OrOp>::OpConversionPattern;

  auto matchAndRewrite(bigint::OrOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto value = callRuntimeValue(
        op.getLoc(), rewriter, op, "mulberry_bigint_bit_or",
        getPtrType(op.getContext()),
        ValueRange{adaptor.getLhs(), adaptor.getRhs()});
    if (failed(value))
      return rewriter.notifyMatchFailure(
          op, "could not create BigInt bitwise-or runtime call");
    rewriter.replaceOp(op, *value);
    return success();
  }
};

class BigIntXorOpLowering : public OpConversionPattern<bigint::XorOp> {
public:
  using OpConversionPattern<bigint::XorOp>::OpConversionPattern;

  auto matchAndRewrite(bigint::XorOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto value = callRuntimeValue(
        op.getLoc(), rewriter, op, "mulberry_bigint_bit_xor",
        getPtrType(op.getContext()),
        ValueRange{adaptor.getLhs(), adaptor.getRhs()});
    if (failed(value))
      return rewriter.notifyMatchFailure(
          op, "could not create BigInt bitwise-xor runtime call");
    rewriter.replaceOp(op, *value);
    return success();
  }
};

class BigIntShiftLeftOpLowering
    : public OpConversionPattern<bigint::ShiftLeftOp> {
public:
  using OpConversionPattern<bigint::ShiftLeftOp>::OpConversionPattern;

  auto matchAndRewrite(bigint::ShiftLeftOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto value = callRuntimeValue(
        op.getLoc(), rewriter, op, "mulberry_bigint_shift_left",
        getPtrType(op.getContext()),
        ValueRange{adaptor.getValue(), adaptor.getCount()});
    if (failed(value))
      return rewriter.notifyMatchFailure(
          op, "could not create BigInt left-shift runtime call");
    rewriter.replaceOp(op, *value);
    return success();
  }
};

class BigIntShiftRightOpLowering
    : public OpConversionPattern<bigint::ShiftRightOp> {
public:
  using OpConversionPattern<bigint::ShiftRightOp>::OpConversionPattern;

  auto matchAndRewrite(bigint::ShiftRightOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto value = callRuntimeValue(
        op.getLoc(), rewriter, op, "mulberry_bigint_shift_right",
        getPtrType(op.getContext()),
        ValueRange{adaptor.getValue(), adaptor.getCount()});
    if (failed(value))
      return rewriter.notifyMatchFailure(
          op, "could not create BigInt right-shift runtime call");
    rewriter.replaceOp(op, *value);
    return success();
  }
};

class BigIntCmpOpLowering : public OpConversionPattern<bigint::CmpOp> {
public:
  using OpConversionPattern<bigint::CmpOp>::OpConversionPattern;

  auto matchAndRewrite(bigint::CmpOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto comparison = callRuntimeValue(
        op.getLoc(), rewriter, op, "mulberry_bigint_compare",
        getI32Type(op.getContext()),
        ValueRange{adaptor.getLhs(), adaptor.getRhs()});
    if (failed(comparison))
      return rewriter.notifyMatchFailure(
          op, "could not create BigInt comparison runtime call");

    auto zero = arith::ConstantIntOp::create(rewriter, op.getLoc(), 0, 32);
    auto predicate = op.getPredicate();
    arith::CmpIPredicate comparisonPredicate;
    if (predicate == "eq")
      comparisonPredicate = arith::CmpIPredicate::eq;
    else if (predicate == "ne")
      comparisonPredicate = arith::CmpIPredicate::ne;
    else if (predicate == "lt")
      comparisonPredicate = arith::CmpIPredicate::slt;
    else if (predicate == "le")
      comparisonPredicate = arith::CmpIPredicate::sle;
    else if (predicate == "gt")
      comparisonPredicate = arith::CmpIPredicate::sgt;
    else if (predicate == "ge")
      comparisonPredicate = arith::CmpIPredicate::sge;
    else
      return rewriter.notifyMatchFailure(op, "unknown BigInt predicate");

    auto result = arith::CmpIOp::create(rewriter, op.getLoc(),
                                         comparisonPredicate, *comparison,
                                         zero.getResult());
    rewriter.replaceOp(op, result.getResult());
    return success();
  }
};

} // namespace

void populateMulberryBigIntLoweringPatterns(RewritePatternSet& patterns,
                                            TypeConverter& typeConverter,
                                            MLIRContext* context) {
  patterns.add<BigIntAddOpLowering, BigIntAndOpLowering, BigIntCmpOpLowering,
               BigIntConstantOpLowering, BigIntFromInt64OpLowering,
               BigIntFromUInt64OpLowering,
               BigIntMulOpLowering, BigIntNegOpLowering, BigIntOrOpLowering,
               BigIntShiftLeftOpLowering, BigIntShiftRightOpLowering,
               BigIntSubOpLowering, BigIntXorOpLowering>(typeConverter,
                                                          context);
}

} // namespace mlir::mulberry_core::detail
