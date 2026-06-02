//===--- CherryOps.cpp - Cherry dialect ops -------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/IR/CherryOps.h"
#include "cherry/MLIRGen/IR/CherryDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

using namespace mlir;
using namespace mlir::cherry;

#define GET_OP_CLASSES
#include "cherry/MLIRGen/IR/CherryOps.cpp.inc"

namespace {

auto isI1(Type type) -> bool {
  auto integerType = dyn_cast<IntegerType>(type);
  return integerType && integerType.getWidth() == 1;
}

auto isI64(Type type) -> bool {
  auto integerType = dyn_cast<IntegerType>(type);
  return integerType && integerType.getWidth() == 64;
}

auto isCIRBool(Type type) -> bool {
  return isa<cir::BoolType>(type);
}

auto isCIRUInt64(Type type) -> bool {
  auto intType = dyn_cast<cir::IntType>(type);
  return intType && intType.getWidth() == 64 && intType.isUnsigned();
}

} // end namespace

auto CastOp::verify() -> LogicalResult {
  auto inputType = getInput().getType();
  auto resultType = getResult().getType();

  if ((isI1(inputType) && isI64(resultType)) ||
      (isCIRBool(inputType) && isCIRUInt64(resultType)))
    return success();

  if ((isI64(inputType) && isCIRUInt64(resultType)) ||
      (isCIRUInt64(inputType) && isI64(resultType)))
    return success();

  return emitOpError("only supports bool -> UInt64 and i64 <-> "
                     "!cir.int<u,64> casts for now");
}
