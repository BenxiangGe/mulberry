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

auto BridgeCastOp::verify() -> LogicalResult {
  auto inputType = getInput().getType();
  auto resultType = getResult().getType();
  auto inputIntType = dyn_cast<IntegerType>(inputType);
  auto inputCIRIntType = dyn_cast<cir::IntType>(inputType);
  auto resultIntType = dyn_cast<IntegerType>(resultType);
  auto resultCIRIntType = dyn_cast<cir::IntType>(resultType);
  auto inputFloatType = dyn_cast<Float32Type>(inputType);
  auto inputCIRFloatType = dyn_cast<cir::SingleType>(inputType);
  auto inputCIRBoolType = dyn_cast<cir::BoolType>(inputType);
  auto resultFloatType = dyn_cast<Float32Type>(resultType);
  auto resultCIRFloatType = dyn_cast<cir::SingleType>(resultType);
  auto resultCIRBoolType = dyn_cast<cir::BoolType>(resultType);

  if (inputIntType && inputIntType.getWidth() == 64 && resultCIRIntType &&
      resultCIRIntType.getWidth() == 64 && resultCIRIntType.isUnsigned())
    return success();
  if (inputCIRIntType && inputCIRIntType.getWidth() == 64 &&
      inputCIRIntType.isUnsigned() && resultIntType &&
      resultIntType.getWidth() == 64)
    return success();
  if (inputIntType && inputIntType.getWidth() == 1 && resultCIRBoolType)
    return success();
  if (inputCIRBoolType && resultIntType && resultIntType.getWidth() == 1)
    return success();
  if (inputFloatType && resultCIRFloatType)
    return success();
  if (inputCIRFloatType && resultFloatType)
    return success();

  return emitOpError("only supports MLIR/CIR scalar bridge casts");
}
