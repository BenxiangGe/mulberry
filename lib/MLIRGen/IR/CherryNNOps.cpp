#include "cherry/MLIRGen/IR/CherryNNOps.h"
#include "cherry/MLIRGen/IR/CherryNNDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

using namespace mlir;
using namespace mlir::cherry_nn;

#define GET_OP_CLASSES
#include "cherry/MLIRGen/IR/CherryNNOps.cpp.inc"

auto CastOp::verify() -> LogicalResult {
  auto inputType = getInput().getType();
  auto resultType = getResult().getType();
  auto inputIntType = dyn_cast<IntegerType>(inputType);
  auto inputCIRIntType = dyn_cast<cir::IntType>(inputType);
  auto resultIntType = dyn_cast<IntegerType>(resultType);
  auto resultCIRIntType = dyn_cast<cir::IntType>(resultType);

  if (inputIntType && inputIntType.getWidth() == 64 && resultCIRIntType &&
      resultCIRIntType.getWidth() == 64 && resultCIRIntType.isUnsigned())
    return success();
  if (inputCIRIntType && inputCIRIntType.getWidth() == 64 &&
      inputCIRIntType.isUnsigned() && resultIntType &&
      resultIntType.getWidth() == 64)
    return success();

  return emitOpError("only supports i64 <-> !cir.int<u,64> for now");
}
