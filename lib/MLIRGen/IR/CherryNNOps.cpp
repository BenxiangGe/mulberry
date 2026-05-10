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
  auto resultIntType = dyn_cast<cir::IntType>(resultType);

  if (inputIntType && inputIntType.getWidth() == 64 && resultIntType &&
      resultIntType.getWidth() == 64 && resultIntType.isUnsigned())
    return success();

  return emitOpError("only supports i64 to !cir.int<u,64> for now");
}
