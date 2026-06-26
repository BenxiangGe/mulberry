#include "MulberryNN/MulberryNNDialect.h"
#include "MulberryNN/MulberryNNOps.h"

using namespace mlir;
using namespace mlir::mulberry_nn;

#include "MulberryNN/MulberryNNOpsDialect.cpp.inc"

void MulberryNNDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "MulberryNN/MulberryNNOps.cpp.inc"
      >();
}

