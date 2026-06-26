#include "MulberryNN/MulberryNNOps.h"
#include "MulberryNN/MulberryNNDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::mulberry_nn;

#define GET_OP_CLASSES
#include "MulberryNN/MulberryNNOps.cpp.inc"

