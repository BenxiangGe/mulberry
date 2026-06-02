#include "cherry/MLIRGen/IR/CherryNNOps.h"
#include "cherry/MLIRGen/IR/CherryNNDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::cherry_nn;

#define GET_OP_CLASSES
#include "cherry/MLIRGen/IR/CherryNNOps.cpp.inc"
