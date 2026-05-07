#include "cherry/MLIRGen/IR/CherryNNDialect.h"
#include "cherry/MLIRGen/IR/CherryNNOps.h"

using namespace mlir;
using namespace mlir::cherry_nn;

#include "cherry/MLIRGen/IR/CherryNNOpsDialect.cpp.inc"

void CherryNNDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "cherry/MLIRGen/IR/CherryNNOps.cpp.inc"
      >();
}