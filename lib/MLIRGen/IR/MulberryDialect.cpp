//===--- MulberryDialect.cpp - Mulberry dialect --------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/IR/MulberryDialect.h"
#include "cherry/MLIRGen/IR/MulberryOps.h"
#include "cherry/MLIRGen/IR/MulberryTypes.h"

using namespace mlir;
using namespace mlir::mulberry;

#include "cherry/MLIRGen/IR/MulberryOpsDialect.cpp.inc"

void MulberryDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "cherry/MLIRGen/IR/MulberryOps.cpp.inc"
      >();
  registerTypes();
}
