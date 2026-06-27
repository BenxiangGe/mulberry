//===--- MulberryDialect.cpp - Mulberry dialect ---------------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "mulberry/MLIRGen/IR/MulberryDialect.h"
#include "mulberry/MLIRGen/IR/MulberryOps.h"
#include "mulberry/MLIRGen/IR/MulberryTypes.h"

using namespace mlir;
using namespace mlir::mulberry;

#include "mulberry/MLIRGen/IR/MulberryOpsDialect.cpp.inc"

void MulberryDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mulberry/MLIRGen/IR/MulberryOps.cpp.inc"
      >();
  registerTypes();
}
