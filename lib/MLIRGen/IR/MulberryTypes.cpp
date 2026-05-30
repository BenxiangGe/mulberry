//===--- MulberryTypes.cpp - Mulberry dialect types ----------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/IR/MulberryTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::mulberry;

#define GET_TYPEDEF_CLASSES
#include "cherry/MLIRGen/IR/MulberryOpsTypes.cpp.inc"

void MulberryDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "cherry/MLIRGen/IR/MulberryOpsTypes.cpp.inc"
      >();
}
