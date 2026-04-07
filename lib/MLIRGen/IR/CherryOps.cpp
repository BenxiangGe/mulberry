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

using namespace mlir;
using namespace mlir::cherry;

auto StructWriteOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                          mlir::Value structValue, ArrayRef<int64_t> indexes,
                          mlir::Value valueToStore) -> void {
  auto attrs = builder.getDenseI64ArrayAttr(indexes);
  build(builder, state, structValue.getType(), structValue, attrs,
        valueToStore);
}

#define GET_OP_CLASSES
#include "cherry/MLIRGen/IR/CherryOps.cpp.inc"
