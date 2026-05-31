//===--- MulberryOps.cpp - Mulberry dialect ops --------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/IR/MulberryOps.h"
#include "cherry/MLIRGen/IR/MulberryDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::mulberry;

#define GET_OP_CLASSES
#include "cherry/MLIRGen/IR/MulberryOps.cpp.inc"

auto TensorPackOp::verify() -> LogicalResult {
  auto resultType = llvm::cast<TensorDescriptorType>(getResult().getType());
  if (getTensor().getType() != resultType.getMemrefType())
    return emitOpError("result descriptor type does not match tensor type");

  return success();
}

auto TensorUnpackOp::verify() -> LogicalResult {
  auto tensorType = llvm::cast<TensorDescriptorType>(getTensor().getType());
  if (getResult().getType() != tensorType.getMemrefType())
    return emitOpError("result memref type does not match descriptor type");

  return success();
}

auto ListCreateOp::verify() -> LogicalResult {
  auto resultType = llvm::cast<ListType>(getResult().getType());
  auto elementType = resultType.getElementType();

  for (auto element : getElements()) {
    if (element.getType() != elementType)
      return emitOpError("element type does not match list element type");
  }

  return success();
}

auto ListGetOp::verify() -> LogicalResult {
  auto listType = llvm::cast<ListType>(getList().getType());
  if (getResult().getType() != listType.getElementType())
    return emitOpError("result type does not match list element type");

  return success();
}
