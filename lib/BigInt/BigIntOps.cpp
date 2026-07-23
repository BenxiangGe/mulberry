//===--- BigIntOps.cpp - BigInt semantic operations -----------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "mulberry/BigInt/BigIntOps.h"

#include "mulberry/Basic/IntegerLiteral.h"

#include <string_view>

using namespace mlir;
using namespace mlir::bigint;

#define GET_OP_CLASSES
#include "mulberry/BigInt/BigIntOps.cpp.inc"

LogicalResult ConstantOp::verify() {
  auto spelling = getValue();
  if (!mulberry::isValidIntegerLiteralSpelling(
          std::string_view(spelling.data(), spelling.size())))
    return emitOpError("value must be a valid integer literal spelling");

  return success();
}

LogicalResult CmpOp::verify() {
  StringRef predicate = getPredicate();
  if (predicate != "eq" && predicate != "ne" && predicate != "lt" &&
      predicate != "le" && predicate != "gt" && predicate != "ge")
    return emitOpError("predicate must be one of eq, ne, lt, le, gt, or ge");

  return success();
}
