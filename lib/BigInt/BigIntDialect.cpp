//===--- BigIntDialect.cpp - BigInt semantic dialect ---------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "mulberry/BigInt/BigIntDialect.h"
#include "mulberry/BigInt/BigIntOps.h"
#include "mulberry/BigInt/BigIntTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::bigint;

#define GET_TYPEDEF_CLASSES
#include "mulberry/BigInt/BigIntOpsTypes.cpp.inc"

#include "mulberry/BigInt/BigIntOpsDialect.cpp.inc"

void BigIntDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mulberry/BigInt/BigIntOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mulberry/BigInt/BigIntOpsTypes.cpp.inc"
      >();
}
