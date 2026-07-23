//===--- BigIntOps.h - BigInt semantic operations ---------------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_BIGINT_OPS_H
#define MULBERRY_BIGINT_OPS_H

#include "mulberry/BigInt/BigIntDialect.h"
#include "mulberry/BigInt/BigIntTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "mulberry/BigInt/BigIntOps.h.inc"

#endif // MULBERRY_BIGINT_OPS_H
