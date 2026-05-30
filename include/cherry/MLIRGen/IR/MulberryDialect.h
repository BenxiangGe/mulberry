//===--- MulberryDialect.h - Mulberry dialect -----------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_MULBERRYDIALECT_H
#define CHERRY_MULBERRYDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"

#include "cherry/MLIRGen/IR/MulberryOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "cherry/MLIRGen/IR/MulberryOpsTypes.h.inc"

#endif // CHERRY_MULBERRYDIALECT_H
