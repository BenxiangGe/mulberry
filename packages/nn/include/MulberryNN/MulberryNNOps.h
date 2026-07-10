//===--- MulberryNNOps.h - Mulberry NN ops ---------------------*- C++ -*-===//

#ifndef MULBERRY_NN_OPS_H
#define MULBERRY_NN_OPS_H

#include "MulberryNN/MulberryNNDialect.h"
#include "mulberry/MLIRGen/IR/MulberryTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "MulberryNN/MulberryNNOps.h.inc"

#endif // MULBERRY_NN_OPS_H
