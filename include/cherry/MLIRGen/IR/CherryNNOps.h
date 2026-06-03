#ifndef CHERRY_NN_OOPS_H
#define CHERRY_NN_OOPS_H

#include "cherry/MLIRGen/IR/CherryNNDialect.h"
#include "cherry/MLIRGen/IR/MulberryTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "cherry/MLIRGen/IR/CherryNNOps.h.inc"

#endif // CHERRY_NN_OOPS_H
