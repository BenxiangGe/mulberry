//===--- CherryOps.h - Cherry dialect ops -----------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_OPS
#define CHERRY_OPS

include "cherry/MLIRGen/IR/CherryTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

def Cherry_Type : AnyTypeOf<[I64, I1, Cherry_Struct]>;

def PrintOp : Cherry_Op<"print"> {
  let summary = "Print operation";
  let description = [{
      The "print" builtin operation prints a given input
  }];

  let arguments = (ins AnyType:$input);
  let results = (outs I64:$result);

  let assemblyFormat = "$input attr-dict `:` functional-type($input, $result)";
}

def CastOp : Cherry_Op<"cast"> {
  let summary = "Cast operation";
  let description = [{
    Casts from I64 to I1
  }];

  let arguments = (ins AnyType:$input);
  let results = (outs I64:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($result)";
}

#endif // CHERRY_OPS
