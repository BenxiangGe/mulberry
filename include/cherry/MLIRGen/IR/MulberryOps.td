//===--- MulberryOps.td - Mulberry dialect ops ------------*- tablegen -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_OPS
#define MULBERRY_OPS

include "cherry/MLIRGen/IR/MulberryDialect.td"
include "cherry/MLIRGen/IR/MulberryTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ListCreateOp : Mulberry_Op<"list.create", [Pure]> {
  let summary = "Create a Mulberry list value";

  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs ListType:$result);

  let assemblyFormat = "$elements attr-dict `:` functional-type($elements, $result)";
  let hasVerifier = 1;
}

def ListGetOp : Mulberry_Op<"list.get", [Pure]> {
  let summary = "Read an element from a Mulberry list";

  let arguments = (ins ListType:$list, Index:$index);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$list `[` $index `]` attr-dict `:` type($list) `->` type($result)";
  let hasVerifier = 1;
}

def ListSizeOp : Mulberry_Op<"list.size", [Pure]> {
  let summary = "Read the length of a Mulberry list";

  let arguments = (ins ListType:$list);
  let results = (outs I64:$result);

  let assemblyFormat = "$list attr-dict `:` type($list) `->` type($result)";
}

#endif // MULBERRY_OPS
