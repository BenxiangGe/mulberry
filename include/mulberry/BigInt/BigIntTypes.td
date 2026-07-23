//===--- BigIntTypes.td - BigInt semantic types -----------*- tablegen -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_BIGINT_TYPES
#define MULBERRY_BIGINT_TYPES

include "mulberry/BigInt/BigIntDialect.td"
include "mlir/IR/AttrTypeBase.td"

def BigInt_IntType : TypeDef<BigInt_Dialect, "Int"> {
  let mnemonic = "int";
  let summary = "Semantic arbitrary-precision integer type";
  let description = [{
    A source-level arbitrary-precision signed integer. Runtime representation,
    allocation, and limb width are intentionally outside this semantic type.
  }];
}

#endif // MULBERRY_BIGINT_TYPES
