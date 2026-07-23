//===--- BigIntDialect.td - BigInt semantic dialect --------*- tablegen -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_BIGINT_DIALECT
#define MULBERRY_BIGINT_DIALECT

include "mlir/IR/OpBase.td"

def BigInt_Dialect : Dialect {
  let name = "bigint";
  let summary = "Mulberry arbitrary-precision integer semantic dialect.";
  let cppNamespace = "::mlir::bigint";

  let useDefaultTypePrinterParser = 1;
}

class BigInt_Op<string mnemonic, list<Trait> traits = []> :
        Op<BigInt_Dialect, mnemonic, traits>;

#endif // MULBERRY_BIGINT_DIALECT
