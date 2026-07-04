//===--- MulberryDialect.td - Mulberry core dialect -------------*- tablegen -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_DIALECT
#define MULBERRY_DIALECT

include "mlir/IR/OpBase.td"

def Mulberry_Dialect : Dialect {
  let name = "mulberry_core";
  let summary = "Mulberry core MLIR dialect.";
  let cppNamespace = "::mlir::mulberry_core";

  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

class Mulberry_Op<string mnemonic, list<Trait> traits = []> :
        Op<Mulberry_Dialect, mnemonic, traits>;

#endif // MULBERRY_DIALECT
