#ifndef CHERRY_NN_DIALECT
#define CHERRY_NN_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// CherryNN dialect definition.
//===----------------------------------------------------------------------===//

def CherryNN_Dialect : Dialect {
  let name = "cherry_nn";
  let summary = "Cherry neural network dialect.";
  let cppNamespace = "::mlir::cherry_nn";
}

//===----------------------------------------------------------------------===//
// Cherry neural network operation definition.
//===----------------------------------------------------------------------===//

class CherryNN_Op<string mnemonic, list<Trait> traits = []> :
        Op<CherryNN_Dialect, mnemonic, traits>;

#endif // CHERRY_NN_DIALECT