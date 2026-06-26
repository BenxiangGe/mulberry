#ifndef MULBERRY_NN_DIALECT
#define MULBERRY_NN_DIALECT

include "mlir/IR/OpBase.td"

def MulberryNN_Dialect : Dialect {
  let name = "mulberry_nn";
  let summary = "Mulberry neural network package dialect.";
  let cppNamespace = "::mlir::mulberry_nn";
}

class MulberryNN_Op<string mnemonic, list<Trait> traits = []> :
        Op<MulberryNN_Dialect, mnemonic, traits>;

#endif // MULBERRY_NN_DIALECT

