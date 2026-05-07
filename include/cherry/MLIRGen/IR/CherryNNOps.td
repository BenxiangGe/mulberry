#ifndef CHERRY_NN_OPS
#define CHERRY_NN_OPS

include "cherry/MLIRGen/IR/CherryNNDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def MatmulOp : CherryNN_Op<"matmul"> {
  let summary = "matmul operation";
  let description = [{
      The "matmul" operation does matrix multiplication
  }];

  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$out);

  let assemblyFormat = "`ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)` "
                       "`outs` `(` $out `:` type($out) `)` attr-dict";

  let hasVerifier = 0;
}


#endif // CHERRY_NN_OPS