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

def MataddOp : CherryNN_Op<"matadd"> {
  let summary = "matrix add operation";
  let description = [{
      The "matadd" operation does elementwise matrix addition
  }];

  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$out);

  let assemblyFormat = "`ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)` "
                       "`outs` `(` $out `:` type($out) `)` attr-dict";

  let hasVerifier = 0;
}

def TransposeOp : CherryNN_Op<"transpose"> {
  let summary = "matrix transpose operation";
  let description = [{
      The "transpose" operation transposes a matrix
  }];

  let arguments = (ins AnyMemRef:$input, AnyMemRef:$out);

  let assemblyFormat = "`ins` `(` $input `:` type($input) `)` "
                       "`outs` `(` $out `:` type($out) `)` attr-dict";

  let hasVerifier = 0;
}

def ExpOp : CherryNN_Op<"exp"> {
  let summary = "elementwise exponential operation";
  let description = [{
      The "exp" operation applies exponential elementwise to a matrix
  }];

  let arguments = (ins AnyMemRef:$input, AnyMemRef:$out);

  let assemblyFormat = "`ins` `(` $input `:` type($input) `)` "
                       "`outs` `(` $out `:` type($out) `)` attr-dict";

  let hasVerifier = 0;
}

def SigmoidOp : CherryNN_Op<"sigmoid"> {
  let summary = "elementwise sigmoid operation";
  let description = [{
      The "sigmoid" operation applies sigmoid elementwise to a matrix
  }];

  let arguments = (ins AnyMemRef:$input, AnyMemRef:$out);

  let assemblyFormat = "`ins` `(` $input `:` type($input) `)` "
                       "`outs` `(` $out `:` type($out) `)` attr-dict";

  let hasVerifier = 0;
}

def ArgmaxOp : CherryNN_Op<"argmax"> {
  let summary = "argmax operation";
  let description = [{
    The "argmax" operation returns the flat index of the maximum element.
  }];

  let arguments = (ins AnyMemRef:$input);
  let results = (outs I64:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";

  let hasVerifier = 0;
}

#endif // CHERRY_NN_OPS
