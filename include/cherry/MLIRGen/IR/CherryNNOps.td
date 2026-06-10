#ifndef CHERRY_NN_OPS
#define CHERRY_NN_OPS

include "cherry/MLIRGen/IR/CherryNNDialect.td"
include "cherry/MLIRGen/IR/MulberryTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def CherryNN_AnyTensorType : Type<
    CPred<"::llvm::isa<::mlir::mulberry::TensorType>($_self)">,
    "Mulberry tensor type">;

def MatmulOp : CherryNN_Op<"matmul"> {
  let summary = "matmul operation";
  let description = [{
      The "matmul" operation does matrix multiplication
  }];

  let arguments = (ins CherryNN_AnyTensorType:$lhs,
                       CherryNN_AnyTensorType:$rhs,
                       CherryNN_AnyTensorType:$out);

  let assemblyFormat = "`ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)` "
                       "`outs` `(` $out `:` type($out) `)` attr-dict";

  let hasVerifier = 0;
}

def MataddOp : CherryNN_Op<"matadd"> {
  let summary = "matrix add operation";
  let description = [{
      The "matadd" operation does elementwise matrix addition
  }];

  let arguments = (ins CherryNN_AnyTensorType:$lhs,
                       CherryNN_AnyTensorType:$rhs,
                       CherryNN_AnyTensorType:$out);

  let assemblyFormat = "`ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)` "
                       "`outs` `(` $out `:` type($out) `)` attr-dict";

  let hasVerifier = 0;
}

def MatsubOp : CherryNN_Op<"matsub"> {
  let summary = "matrix subtract operation";
  let description = [{
      The "matsub" operation does elementwise matrix subtraction
  }];

  let arguments = (ins CherryNN_AnyTensorType:$lhs,
                       CherryNN_AnyTensorType:$rhs,
                       CherryNN_AnyTensorType:$out);

  let assemblyFormat = "`ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)` "
                       "`outs` `(` $out `:` type($out) `)` attr-dict";

  let hasVerifier = 0;
}

def HadamardOp : CherryNN_Op<"hadamard"> {
  let summary = "Hadamard product operation";
  let description = [{
      The "hadamard" operation does elementwise matrix multiplication
  }];

  let arguments = (ins CherryNN_AnyTensorType:$lhs,
                       CherryNN_AnyTensorType:$rhs,
                       CherryNN_AnyTensorType:$out);

  let assemblyFormat = "`ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)` "
                       "`outs` `(` $out `:` type($out) `)` attr-dict";

  let hasVerifier = 0;
}

def MatscaleOp : CherryNN_Op<"matscale"> {
  let summary = "matrix scalar multiplication operation";
  let description = [{
      The "matscale" operation multiplies each matrix element by a Float32 scalar
  }];

  let arguments = (ins CherryNN_AnyTensorType:$input,
                       F32:$scale,
                       CherryNN_AnyTensorType:$out);

  let assemblyFormat = "`ins` `(` $input `,` $scale `:` type($input) `,` "
                       "type($scale) `)` `outs` `(` $out `:` type($out) `)` "
                       "attr-dict";

  let hasVerifier = 0;
}

def TransposeOp : CherryNN_Op<"transpose"> {
  let summary = "matrix transpose operation";
  let description = [{
      The "transpose" operation transposes a matrix
  }];

  let arguments = (ins CherryNN_AnyTensorType:$input,
                       CherryNN_AnyTensorType:$out);

  let assemblyFormat = "`ins` `(` $input `:` type($input) `)` "
                       "`outs` `(` $out `:` type($out) `)` attr-dict";

  let hasVerifier = 0;
}

def ExpOp : CherryNN_Op<"exp"> {
  let summary = "elementwise exponential operation";
  let description = [{
      The "exp" operation applies exponential elementwise to a matrix
  }];

  let arguments = (ins CherryNN_AnyTensorType:$input,
                       CherryNN_AnyTensorType:$out);

  let assemblyFormat = "`ins` `(` $input `:` type($input) `)` "
                       "`outs` `(` $out `:` type($out) `)` attr-dict";

  let hasVerifier = 0;
}

def SigmoidOp : CherryNN_Op<"sigmoid"> {
  let summary = "elementwise sigmoid operation";
  let description = [{
      The "sigmoid" operation applies sigmoid elementwise to a matrix
  }];

  let arguments = (ins CherryNN_AnyTensorType:$input,
                       CherryNN_AnyTensorType:$out);

  let assemblyFormat = "`ins` `(` $input `:` type($input) `)` "
                       "`outs` `(` $out `:` type($out) `)` attr-dict";

  let hasVerifier = 0;
}

def SigmoidPrimeOp : CherryNN_Op<"sigmoid_prime"> {
  let summary = "elementwise sigmoid derivative operation";
  let description = [{
      The "sigmoid_prime" operation applies the sigmoid derivative
      elementwise to a matrix
  }];

  let arguments = (ins CherryNN_AnyTensorType:$input,
                       CherryNN_AnyTensorType:$out);

  let assemblyFormat = "`ins` `(` $input `:` type($input) `)` "
                       "`outs` `(` $out `:` type($out) `)` attr-dict";

  let hasVerifier = 0;
}

def ArgmaxOp : CherryNN_Op<"argmax"> {
  let summary = "argmax operation";
  let description = [{
    The "argmax" operation returns the flat index of the maximum element.
  }];

  let arguments = (ins CherryNN_AnyTensorType:$input);
  let results = (outs I64:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";

  let hasVerifier = 0;
}

#endif // CHERRY_NN_OPS
