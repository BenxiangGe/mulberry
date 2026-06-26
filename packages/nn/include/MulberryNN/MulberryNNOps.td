#ifndef MULBERRY_NN_OPS
#define MULBERRY_NN_OPS

include "MulberryNN/MulberryNNDialect.td"
include "cherry/MLIRGen/IR/MulberryTypes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def MulberryNN_AnyTensorType : Type<
    CPred<"::llvm::isa<::mlir::mulberry::TensorType>($_self)">,
    "Mulberry tensor type">;

def MulberryNN_AnyTensorStorageType : Type<
    Or<[CPred<"::llvm::isa<::mlir::mulberry::TensorType>($_self)">,
        IsMemRefTypePred]>,
    "Mulberry tensor or lowered memref tensor storage">;

def MatmulOp : MulberryNN_Op<"matmul"> {
  let summary = "matmul operation";
  let description = [{
      The "matmul" operation does matrix multiplication
  }];

  let arguments = (ins MulberryNN_AnyTensorStorageType:$lhs,
                       MulberryNN_AnyTensorStorageType:$rhs,
                       MulberryNN_AnyTensorStorageType:$out);

  let assemblyFormat = "`ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)` "
                       "`outs` `(` $out `:` type($out) `)` attr-dict";

  let hasVerifier = 0;
}

def MataddOp : MulberryNN_Op<"matadd"> {
  let summary = "matrix add operation";
  let description = [{
      The "matadd" operation does elementwise matrix addition
  }];

  let arguments = (ins MulberryNN_AnyTensorStorageType:$lhs,
                       MulberryNN_AnyTensorStorageType:$rhs,
                       MulberryNN_AnyTensorStorageType:$out);

  let assemblyFormat = "`ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)` "
                       "`outs` `(` $out `:` type($out) `)` attr-dict";

  let hasVerifier = 0;
}

def MatsubOp : MulberryNN_Op<"matsub"> {
  let summary = "matrix subtract operation";
  let description = [{
      The "matsub" operation does elementwise matrix subtraction
  }];

  let arguments = (ins MulberryNN_AnyTensorStorageType:$lhs,
                       MulberryNN_AnyTensorStorageType:$rhs,
                       MulberryNN_AnyTensorStorageType:$out);

  let assemblyFormat = "`ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)` "
                       "`outs` `(` $out `:` type($out) `)` attr-dict";

  let hasVerifier = 0;
}

def HadamardOp : MulberryNN_Op<"hadamard"> {
  let summary = "Hadamard product operation";
  let description = [{
      The "hadamard" operation does elementwise matrix multiplication
  }];

  let arguments = (ins MulberryNN_AnyTensorStorageType:$lhs,
                       MulberryNN_AnyTensorStorageType:$rhs,
                       MulberryNN_AnyTensorStorageType:$out);

  let assemblyFormat = "`ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)` "
                       "`outs` `(` $out `:` type($out) `)` attr-dict";

  let hasVerifier = 0;
}

def MatscaleOp : MulberryNN_Op<"matscale"> {
  let summary = "matrix scalar multiplication operation";
  let description = [{
      The "matscale" operation multiplies each matrix element by a Float32 scalar
  }];

  let arguments = (ins MulberryNN_AnyTensorStorageType:$input,
                       F32:$scale,
                       MulberryNN_AnyTensorStorageType:$out);

  let assemblyFormat = "`ins` `(` $input `,` $scale `:` type($input) `,` "
                       "type($scale) `)` `outs` `(` $out `:` type($out) `)` "
                       "attr-dict";

  let hasVerifier = 0;
}

def TransposeOp : MulberryNN_Op<"transpose"> {
  let summary = "matrix transpose operation";
  let description = [{
      The "transpose" operation transposes a matrix
  }];

  let arguments = (ins MulberryNN_AnyTensorStorageType:$input,
                       MulberryNN_AnyTensorStorageType:$out);

  let assemblyFormat = "`ins` `(` $input `:` type($input) `)` "
                       "`outs` `(` $out `:` type($out) `)` attr-dict";

  let hasVerifier = 0;
}

def ExpOp : MulberryNN_Op<"exp"> {
  let summary = "elementwise exponential operation";
  let description = [{
      The "exp" operation applies exponential elementwise to a matrix
  }];

  let arguments = (ins MulberryNN_AnyTensorStorageType:$input,
                       MulberryNN_AnyTensorStorageType:$out);

  let assemblyFormat = "`ins` `(` $input `:` type($input) `)` "
                       "`outs` `(` $out `:` type($out) `)` attr-dict";

  let hasVerifier = 0;
}

def SigmoidOp : MulberryNN_Op<"sigmoid"> {
  let summary = "elementwise sigmoid operation";
  let description = [{
      The "sigmoid" operation applies sigmoid elementwise to a matrix
  }];

  let arguments = (ins MulberryNN_AnyTensorStorageType:$input,
                       MulberryNN_AnyTensorStorageType:$out);

  let assemblyFormat = "`ins` `(` $input `:` type($input) `)` "
                       "`outs` `(` $out `:` type($out) `)` attr-dict";

  let hasVerifier = 0;
}

def SigmoidPrimeOp : MulberryNN_Op<"sigmoid_prime"> {
  let summary = "elementwise sigmoid derivative operation";
  let description = [{
      The "sigmoid_prime" operation applies the sigmoid derivative
      elementwise to a matrix
  }];

  let arguments = (ins MulberryNN_AnyTensorStorageType:$input,
                       MulberryNN_AnyTensorStorageType:$out);

  let assemblyFormat = "`ins` `(` $input `:` type($input) `)` "
                       "`outs` `(` $out `:` type($out) `)` attr-dict";

  let hasVerifier = 0;
}

def ArgmaxOp : MulberryNN_Op<"argmax"> {
  let summary = "argmax operation";
  let description = [{
    The "argmax" operation returns the flat index of the maximum element.
  }];

  let arguments = (ins MulberryNN_AnyTensorStorageType:$input);
  let results = (outs I64:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";

  let hasVerifier = 0;
}

#endif // MULBERRY_NN_OPS
