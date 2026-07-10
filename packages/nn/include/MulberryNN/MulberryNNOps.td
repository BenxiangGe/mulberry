#ifndef MULBERRY_NN_OPS
#define MULBERRY_NN_OPS

include "MulberryNN/MulberryNNDialect.td"
include "mulberry/MLIRGen/IR/MulberryTypes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def MulberryNN_AnyTensorType : Type<
    CPred<"::llvm::isa<::mlir::mulberry_core::TensorType>($_self)">,
    "Mulberry tensor type">;

def MulberryNN_AnyTensorStorageType : Type<
    Or<[CPred<"::llvm::isa<::mlir::mulberry_core::TensorType>($_self)">,
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

def ReluOp : MulberryNN_Op<"relu"> {
  let summary = "elementwise rectified linear unit operation";
  let description = [{
      Applies max(0, x) elementwise while preserving the input shape.
  }];

  let arguments = (ins MulberryNN_AnyTensorStorageType:$input,
                       MulberryNN_AnyTensorStorageType:$out);

  let assemblyFormat = "`ins` `(` $input `:` type($input) `)` "
                       "`outs` `(` $out `:` type($out) `)` attr-dict";

  let hasVerifier = 1;
}

def SoftmaxOp : MulberryNN_Op<"softmax"> {
  let summary = "row-wise softmax operation";
  let description = [{
      Applies softmax over the final dimension of a rank-2 tensor.
  }];

  let arguments = (ins MulberryNN_AnyTensorStorageType:$input,
                       MulberryNN_AnyTensorStorageType:$out);

  let assemblyFormat = "`ins` `(` $input `:` type($input) `)` "
                       "`outs` `(` $out `:` type($out) `)` attr-dict";

  let hasVerifier = 1;
}

def SoftmaxCrossEntropyOp : MulberryNN_Op<"softmax_cross_entropy"> {
  let summary = "mean row-wise softmax cross-entropy operation";
  let description = [{
      Computes the mean categorical cross-entropy from rank-2 logits and
      expected values. The operation uses a stable log-sum-exp formulation and
      does not materialize the softmax probabilities.
  }];

  let arguments = (ins MulberryNN_AnyTensorStorageType:$logits,
                       MulberryNN_AnyTensorStorageType:$expected);
  let results = (outs F32:$result);

  let assemblyFormat = "`ins` `(` $logits `,` $expected `:` type($logits) `,` "
                       "type($expected) `)` attr-dict `->` type($result)";

  let hasVerifier = 1;
}

def Conv2DOp : MulberryNN_Op<"conv2d"> {
  let summary = "NCHW 2D convolution operation";
  let description = [{
      Applies an OIHW filter and rank-1 output-channel bias to an NCHW input.
      Padding is [top, bottom, left, right]; strides and dilations are [h, w].
  }];

  let arguments = (ins MulberryNN_AnyTensorStorageType:$input,
                       MulberryNN_AnyTensorStorageType:$weight,
                       MulberryNN_AnyTensorStorageType:$bias,
                       MulberryNN_AnyTensorStorageType:$out,
                       DenseI64ArrayAttr:$padding,
                       DenseI64ArrayAttr:$strides,
                       DenseI64ArrayAttr:$dilations);

  let assemblyFormat = "`ins` `(` $input `,` $weight `,` $bias `:` "
                       "type($input) `,` type($weight) `,` type($bias) `)` "
                       "`outs` `(` $out `:` type($out) `)` attr-dict";

  let hasVerifier = 1;
}

def MaxPool2DOp : MulberryNN_Op<"max_pool2d"> {
  let summary = "NCHW 2D max pooling operation";
  let description = [{
      Applies a 2D max-pooling window to an NCHW input. Kernel and strides are
      [h, w]; padding is [top, bottom, left, right].
  }];

  let arguments = (ins MulberryNN_AnyTensorStorageType:$input,
                       MulberryNN_AnyTensorStorageType:$out,
                       DenseI64ArrayAttr:$kernel,
                       DenseI64ArrayAttr:$padding,
                       DenseI64ArrayAttr:$strides);

  let assemblyFormat = "`ins` `(` $input `:` type($input) `)` "
                       "`outs` `(` $out `:` type($out) `)` attr-dict";

  let hasVerifier = 1;
}

def ReluBackwardOp : MulberryNN_Op<"relu_backward"> {
  let summary = "rectified linear unit backward operation";
  let description = [{
      Computes the input gradient for ReLU. The derivative is one where the
      forward input is greater than zero and zero otherwise, including at zero.
      The input-gradient destination is fully initialized by the operation.
  }];

  let arguments = (ins MulberryNN_AnyTensorStorageType:$input,
                       MulberryNN_AnyTensorStorageType:$outputGradient,
                       MulberryNN_AnyTensorStorageType:$inputGradient);

  let assemblyFormat = "`ins` `(` $input `,` $outputGradient `:` "
                       "type($input) `,` type($outputGradient) `)` "
                       "`outs` `(` $inputGradient `:` type($inputGradient) `)` "
                       "attr-dict";

  let hasVerifier = 1;
}

def Conv2DBackwardOp : MulberryNN_Op<"conv2d_backward"> {
  let summary = "NCHW 2D convolution backward operation";
  let description = [{
      Computes input, OIHW weight, and output-channel bias gradients from an
      NCHW input, OIHW weight, and NCHW output gradient. Padding is
      [top, bottom, left, right]; strides and dilations are [h, w]. All three
      gradient destinations are fully initialized by the operation.
  }];

  let arguments = (ins MulberryNN_AnyTensorStorageType:$input,
                       MulberryNN_AnyTensorStorageType:$weight,
                       MulberryNN_AnyTensorStorageType:$outputGradient,
                       MulberryNN_AnyTensorStorageType:$inputGradient,
                       MulberryNN_AnyTensorStorageType:$weightGradient,
                       MulberryNN_AnyTensorStorageType:$biasGradient,
                       DenseI64ArrayAttr:$padding,
                       DenseI64ArrayAttr:$strides,
                       DenseI64ArrayAttr:$dilations);

  let assemblyFormat = "`ins` `(` $input `,` $weight `,` $outputGradient `:` "
                       "type($input) `,` type($weight) `,` "
                       "type($outputGradient) `)` `outs` `(` $inputGradient `,` "
                       "$weightGradient `,` $biasGradient `:` "
                       "type($inputGradient) `,` type($weightGradient) `,` "
                       "type($biasGradient) `)` attr-dict";

  let hasVerifier = 1;
}

def MaxPool2DBackwardOp : MulberryNN_Op<"max_pool2d_backward"> {
  let summary = "NCHW 2D max-pooling backward operation";
  let description = [{
      Recomputes each pooling window's winner from the forward input and routes
      the output gradient to the first maximum in row-major window order.
      Padding positions never receive gradients; overlapping windows
      accumulate into the input gradient. Kernel and strides are [h, w], and
      padding is [top, bottom, left, right]. The input-gradient destination is
      fully initialized by the operation.
  }];

  let arguments = (ins MulberryNN_AnyTensorStorageType:$input,
                       MulberryNN_AnyTensorStorageType:$outputGradient,
                       MulberryNN_AnyTensorStorageType:$inputGradient,
                       DenseI64ArrayAttr:$kernel,
                       DenseI64ArrayAttr:$padding,
                       DenseI64ArrayAttr:$strides);

  let assemblyFormat = "`ins` `(` $input `,` $outputGradient `:` "
                       "type($input) `,` type($outputGradient) `)` "
                       "`outs` `(` $inputGradient `:` type($inputGradient) `)` "
                       "attr-dict";

  let hasVerifier = 1;
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
