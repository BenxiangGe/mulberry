// RUN: not mulberry-opt --load-dialect-plugin=%mulberry_libs/MulberryNNPackage%shlibext --split-input-file %s 2>&1 | FileCheck %s

module {
  func.func @bad_element_type(
      %input: memref<2x2xi64>, %out: memref<2x2xi64>) {
    mulberry_nn.relu ins(%input : memref<2x2xi64>)
        outs(%out : memref<2x2xi64>)
    return
  }
}

// -----

module {
  func.func @bad_rank(
      %input: memref<4xf32>, %out: memref<4xf32>) {
    mulberry_nn.softmax ins(%input : memref<4xf32>)
        outs(%out : memref<4xf32>)
    return
  }
}

// -----

module {
  func.func @bad_shape(
      %input: memref<2x2xf32>, %out: memref<2x3xf32>) {
    mulberry_nn.matadd ins(%input, %input : memref<2x2xf32>, memref<2x2xf32>)
        outs(%out : memref<2x3xf32>)
    return
  }
}

// -----

module {
  func.func @bad_window_attributes(
      %input: memref<1x1x4x4xf32>, %weight: memref<1x1x3x3xf32>,
      %bias: memref<1xf32>, %out: memref<1x1x2x2xf32>) {
    mulberry_nn.conv2d
        ins(%input, %weight, %bias : memref<1x1x4x4xf32>,
            memref<1x1x3x3xf32>, memref<1xf32>)
        outs(%out : memref<1x1x2x2xf32>) {
          dilations = array<i64: 1, 1>,
          padding = array<i64: 0, 0, 0, 0>,
          strides = array<i64: 0, 1>
        }
    return
  }
}

// -----

module {
  func.func @bad_matmul_shape(
      %lhs: memref<2x3xf32>, %rhs: memref<4x2xf32>,
      %out: memref<2x2xf32>) {
    mulberry_nn.matmul ins(%lhs, %rhs : memref<2x3xf32>, memref<4x2xf32>)
        outs(%out : memref<2x2xf32>)
    return
  }
}

// -----

module {
  func.func @bad_transpose_shape(
      %input: memref<2x3xf32>, %out: memref<2x3xf32>) {
    mulberry_nn.transpose ins(%input : memref<2x3xf32>)
        outs(%out : memref<2x3xf32>)
    return
  }
}

// -----

module {
  func.func @bad_argmax_rank(%input: memref<2x2x2xf32>) -> i64 {
    %result = mulberry_nn.argmax %input : memref<2x2x2xf32> -> i64
    return %result : i64
  }
}

// -----

module {
  func.func @bad_representation(
      %lhs: !mulberry_core.tensor<2x2xf32>,
      %rhs: !mulberry_core.tensor<2x2xf32>,
      %out: memref<2x2xf32>) {
    mulberry_nn.matadd
        ins(%lhs, %rhs : !mulberry_core.tensor<2x2xf32>,
            !mulberry_core.tensor<2x2xf32>)
        outs(%out : memref<2x2xf32>)
    return
  }
}

// -----

module {
  func.func @bad_backward_shape(
      %input: memref<2x2xf32>, %outputGradient: memref<2x3xf32>,
      %inputGradient: memref<2x2xf32>) {
    mulberry_nn.relu_backward
        ins(%input, %outputGradient : memref<2x2xf32>, memref<2x3xf32>)
        outs(%inputGradient : memref<2x2xf32>)
    return
  }
}

// CHECK: error: 'mulberry_nn.relu' op input must have Float32 elements
// CHECK: error: 'mulberry_nn.softmax' op input must have rank 2
// CHECK: error: 'mulberry_nn.matadd' op input and output dimensions must match
// CHECK: error: 'mulberry_nn.conv2d' op strides values must be positive
// CHECK: error: 'mulberry_nn.matmul' op lhs and rhs contraction dimensions must match
// CHECK: error: 'mulberry_nn.transpose' op input rows and output columns must match
// CHECK: error: 'mulberry_nn.argmax' op input must have rank 1 or 2
// CHECK: error: 'mulberry_nn.matadd' op tensor operands must use the same core-tensor or memref representation
// CHECK: error: 'mulberry_nn.relu_backward' op input and output-gradient dimensions must match
