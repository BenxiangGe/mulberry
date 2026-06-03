// RUN: cherry-opt %s | FileCheck %s

module {
  func.func @tensor_ops(%i: index, %j: index) -> f32 {
    %tensor = mulberry.tensor.alloc : !mulberry.tensor<2x3xf32>
    %value = arith.constant 1.000000e+00 : f32
    mulberry.tensor.store %value, %tensor[%i, %j] : f32, !mulberry.tensor<2x3xf32>
    %loaded = mulberry.tensor.load %tensor[%i, %j] : !mulberry.tensor<2x3xf32> -> f32
    return %loaded : f32
  }
}

// CHECK-LABEL: func.func @tensor_ops
// CHECK: mulberry.tensor.alloc : !mulberry.tensor<2x3xf32>
// CHECK: mulberry.tensor.store
// CHECK: mulberry.tensor.load
