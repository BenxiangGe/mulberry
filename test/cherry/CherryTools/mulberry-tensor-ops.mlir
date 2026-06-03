// RUN: cherry-opt %s | FileCheck %s

module {
  func.func @tensor_ops(%i: index, %j: index) -> f32 {
    %tensor = mulberry.tensor.alloc() : !mulberry.tensor<2x3xf32>
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %dynamic = mulberry.tensor.alloc(%c2, %c3) : !mulberry.tensor<?x?xf32>
    %dim = mulberry.tensor.dim %dynamic, %c0 : !mulberry.tensor<?x?xf32>
    %cast = mulberry.tensor.cast %tensor : !mulberry.tensor<2x3xf32> to !mulberry.tensor<?x?xf32>
    %value = arith.constant 1.000000e+00 : f32
    mulberry.tensor.store %value, %tensor[%i, %j] : f32, !mulberry.tensor<2x3xf32>
    %loaded = mulberry.tensor.load %tensor[%i, %j] : !mulberry.tensor<2x3xf32> -> f32
    return %loaded : f32
  }
}

// CHECK-LABEL: func.func @tensor_ops
// CHECK: mulberry.tensor.alloc() : !mulberry.tensor<2x3xf32>
// CHECK: mulberry.tensor.alloc(%{{.*}}, %{{.*}}) : !mulberry.tensor<?x?xf32>
// CHECK: mulberry.tensor.dim
// CHECK: mulberry.tensor.cast
// CHECK: mulberry.tensor.store
// CHECK: mulberry.tensor.load
