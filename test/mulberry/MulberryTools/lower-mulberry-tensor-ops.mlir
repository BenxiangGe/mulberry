// RUN: mulberry-opt --lower-mulberry %s | FileCheck %s

module {
  func.func @use(%arg0: !mulberry.tensor<2x3xf32>)
      -> !mulberry.tensor<?x?xf32> {
    %cast = mulberry.tensor.cast %arg0 : !mulberry.tensor<2x3xf32>
        to !mulberry.tensor<?x?xf32>
    return %cast : !mulberry.tensor<?x?xf32>
  }

  func.func @main(%i: index, %j: index, %n: index, %m: index) -> f32 {
    %tensor = mulberry.tensor.alloc() : !mulberry.tensor<2x3xf32>
    %dynamic = mulberry.tensor.alloc(%n, %m) : !mulberry.tensor<?x?xf32>
    %c0 = arith.constant 0 : index
    %dim = mulberry.tensor.dim %dynamic, %c0 : !mulberry.tensor<?x?xf32>
    %value = arith.constant 1.000000e+00 : f32
    mulberry.tensor.store %value, %tensor[%i, %j] : f32,
        !mulberry.tensor<2x3xf32>
    %loaded = mulberry.tensor.load %tensor[%i, %j]
        : !mulberry.tensor<2x3xf32> -> f32
    return %loaded : f32
  }
}

// CHECK-LABEL: func.func @use
// CHECK-SAME: (%[[ARG:.*]]: memref<2x3xf32>) -> memref<?x?xf32>
// CHECK: %[[CAST:.*]] = memref.cast %[[ARG]] : memref<2x3xf32> to memref<?x?xf32>
// CHECK: return %[[CAST]] : memref<?x?xf32>
// CHECK-LABEL: func.func @main
// CHECK: %[[TENSOR:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK: %[[DYNAMIC:.*]] = memref.alloc(%{{.*}}, %{{.*}}) : memref<?x?xf32>
// CHECK: memref.dim %[[DYNAMIC]]
// CHECK: memref.store {{.*}}, %[[TENSOR]]
// CHECK: %[[LOADED:.*]] = memref.load %[[TENSOR]]
// CHECK: return %[[LOADED]] : f32
// CHECK-NOT: mulberry.
