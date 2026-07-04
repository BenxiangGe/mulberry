// RUN: mulberry-opt --lower-mulberry %s | FileCheck %s

module {
  func.func @use(%arg0: !mulberry_core.tensor<2x3xf32>)
      -> !mulberry_core.tensor<?x?xf32> {
    %cast = mulberry_core.tensor.cast %arg0 : !mulberry_core.tensor<2x3xf32>
        to !mulberry_core.tensor<?x?xf32>
    return %cast : !mulberry_core.tensor<?x?xf32>
  }

  func.func @main(%i: index, %j: index, %n: index, %m: index) -> f32 {
    %tensor = mulberry_core.tensor.alloc() : !mulberry_core.tensor<2x3xf32>
    %dynamic = mulberry_core.tensor.alloc(%n, %m) : !mulberry_core.tensor<?x?xf32>
    %c0 = arith.constant 0 : index
    %dim = mulberry_core.tensor.dim %dynamic, %c0 : !mulberry_core.tensor<?x?xf32>
    %value = arith.constant 1.000000e+00 : f32
    mulberry_core.tensor.store %value, %tensor[%i, %j] : f32,
        !mulberry_core.tensor<2x3xf32>
    %loaded = mulberry_core.tensor.load %tensor[%i, %j]
        : !mulberry_core.tensor<2x3xf32> -> f32
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
