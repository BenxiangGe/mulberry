// RUN: cherry-opt --lower-mulberry %s | FileCheck %s

module {
  func.func @main() -> !mulberry.tensor<2xf32> {
    %ptr = mulberry.alloca !mulberry.tensor<2xf32>
        : !mulberry.ptr<!mulberry.tensor<2xf32>>
    %tensor = mulberry.tensor.alloc() : !mulberry.tensor<2xf32>
    mulberry.store %tensor, %ptr : !mulberry.tensor<2xf32>,
        !mulberry.ptr<!mulberry.tensor<2xf32>>
    %loaded = mulberry.load %ptr
        : !mulberry.ptr<!mulberry.tensor<2xf32>> -> !mulberry.tensor<2xf32>
    return %loaded : !mulberry.tensor<2xf32>
  }
}

// CHECK-LABEL: func.func @main
// CHECK-SAME: -> memref<2xf32>
// CHECK: %[[SLOT:.*]] = memref.alloca() : memref<memref<2xf32>>
// CHECK: %[[TENSOR:.*]] = memref.alloc() : memref<2xf32>
// CHECK: memref.store %[[TENSOR]], %[[SLOT]][] : memref<memref<2xf32>>
// CHECK: %[[LOADED:.*]] = memref.load %[[SLOT]][] : memref<memref<2xf32>>
// CHECK: return %[[LOADED]] : memref<2xf32>
// CHECK-NOT: mulberry.
