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
// CHECK: %[[SLOT:.*]] = llvm.alloca %{{.*}} x !llvm.struct<(ptr, array<1 x i64>, array<1 x i64>)>
// CHECK: %[[TENSOR:.*]] = memref.alloc() : memref<2xf32>
// CHECK: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, array<1 x i64>, array<1 x i64>)>
// CHECK: memref.extract_aligned_pointer_as_index %[[TENSOR]]
// CHECK: llvm.store %{{.*}}, %[[SLOT]]
// CHECK: %[[LOADED_DESC:.*]] = llvm.load %[[SLOT]]
// CHECK: builtin.unrealized_conversion_cast
// CHECK: %[[LOADED:.*]] = memref.memory_space_cast
// CHECK: return %[[LOADED]] : memref<2xf32>
// CHECK-NOT: mulberry.
