// RUN: mulberry-opt --lower-mulberry %s | FileCheck %s

!tensor_f32 = !mulberry_core.tensor<?x?xf32>

module {
  func.func @stack_round_trip(%tensor: !tensor_f32) -> !tensor_f32 {
    %slot = mulberry_core.alloca !tensor_f32
        : !mulberry_core.ptr<!tensor_f32>
    mulberry_core.store %tensor, %slot
        : !tensor_f32, !mulberry_core.ptr<!tensor_f32>
    %result = mulberry_core.load %slot
        : !mulberry_core.ptr<!tensor_f32> -> !tensor_f32
    return %result : !tensor_f32
  }

  func.func @heap_round_trip(%tensor: !tensor_f32, %index: index)
      -> !tensor_f32 {
    %count = arith.constant 2 : index
    %storage = mulberry_core.heap.alloc !tensor_f32, %count
        : !mulberry_core.ptr<!tensor_f32>
    %slot = mulberry_core.ptr.index %storage[%index]
        : !mulberry_core.ptr<!tensor_f32>
        -> !mulberry_core.ptr<!tensor_f32>
    mulberry_core.store %tensor, %slot
        : !tensor_f32, !mulberry_core.ptr<!tensor_f32>
    %result = mulberry_core.load %slot
        : !mulberry_core.ptr<!tensor_f32> -> !tensor_f32
    return %result : !tensor_f32
  }
}

// CHECK-LABEL: func.func @stack_round_trip(
// CHECK-SAME: %[[TENSOR:.*]]: memref<?x?xf32>) -> memref<?x?xf32>
// CHECK: %[[SLOT:.*]] = llvm.alloca {{.*}} x !llvm.struct
// CHECK: llvm.store {{.*}}, %[[SLOT]]
// CHECK: llvm.load %[[SLOT]]
// CHECK: memref.reinterpret_cast
// CHECK: return {{.*}} : memref<?x?xf32>

// CHECK-LABEL: func.func @heap_round_trip(
// CHECK-SAME: %[[TENSOR:.*]]: memref<?x?xf32>, %[[INDEX:.*]]: index)
// CHECK: llvm.getelementptr {{.*}}[1]
// CHECK-SAME: !llvm.struct
// CHECK: llvm.call @mulberry_boehm_malloc
// CHECK: llvm.getelementptr {{.*}}[%{{.*}}]
// CHECK-SAME: !llvm.struct
// CHECK: llvm.store
// CHECK: llvm.load
// CHECK: memref.reinterpret_cast
// CHECK: return {{.*}} : memref<?x?xf32>
// CHECK-NOT: mulberry_core.
