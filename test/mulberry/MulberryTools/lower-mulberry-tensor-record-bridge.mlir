// RUN: mulberry-opt --lower-mulberry %s | FileCheck %s

!list_u64 = !mulberry_core.record<ListU64 {
  length: i64,
  capacity: i64,
  data: !mulberry_core.ptr<i64>}>
!tensor_f32 = !mulberry_core.record<TensorF32 {
  data: !mulberry_core.ptr<f32>,
  rank: i64,
  numel: i64,
  sizes: !mulberry_core.ptr<!list_u64>,
  strides: !mulberry_core.ptr<!list_u64>}>

module {
  func.func @view(%tensor: !tensor_f32) -> !mulberry_core.tensor<?x?xf32> {
    %result = mulberry_core.tensor.view %tensor
        : !tensor_f32 -> !mulberry_core.tensor<?x?xf32>
    return %result : !mulberry_core.tensor<?x?xf32>
  }

  func.func @pack(%tensor: !mulberry_core.tensor<?x?xf32>)
      -> !tensor_f32 {
    %result = mulberry_core.tensor.pack %tensor
        : !mulberry_core.tensor<?x?xf32> -> !tensor_f32
    return %result : !tensor_f32
  }

  func.func @release(%tensor: !mulberry_core.tensor<?x?xf32>) {
    mulberry_core.tensor.release %tensor
        : !mulberry_core.tensor<?x?xf32>
    return
  }
}

// CHECK-LABEL: func.func @view
// CHECK: llvm.extractvalue
// CHECK: llvm.load
// CHECK: memref.reinterpret_cast
// CHECK: return

// CHECK-LABEL: func.func @pack
// CHECK: memref.copy
// CHECK: llvm.store
// CHECK: llvm.insertvalue
// CHECK: return
// CHECK-NOT: mulberry_core.

// CHECK-LABEL: func.func @release
// CHECK: memref.dealloc %arg0
// CHECK: return
