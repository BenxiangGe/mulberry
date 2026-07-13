// RUN: mulberry-opt --load-dialect-plugin=%mulberry_libs/MulberryNNPackage%shlibext --load-pass-plugin=%mulberry_libs/MulberryNNPackage%shlibext --pass-pipeline='builtin.module(prepare-mulberry-nn-calls)' %s | FileCheck %s

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
  func.func private @mulberry.nn.sigmoid(!tensor_f32) -> !tensor_f32

  func.func @bridge(%input: !tensor_f32) -> !tensor_f32 {
    %result = call @mulberry.nn.sigmoid(%input)
        : (!tensor_f32) -> !tensor_f32
    return %result : !tensor_f32
  }
}

// CHECK-LABEL: func.func @bridge
// CHECK: %[[VIEW:.*]] = mulberry_core.tensor.view
// CHECK: %[[CALL:.*]] = call @mulberry.nn.__tensor.sigmoid(%[[VIEW]])
// CHECK: %[[PACK:.*]] = mulberry_core.tensor.pack %[[CALL]]
// CHECK-NEXT: mulberry_core.tensor.release %[[CALL]]
// CHECK: return %[[PACK]]
// CHECK-NOT: call @mulberry.nn.sigmoid
