// RUN: cherry-opt --convert-cherry-to-llvm %s | FileCheck %s

module {
  func.func @pack_only(%tensor: memref<30x?xf32>)
      -> !mulberry.record<TensorDescriptorFloat32Rank2 {allocated: !mulberry.ptr<f32>, aligned: !mulberry.ptr<f32>, offset: i64, sizes: !mulberry.record<TensorSizesRank2 {dim0: i64, dim1: i64}>, strides: !mulberry.record<TensorStridesRank2 {dim0: i64, dim1: i64}>}> {
    %descriptor = mulberry.tensor.pack %tensor : memref<30x?xf32> -> !mulberry.record<TensorDescriptorFloat32Rank2 {allocated: !mulberry.ptr<f32>, aligned: !mulberry.ptr<f32>, offset: i64, sizes: !mulberry.record<TensorSizesRank2 {dim0: i64, dim1: i64}>, strides: !mulberry.record<TensorStridesRank2 {dim0: i64, dim1: i64}>}>
    return %descriptor : !mulberry.record<TensorDescriptorFloat32Rank2 {allocated: !mulberry.ptr<f32>, aligned: !mulberry.ptr<f32>, offset: i64, sizes: !mulberry.record<TensorSizesRank2 {dim0: i64, dim1: i64}>, strides: !mulberry.record<TensorStridesRank2 {dim0: i64, dim1: i64}>}>
  }
}

// CHECK-LABEL: llvm.func @pack_only
// CHECK-SAME: -> !llvm.struct<(ptr, ptr, i64, struct<(i64, i64)>, struct<(i64, i64)>)>
// CHECK: llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: llvm.extractvalue {{.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: llvm.extractvalue {{.*}}[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: llvm.extractvalue {{.*}}[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: llvm.insertvalue {{.*}}[3] : !llvm.struct<(ptr, ptr, i64, struct<(i64, i64)>, struct<(i64, i64)>)>
// CHECK: llvm.insertvalue {{.*}}[4] : !llvm.struct<(ptr, ptr, i64, struct<(i64, i64)>, struct<(i64, i64)>)>
// CHECK-NOT: mulberry.tensor.pack
