// RUN: cherry-opt %s | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: cherry-opt --canonicalize %s | FileCheck %s --check-prefix=CANON
// RUN: cherry-opt --convert-cherry-to-llvm %s | FileCheck %s --check-prefix=LLVM

module {
  func.func @pack_unpack(%tensor: memref<30x?xf32>)
      -> memref<30x?xf32> {
    %descriptor = mulberry.tensor.pack %tensor : memref<30x?xf32> -> !mulberry.record<TensorDescriptorFloat32Rank2 {allocated: !mulberry.ptr<f32>, aligned: !mulberry.ptr<f32>, offset: i64, sizes: !mulberry.record<TensorSizesRank2 {dim0: i64, dim1: i64}>, strides: !mulberry.record<TensorStridesRank2 {dim0: i64, dim1: i64}>}>
    %unpacked = mulberry.tensor.unpack %descriptor : !mulberry.record<TensorDescriptorFloat32Rank2 {allocated: !mulberry.ptr<f32>, aligned: !mulberry.ptr<f32>, offset: i64, sizes: !mulberry.record<TensorSizesRank2 {dim0: i64, dim1: i64}>, strides: !mulberry.record<TensorStridesRank2 {dim0: i64, dim1: i64}>}> -> memref<30x?xf32>
    return %unpacked : memref<30x?xf32>
  }
}

// ROUNDTRIP-LABEL: func.func @pack_unpack
// ROUNDTRIP: mulberry.tensor.pack {{.*}} : memref<30x?xf32> -> !mulberry.record<TensorDescriptorFloat32Rank2 {allocated: !mulberry.ptr<f32>, aligned: !mulberry.ptr<f32>, offset: i64, sizes: !mulberry.record<TensorSizesRank2 {dim0: i64, dim1: i64}>, strides: !mulberry.record<TensorStridesRank2 {dim0: i64, dim1: i64}>}>
// ROUNDTRIP: mulberry.tensor.unpack {{.*}} : !mulberry.record<TensorDescriptorFloat32Rank2 {allocated: !mulberry.ptr<f32>, aligned: !mulberry.ptr<f32>, offset: i64, sizes: !mulberry.record<TensorSizesRank2 {dim0: i64, dim1: i64}>, strides: !mulberry.record<TensorStridesRank2 {dim0: i64, dim1: i64}>}> -> memref<30x?xf32>

// CANON-LABEL: func.func @pack_unpack
// CANON-SAME: (%[[TENSOR:.*]]: memref<30x?xf32>)
// CANON: return %[[TENSOR]] : memref<30x?xf32>
// CANON-NOT: mulberry.tensor.pack
// CANON-NOT: mulberry.tensor.unpack

// LLVM-LABEL: llvm.func @pack_unpack
// LLVM: llvm.extractvalue {{.*}}[3] : !llvm.struct<(ptr, ptr, i64, struct<(i64, i64)>, struct<(i64, i64)>)>
// LLVM: llvm.extractvalue {{.*}}[4] : !llvm.struct<(ptr, ptr, i64, struct<(i64, i64)>, struct<(i64, i64)>)>
// LLVM: llvm.extractvalue {{.*}}[0] : !llvm.struct<(i64, i64)>
// LLVM: llvm.extractvalue {{.*}}[1] : !llvm.struct<(i64, i64)>
// LLVM: llvm.insertvalue {{.*}}[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// LLVM: llvm.insertvalue {{.*}}[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// LLVM-NOT: mulberry.tensor.unpack
