// RUN: cherry-opt %s | FileCheck %s

module {
  func.func @pack_unpack(%tensor: memref<30x?xf32>)
      -> memref<30x?xf32> {
    %descriptor = mulberry.tensor.pack %tensor : memref<30x?xf32> -> !mulberry.record<TensorDescriptorFloat32Rank2 {data: !mulberry.ptr<f32>, shape: !mulberry.record<TensorShapeRank2 {dim0: i64, dim1: i64}>}>
    %unpacked = mulberry.tensor.unpack %descriptor : !mulberry.record<TensorDescriptorFloat32Rank2 {data: !mulberry.ptr<f32>, shape: !mulberry.record<TensorShapeRank2 {dim0: i64, dim1: i64}>}> -> memref<30x?xf32>
    return %unpacked : memref<30x?xf32>
  }
}

// CHECK-LABEL: func.func @pack_unpack
// CHECK: mulberry.tensor.pack {{.*}} : memref<30x?xf32> -> !mulberry.record<TensorDescriptorFloat32Rank2 {data: !mulberry.ptr<f32>, shape: !mulberry.record<TensorShapeRank2 {dim0: i64, dim1: i64}>}>
// CHECK: mulberry.tensor.unpack {{.*}} : !mulberry.record<TensorDescriptorFloat32Rank2 {data: !mulberry.ptr<f32>, shape: !mulberry.record<TensorShapeRank2 {dim0: i64, dim1: i64}>}> -> memref<30x?xf32>
