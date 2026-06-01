// RUN: not cherry-opt -split-input-file %s 2>&1 | FileCheck %s

module {
  func.func @bad_data_field(%tensor: memref<2xf32>) {
    %descriptor = mulberry.tensor.pack %tensor : memref<2xf32> -> !mulberry.record<TensorDescriptorFloat32Rank1 {shape: !mulberry.record<TensorShapeRank1 {dim0: i64}>}>
    return
  }
}

// CHECK: error: 'mulberry.tensor.pack' op tensor descriptor must have data pointer field

// -----

module {
  func.func @bad_element_type(%tensor: memref<2xf32>) {
    %descriptor = mulberry.tensor.pack %tensor : memref<2xf32> -> !mulberry.record<TensorDescriptorUInt64Rank1 {data: !mulberry.ptr<i64>, shape: !mulberry.record<TensorShapeRank1 {dim0: i64}>}>
    return
  }
}

// CHECK: error: 'mulberry.tensor.pack' op tensor descriptor data element type must match memref element type

// -----

module {
  func.func @bad_shape_field(%tensor: memref<2xf32>) {
    %descriptor = mulberry.tensor.pack %tensor : memref<2xf32> -> !mulberry.record<TensorDescriptorFloat32Rank1 {data: !mulberry.ptr<f32>, shape: i64}>
    return
  }
}

// CHECK: error: 'mulberry.tensor.pack' op tensor descriptor must have i64 shape record

// -----

module {
  func.func @bad_rank(%tensor: memref<2xf32>) {
    %descriptor = mulberry.tensor.pack %tensor : memref<2xf32> -> !mulberry.record<TensorDescriptorFloat32Rank2 {data: !mulberry.ptr<f32>, shape: !mulberry.record<TensorShapeRank2 {dim0: i64, dim1: i64}>}>
    return
  }
}

// CHECK: error: 'mulberry.tensor.pack' op tensor descriptor rank must match memref rank

// -----

module {
  func.func @bad_unpack(%descriptor: !mulberry.record<TensorDescriptorFloat32Rank1 {data: !mulberry.ptr<f32>, shape: !mulberry.record<TensorShapeRank1 {dim0: i64}>}>)
      -> memref<2xi64> {
    %tensor = mulberry.tensor.unpack %descriptor : !mulberry.record<TensorDescriptorFloat32Rank1 {data: !mulberry.ptr<f32>, shape: !mulberry.record<TensorShapeRank1 {dim0: i64}>}> -> memref<2xi64>
    return %tensor : memref<2xi64>
  }
}

// CHECK: error: 'mulberry.tensor.unpack' op tensor descriptor data element type must match memref element type
