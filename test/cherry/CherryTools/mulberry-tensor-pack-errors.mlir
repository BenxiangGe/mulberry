// RUN: not cherry-opt -split-input-file %s 2>&1 | FileCheck %s

module {
  func.func @bad_pointer_fields(%tensor: memref<2xf32>) {
    %descriptor = mulberry.tensor.pack %tensor : memref<2xf32> -> !mulberry.record<TensorDescriptorFloat32Rank1 {allocated: !mulberry.ptr<f32>, offset: i64, sizes: !mulberry.record<TensorSizesRank1 {dim0: i64}>, strides: !mulberry.record<TensorStridesRank1 {dim0: i64}>}>
    return
  }
}

// CHECK: error: 'mulberry.tensor.pack' op tensor descriptor must have matching allocated/aligned pointer fields

// -----

module {
  func.func @bad_pointer_element_type(%tensor: memref<2xf32>) {
    %descriptor = mulberry.tensor.pack %tensor : memref<2xf32> -> !mulberry.record<TensorDescriptorUInt64Rank1 {allocated: !mulberry.ptr<i64>, aligned: !mulberry.ptr<i64>, offset: i64, sizes: !mulberry.record<TensorSizesRank1 {dim0: i64}>, strides: !mulberry.record<TensorStridesRank1 {dim0: i64}>}>
    return
  }
}

// CHECK: error: 'mulberry.tensor.pack' op tensor descriptor pointer element type must match memref element type

// -----

module {
  func.func @bad_mismatched_pointer_fields(%tensor: memref<2xf32>) {
    %descriptor = mulberry.tensor.pack %tensor : memref<2xf32> -> !mulberry.record<TensorDescriptorFloat32Rank1 {allocated: !mulberry.ptr<f32>, aligned: !mulberry.ptr<i64>, offset: i64, sizes: !mulberry.record<TensorSizesRank1 {dim0: i64}>, strides: !mulberry.record<TensorStridesRank1 {dim0: i64}>}>
    return
  }
}

// CHECK: error: 'mulberry.tensor.pack' op tensor descriptor must have matching allocated/aligned pointer fields

// -----

module {
  func.func @bad_offset_field(%tensor: memref<2xf32>) {
    %descriptor = mulberry.tensor.pack %tensor : memref<2xf32> -> !mulberry.record<TensorDescriptorFloat32Rank1 {allocated: !mulberry.ptr<f32>, aligned: !mulberry.ptr<f32>, offset: i1, sizes: !mulberry.record<TensorSizesRank1 {dim0: i64}>, strides: !mulberry.record<TensorStridesRank1 {dim0: i64}>}>
    return
  }
}

// CHECK: error: 'mulberry.tensor.pack' op tensor descriptor must have i64 offset field

// -----

module {
  func.func @bad_sizes_field(%tensor: memref<2xf32>) {
    %descriptor = mulberry.tensor.pack %tensor : memref<2xf32> -> !mulberry.record<TensorDescriptorFloat32Rank1 {allocated: !mulberry.ptr<f32>, aligned: !mulberry.ptr<f32>, offset: i64, sizes: i64, strides: !mulberry.record<TensorStridesRank1 {dim0: i64}>}>
    return
  }
}

// CHECK: error: 'mulberry.tensor.pack' op tensor descriptor must have i64 sizes record

// -----

module {
  func.func @bad_sizes_rank(%tensor: memref<2xf32>) {
    %descriptor = mulberry.tensor.pack %tensor : memref<2xf32> -> !mulberry.record<TensorDescriptorFloat32Rank2 {allocated: !mulberry.ptr<f32>, aligned: !mulberry.ptr<f32>, offset: i64, sizes: !mulberry.record<TensorSizesRank2 {dim0: i64, dim1: i64}>, strides: !mulberry.record<TensorStridesRank1 {dim0: i64}>}>
    return
  }
}

// CHECK: error: 'mulberry.tensor.pack' op tensor descriptor sizes rank must match memref rank

// -----

module {
  func.func @bad_strides_field(%tensor: memref<2xf32>) {
    %descriptor = mulberry.tensor.pack %tensor : memref<2xf32> -> !mulberry.record<TensorDescriptorFloat32Rank1 {allocated: !mulberry.ptr<f32>, aligned: !mulberry.ptr<f32>, offset: i64, sizes: !mulberry.record<TensorSizesRank1 {dim0: i64}>, strides: i64}>
    return
  }
}

// CHECK: error: 'mulberry.tensor.pack' op tensor descriptor must have i64 strides record

// -----

module {
  func.func @bad_strides_rank(%tensor: memref<2xf32>) {
    %descriptor = mulberry.tensor.pack %tensor : memref<2xf32> -> !mulberry.record<TensorDescriptorFloat32Rank2 {allocated: !mulberry.ptr<f32>, aligned: !mulberry.ptr<f32>, offset: i64, sizes: !mulberry.record<TensorSizesRank1 {dim0: i64}>, strides: !mulberry.record<TensorStridesRank2 {dim0: i64, dim1: i64}>}>
    return
  }
}

// CHECK: error: 'mulberry.tensor.pack' op tensor descriptor strides rank must match memref rank

// -----

module {
  func.func @bad_unpack(%descriptor: !mulberry.record<TensorDescriptorFloat32Rank1 {allocated: !mulberry.ptr<f32>, aligned: !mulberry.ptr<f32>, offset: i64, sizes: !mulberry.record<TensorSizesRank1 {dim0: i64}>, strides: !mulberry.record<TensorStridesRank1 {dim0: i64}>}>)
      -> memref<2xi64> {
    %tensor = mulberry.tensor.unpack %descriptor : !mulberry.record<TensorDescriptorFloat32Rank1 {allocated: !mulberry.ptr<f32>, aligned: !mulberry.ptr<f32>, offset: i64, sizes: !mulberry.record<TensorSizesRank1 {dim0: i64}>, strides: !mulberry.record<TensorStridesRank1 {dim0: i64}>}> -> memref<2xi64>
    return %tensor : memref<2xi64>
  }
}

// CHECK: error: 'mulberry.tensor.unpack' op tensor descriptor pointer element type must match memref element type
