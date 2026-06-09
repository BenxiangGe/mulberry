// RUN: not cherry-opt %s 2>&1 | FileCheck %s

module {
  func.func @bad_desc_shape() {
    %tensor = mulberry.tensor.alloc() : !mulberry.tensor<2x3xf32>
    %desc = mulberry.tensor.desc_pack %tensor
        : !mulberry.tensor<2x3xf32> -> !mulberry.tensor_desc<4x3xf32>
    return
  }

  func.func @bad_desc_to_abi() {
    %tensor = mulberry.tensor.alloc() : !mulberry.tensor<2x3xf32>
    %desc = mulberry.tensor.desc_pack %tensor
        : !mulberry.tensor<2x3xf32> -> !mulberry.tensor_desc<2x3xf32>
    %abi = mulberry.tensor.desc_to_abi %desc
        : !mulberry.tensor_desc<2x3xf32> -> !llvm.struct<(ptr, array<1 x i64>, array<2 x i64>)>
    return
  }

  func.func @bad_desc_unpack_shape() {
    %tensor = mulberry.tensor.alloc() : !mulberry.tensor<2x3xf32>
    %desc = mulberry.tensor.desc_pack %tensor
        : !mulberry.tensor<2x3xf32> -> !mulberry.tensor_desc<2x3xf32>
    %unpacked = mulberry.tensor.desc_unpack %desc
        : !mulberry.tensor_desc<2x3xf32> -> !mulberry.tensor<4x3xf32>
    return
  }
}

// CHECK: error: 'mulberry.tensor.desc_pack' op descriptor shape must be compatible with tensor shape
// CHECK: error: 'mulberry.tensor.desc_to_abi' op result type must be a tensor ABI record `{ptr, sizes, strides}`
// CHECK: error: 'mulberry.tensor.desc_unpack' op tensor shape must be compatible with descriptor shape
