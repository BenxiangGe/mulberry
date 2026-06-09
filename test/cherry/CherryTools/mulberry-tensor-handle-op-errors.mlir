// RUN: not cherry-opt %s 2>&1 | FileCheck %s

module {
  func.func @bad_handle_element_type() {
    %tensor = mulberry.tensor.alloc() : !mulberry.tensor<2x3xf32>
    %desc = mulberry.tensor.desc_pack %tensor
        : !mulberry.tensor<2x3xf32> -> !mulberry.tensor_desc<2x3xf32>
    %handle = mulberry.tensor.handle_from_desc %desc
        : !mulberry.tensor_desc<2x3xf32>
            -> !mulberry.tensor_handle<2x3xi64>
    return
  }

  func.func @bad_handle_shape() {
    %tensor = mulberry.tensor.alloc() : !mulberry.tensor<2x3xf32>
    %desc = mulberry.tensor.desc_pack %tensor
        : !mulberry.tensor<2x3xf32> -> !mulberry.tensor_desc<2x3xf32>
    %handle = mulberry.tensor.handle_from_desc %desc
        : !mulberry.tensor_desc<2x3xf32>
            -> !mulberry.tensor_handle<4x3xf32>
    return
  }
}

// CHECK: error: 'mulberry.tensor.handle_from_desc' op handle element type must match descriptor element type
// CHECK: error: 'mulberry.tensor.handle_from_desc' op handle shape must be compatible with descriptor shape
