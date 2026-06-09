// RUN: cherry-opt %s | FileCheck %s

module {
  func.func @tensor_handle_ops()
      -> !mulberry.tensor_handle<?x3xf32> {
    %tensor = mulberry.tensor.alloc() : !mulberry.tensor<2x3xf32>
    %desc = mulberry.tensor.desc_pack %tensor
        : !mulberry.tensor<2x3xf32> -> !mulberry.tensor_desc<2x3xf32>
    %handle = mulberry.tensor.handle_from_desc %desc
        : !mulberry.tensor_desc<2x3xf32>
            -> !mulberry.tensor_handle<?x3xf32>
    return %handle : !mulberry.tensor_handle<?x3xf32>
  }
}

// CHECK-LABEL: func.func @tensor_handle_ops
// CHECK: mulberry.tensor.handle_from_desc
// CHECK-SAME: !mulberry.tensor_desc<2x3xf32> -> !mulberry.tensor_handle<?x3xf32>
