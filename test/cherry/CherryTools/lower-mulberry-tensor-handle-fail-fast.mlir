// RUN: not cherry-opt --lower-mulberry %s 2>&1 | FileCheck %s

module {
  func.func @local_tensor_handle() -> i64 {
    %tensor = mulberry.tensor.alloc() : !mulberry.tensor<2x3xf32>
    %desc = mulberry.tensor.desc_pack %tensor
        : !mulberry.tensor<2x3xf32> -> !mulberry.tensor_desc<2x3xf32>
    %handle = mulberry.tensor.handle_from_desc %desc
        : !mulberry.tensor_desc<2x3xf32>
            -> !mulberry.tensor_handle<?x3xf32>
    %zero = arith.constant 0 : i64
    return %zero : i64
  }
}

// CHECK: failed to legalize operation 'mulberry.tensor.handle_from_desc'
// CHECK: Tensor handle lowering needs explicit reconstruction support
