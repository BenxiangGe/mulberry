// RUN: not cherry-opt %s 2>&1 | FileCheck %s

module {
  func.func @tensor_handle_is_not_tensor() {
    %tensor = mulberry.tensor.alloc() : !mulberry.tensor<2x3xf32>
    %desc = mulberry.tensor.desc_pack %tensor
        : !mulberry.tensor<2x3xf32> -> !mulberry.tensor_desc<2x3xf32>
    %handle = mulberry.tensor.handle_from_desc %desc
        : !mulberry.tensor_desc<2x3xf32>
            -> !mulberry.tensor_handle<?x3xf32>
    %index = arith.constant 0 : index
    %dim = mulberry.tensor.dim %handle, %index
        : !mulberry.tensor_handle<?x3xf32>
    return
  }
}

// CHECK: error: 'mulberry.tensor.dim' op operand #0 must be Mulberry tensor type
