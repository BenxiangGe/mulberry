// RUN: not cherry-opt %s 2>&1 | FileCheck %s

module {
  func.func @cherry_nn_rejects_tensor_handle() -> i64 {
    %tensor = mulberry.tensor.alloc() : !mulberry.tensor<2x3xf32>
    %desc = mulberry.tensor.desc_pack %tensor
        : !mulberry.tensor<2x3xf32> -> !mulberry.tensor_desc<2x3xf32>
    %handle = mulberry.tensor.handle_from_desc %desc
        : !mulberry.tensor_desc<2x3xf32>
            -> !mulberry.tensor_handle<?x3xf32>
    %result = cherry_nn.argmax %handle
        : !mulberry.tensor_handle<?x3xf32> -> i64
    return %result : i64
  }
}

// CHECK: error: 'cherry_nn.argmax' op operand #0 must be Mulberry tensor type
