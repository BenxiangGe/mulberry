// RUN: not cherry-opt --lower-mulberry %s 2>&1 | FileCheck %s

module {
  func.func @nested_tensor_list() -> i64 {
    %a = mulberry.tensor.alloc() : !mulberry.tensor<2xf32>
    %b = mulberry.tensor.alloc() : !mulberry.tensor<2xf32>
    %inner = mulberry.list.create(%a, %b)
        : (!mulberry.tensor<2xf32>, !mulberry.tensor<2xf32>)
            -> !mulberry.list<!mulberry.tensor<2xf32>>
    %outer = mulberry.list.create(%inner)
        : (!mulberry.list<!mulberry.tensor<2xf32>>)
            -> !mulberry.list<!mulberry.list<!mulberry.tensor<2xf32>>>
    %zero = arith.constant 0 : i64
    return %zero : i64
  }
}

// CHECK: failed to legalize operation 'mulberry.list.create'
