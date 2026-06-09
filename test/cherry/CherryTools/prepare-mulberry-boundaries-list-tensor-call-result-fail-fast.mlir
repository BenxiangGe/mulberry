// RUN: not cherry-opt --prepare-mulberry-boundaries %s 2>&1 | FileCheck %s

module {
  func.func @call_tensor_list_result() {
    %list = call @make_tensor_list()
        : () -> !mulberry.list<!mulberry.tensor<?x?xf32>>
    return
  }

  func.func private @make_tensor_list()
      -> !mulberry.list<!mulberry.tensor<?x?xf32>>
}

// CHECK: source-level List<Tensor> call result boundary preparation is not implemented yet
