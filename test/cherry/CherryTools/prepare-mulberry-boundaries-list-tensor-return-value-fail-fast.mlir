// RUN: not cherry-opt --prepare-mulberry-boundaries %s 2>&1 | FileCheck %s

module {
  func.func private @return_tensor_list(%n: index, %m: index)
      -> !mulberry.list<!mulberry.tensor<?x?xf32>> {
    %list = call @make_tensor_list(%n, %m)
        : (index, index)
            -> !mulberry.list<!mulberry.tensor<?x?xf32>>
    return %list : !mulberry.list<!mulberry.tensor<?x?xf32>>
  }

  func.func private @make_tensor_list(%n: index, %m: index)
      -> !mulberry.list<!mulberry.tensor<?x?xf32>>
}

// CHECK: List<Tensor> return rewrite only supports returning a local list.create for now
