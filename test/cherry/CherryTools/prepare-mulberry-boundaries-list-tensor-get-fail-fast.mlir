// RUN: not cherry-opt --prepare-mulberry-boundaries %s 2>&1 | FileCheck %s

module {
  func.func private @consume_tensor_list_element(
      %xs: !mulberry.list<!mulberry.tensor<?x?xf32>>, %index: index)
      -> index {
    %tensor = mulberry.list.get %xs[%index]
        : !mulberry.list<!mulberry.tensor<?x?xf32>>
            -> !mulberry.tensor<?x?xf32>
    %dim = mulberry.tensor.dim %tensor, %index
        : !mulberry.tensor<?x?xf32>
    return %dim : index
  }
}

// CHECK: List<Tensor> boundary rewrite cannot expose tensor_handle to downstream users yet
