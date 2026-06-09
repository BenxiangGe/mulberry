// RUN: not cherry-opt --prepare-mulberry-boundaries %s 2>&1 | FileCheck %s

module {
  func.func @call_tensor_list(%n: index, %m: index) {
    %tensor = mulberry.tensor.alloc(%n, %m) : !mulberry.tensor<?x?xf32>
    %list = mulberry.list.create(%tensor)
        : (!mulberry.tensor<?x?xf32>)
            -> !mulberry.list<!mulberry.tensor<?x?xf32>>
    call @use_tensor_list(%list, %n)
        : (!mulberry.list<!mulberry.tensor<?x?xf32>>, index) -> ()
    return
  }

  func.func private @use_tensor_list(
      %xs: !mulberry.list<!mulberry.tensor<?x?xf32>>, %index: index)
      -> () {
    %tensor = mulberry.list.get %xs[%index]
        : !mulberry.list<!mulberry.tensor<?x?xf32>>
            -> !mulberry.tensor<?x?xf32>
    %dim = mulberry.tensor.dim %tensor, %index
        : !mulberry.tensor<?x?xf32>
    return
  }
}

// CHECK: List<Tensor> boundary rewrite cannot expose tensor_handle to downstream users yet
