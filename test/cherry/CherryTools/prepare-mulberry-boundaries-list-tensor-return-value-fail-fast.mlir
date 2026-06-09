// RUN: not cherry-opt --prepare-mulberry-boundaries %s 2>&1 | FileCheck %s

module {
  func.func @return_tensor_list(%n: index, %m: index)
      -> !mulberry.list<!mulberry.tensor<?x?xf32>> {
    %tensor = mulberry.tensor.alloc(%n, %m) : !mulberry.tensor<?x?xf32>
    %list = mulberry.list.create(%tensor)
        : (!mulberry.tensor<?x?xf32>)
            -> !mulberry.list<!mulberry.tensor<?x?xf32>>
    return %list : !mulberry.list<!mulberry.tensor<?x?xf32>>
  }
}

// CHECK: source-level List<Tensor> return value boundary preparation is not implemented yet
