// RUN: not cherry-opt --prepare-mulberry-boundaries %s 2>&1 | FileCheck %s

module {
  func.func private @make_tensor_list()
      -> !mulberry.list<!mulberry.tensor<?x?xf32>>
}

// CHECK: source-level List<Tensor> function return boundary preparation is not implemented yet
