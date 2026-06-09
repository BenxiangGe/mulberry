// RUN: not cherry-opt --lower-mulberry %s 2>&1 | FileCheck %s

module {
  func.func private @use_tensor_desc_arg(!mulberry.tensor_desc<2x3xf32>)

  func.func private @use_tensor_desc_return()
      -> !mulberry.tensor_desc<2x3xf32>
}

// CHECK: Tensor descriptor function boundaries are not supported yet
