// RUN: not cherry-opt --lower-mulberry %s 2>&1 | FileCheck %s

module {
  func.func private @use_tensor_list(
      !mulberry.list<!mulberry.tensor<2xf32>>)
}

// CHECK: failed to legalize operation 'func.func'
