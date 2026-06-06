// RUN: not cherry-opt --lower-mulberry %s 2>&1 | FileCheck %s

module {
  func.func private @use_record(
      !mulberry.record<Layer {weights: !mulberry.tensor<2xf32>}>)
}

// CHECK: failed to legalize operation 'func.func'
// CHECK: !mulberry.record<Layer {weights: !mulberry.tensor<2xf32>}>
