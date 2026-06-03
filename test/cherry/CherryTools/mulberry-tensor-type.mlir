// RUN: cherry-opt %s | FileCheck %s

module {
  func.func private @use_tensor(!mulberry.tensor<2x3xf32>)
}

// CHECK: func.func private @use_tensor(!mulberry.tensor<2x3xf32>)
