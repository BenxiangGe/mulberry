// RUN: cherry-opt %s | FileCheck %s

module {
  func.func private @use_static_tensor_desc(
      !mulberry.tensor_desc<2x3xf32>)
  func.func private @use_dynamic_tensor_desc(
      !mulberry.tensor_desc<?x?xf32>)
}

// CHECK: func.func private @use_static_tensor_desc(!mulberry.tensor_desc<2x3xf32>)
// CHECK: func.func private @use_dynamic_tensor_desc(!mulberry.tensor_desc<?x?xf32>)
