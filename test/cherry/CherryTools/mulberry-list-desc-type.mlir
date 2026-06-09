// RUN: cherry-opt %s | FileCheck %s

module {
  func.func private @use_scalar_list_desc(!mulberry.list_desc<i64>)
  func.func private @use_tensor_list_desc(
      !mulberry.list_desc<!mulberry.tensor<?x?xf32>>)
}

// CHECK: func.func private @use_scalar_list_desc(!mulberry.list_desc<i64>)
// CHECK: func.func private @use_tensor_list_desc(!mulberry.list_desc<!mulberry.tensor<?x?xf32>>)
