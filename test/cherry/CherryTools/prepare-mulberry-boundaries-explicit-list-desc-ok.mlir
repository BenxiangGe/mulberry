// RUN: cherry-opt --prepare-mulberry-boundaries %s | FileCheck %s

module {
  func.func @call_explicit_tensor_list_desc(
      %desc: !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>) {
    call @use_tensor_list_desc(%desc)
        : (!mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>) -> ()
    return
  }

  func.func private @use_tensor_list_desc(
      !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>)
}

// CHECK-LABEL: func.func @call_explicit_tensor_list_desc
// CHECK-SAME: !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
// CHECK: call @use_tensor_list_desc
// CHECK-SAME: !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
