// RUN: cherry-opt --prepare-mulberry-boundaries %s | FileCheck %s

module {
  func.func private @use_tensor_list(
      %xs: !mulberry.list<!mulberry.tensor<?x?xf32>>, %index: index) {
    %tensor = mulberry.list.get %xs[%index]
        : !mulberry.list<!mulberry.tensor<?x?xf32>>
            -> !mulberry.tensor<?x?xf32>
    return
  }
}

// CHECK-LABEL: func.func private @use_tensor_list
// CHECK-SAME: %[[XS:.*]]: !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
// CHECK-SAME: %[[INDEX:.*]]: index
// CHECK: %[[DESC:.*]] = mulberry.list.desc_get %[[XS]][%[[INDEX]]]
// CHECK-SAME: !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>> -> !mulberry.tensor_desc<?x?xf32>
// CHECK: mulberry.tensor.handle_from_desc %[[DESC]]
// CHECK-SAME: !mulberry.tensor_desc<?x?xf32> -> !mulberry.tensor_handle<?x?xf32>
// CHECK: return
