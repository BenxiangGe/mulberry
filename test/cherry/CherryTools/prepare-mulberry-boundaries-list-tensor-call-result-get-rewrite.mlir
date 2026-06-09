// RUN: cherry-opt --prepare-mulberry-boundaries %s | FileCheck %s

module {
  func.func @call_tensor_list_result(%n: index, %m: index, %index: index) {
    %list = call @make_tensor_list(%n, %m)
        : (index, index) -> !mulberry.list<!mulberry.tensor<?x?xf32>>
    %tensor = mulberry.list.get %list[%index]
        : !mulberry.list<!mulberry.tensor<?x?xf32>>
            -> !mulberry.tensor<?x?xf32>
    return
  }

  func.func private @make_tensor_list(%n: index, %m: index)
      -> !mulberry.list<!mulberry.tensor<?x?xf32>> {
    %tensor = mulberry.tensor.alloc(%n, %m) : !mulberry.tensor<?x?xf32>
    %list = mulberry.list.create(%tensor)
        : (!mulberry.tensor<?x?xf32>)
            -> !mulberry.list<!mulberry.tensor<?x?xf32>>
    return %list : !mulberry.list<!mulberry.tensor<?x?xf32>>
  }
}

// CHECK-LABEL: func.func @call_tensor_list_result
// CHECK: %[[LIST_DESC:.*]] = call @make_tensor_list
// CHECK-SAME: (index, index) -> !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
// CHECK: %[[TENSOR_DESC:.*]] = mulberry.list.desc_get %[[LIST_DESC]]
// CHECK-SAME: !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>> -> !mulberry.tensor_desc<?x?xf32>
// CHECK: mulberry.tensor.desc_unpack %[[TENSOR_DESC]]
// CHECK-SAME: !mulberry.tensor_desc<?x?xf32> -> !mulberry.tensor<?x?xf32>
// CHECK: return
// CHECK-LABEL: func.func private @make_tensor_list
// CHECK-SAME: -> !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
