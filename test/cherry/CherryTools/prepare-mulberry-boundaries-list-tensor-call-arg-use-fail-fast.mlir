// RUN: cherry-opt --prepare-mulberry-boundaries %s | FileCheck %s

module {
  func.func @call_tensor_list(%n: index, %m: index) {
    %tensor = mulberry.tensor.alloc(%n, %m) : !mulberry.tensor<?x?xf32>
    %list = mulberry.list.create(%tensor)
        : (!mulberry.tensor<?x?xf32>)
            -> !mulberry.list<!mulberry.tensor<?x?xf32>>
    call @use_tensor_list(%list, %n)
        : (!mulberry.list<!mulberry.tensor<?x?xf32>>, index) -> ()
    return
  }

  func.func private @use_tensor_list(
      %xs: !mulberry.list<!mulberry.tensor<?x?xf32>>, %index: index)
      -> () {
    %tensor = mulberry.list.get %xs[%index]
        : !mulberry.list<!mulberry.tensor<?x?xf32>>
            -> !mulberry.tensor<?x?xf32>
    %dim = mulberry.tensor.dim %tensor, %index
        : !mulberry.tensor<?x?xf32>
    return
  }
}

// CHECK-LABEL: func.func @call_tensor_list
// CHECK: %[[DESC:.*]] = mulberry.list.to_desc
// CHECK-SAME: -> !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
// CHECK: call @use_tensor_list(%[[DESC]]
// CHECK-SAME: (!mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>, index) -> ()
// CHECK-LABEL: func.func private @use_tensor_list
// CHECK-SAME: %[[XS:.*]]: !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
// CHECK: %[[TENSOR_DESC:.*]] = mulberry.list.desc_get %[[XS]]
// CHECK: %[[TENSOR:.*]] = mulberry.tensor.desc_unpack %[[TENSOR_DESC]]
// CHECK: mulberry.tensor.dim %[[TENSOR]]
