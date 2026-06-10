// RUN: cherry-opt --prepare-mulberry-boundaries %s | FileCheck %s

module {
  func.func private @consume_tensor_list_element(
      %xs: !mulberry.list<!mulberry.tensor<?x?xf32>>, %index: index)
      -> index {
    %tensor = mulberry.list.get %xs[%index]
        : !mulberry.list<!mulberry.tensor<?x?xf32>>
            -> !mulberry.tensor<?x?xf32>
    %dim = mulberry.tensor.dim %tensor, %index
        : !mulberry.tensor<?x?xf32>
    return %dim : index
  }
}

// CHECK-LABEL: func.func private @consume_tensor_list_element
// CHECK-SAME: %[[XS:.*]]: !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
// CHECK: %[[DESC:.*]] = mulberry.list.desc_get %[[XS]]
// CHECK-SAME: -> !mulberry.tensor_desc<?x?xf32>
// CHECK: %[[TENSOR:.*]] = mulberry.tensor.desc_unpack %[[DESC]]
// CHECK-SAME: -> !mulberry.tensor<?x?xf32>
// CHECK: mulberry.tensor.dim %[[TENSOR]]
// CHECK: return
