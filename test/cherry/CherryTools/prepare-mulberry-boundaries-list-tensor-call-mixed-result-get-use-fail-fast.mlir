// RUN: cherry-opt --prepare-mulberry-boundaries %s | FileCheck %s

module {
  func.func @call_tensor_list_result(%n: index, %m: index, %index: index)
      -> index {
    %ignoredSize, %list = call @make_tensor_list(%n, %m)
        : (index, index) -> (i64, !mulberry.list<!mulberry.tensor<?x?xf32>>)
    %tensor = mulberry.list.get %list[%index]
        : !mulberry.list<!mulberry.tensor<?x?xf32>>
            -> !mulberry.tensor<?x?xf32>
    %dim = mulberry.tensor.dim %tensor, %index
        : !mulberry.tensor<?x?xf32>
    return %dim : index
  }

  func.func private @make_tensor_list(%n: index, %m: index)
      -> (i64, !mulberry.list<!mulberry.tensor<?x?xf32>>) {
    %size = arith.constant 1 : i64
    %tensor = mulberry.tensor.alloc(%n, %m) : !mulberry.tensor<?x?xf32>
    %list = mulberry.list.create(%tensor)
        : (!mulberry.tensor<?x?xf32>)
            -> !mulberry.list<!mulberry.tensor<?x?xf32>>
    return %size, %list
        : i64, !mulberry.list<!mulberry.tensor<?x?xf32>>
  }
}

// CHECK-LABEL: func.func @call_tensor_list_result
// CHECK: %{{.*}}:2 = call @make_tensor_list
// CHECK-SAME: -> (i64, !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>)
// CHECK: %[[TENSOR_DESC:.*]] = mulberry.list.desc_get %{{.*}}#1
// CHECK: %[[TENSOR:.*]] = mulberry.tensor.desc_unpack %[[TENSOR_DESC]]
// CHECK: mulberry.tensor.dim %[[TENSOR]]
// CHECK: return
