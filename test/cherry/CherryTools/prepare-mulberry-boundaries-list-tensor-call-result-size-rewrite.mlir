// RUN: cherry-opt --prepare-mulberry-boundaries %s | FileCheck %s

module {
  func.func @call_tensor_list_result(%n: index, %m: index) -> i64 {
    %list = call @make_tensor_list(%n, %m)
        : (index, index) -> !mulberry.list<!mulberry.tensor<?x?xf32>>
    %size = mulberry.list.size %list
        : !mulberry.list<!mulberry.tensor<?x?xf32>>
    return %size : i64
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
// CHECK: %[[LENGTH:.*]] = mulberry.list.desc_length %[[LIST_DESC]]
// CHECK-SAME: !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
// CHECK: %[[SIZE:.*]] = arith.index_cast %[[LENGTH]] : index to i64
// CHECK: return %[[SIZE]] : i64
// CHECK-LABEL: func.func private @make_tensor_list
// CHECK-SAME: -> !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
// CHECK: mulberry.list.escape_storage
// CHECK: return %{{.*}} : !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
