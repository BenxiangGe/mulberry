// RUN: cherry-opt --prepare-mulberry-boundaries %s | FileCheck %s

module {
  func.func @call_tensor_list(%n: index, %m: index) -> i64 {
    %tensor = mulberry.tensor.alloc(%n, %m) : !mulberry.tensor<?x?xf32>
    %list = mulberry.list.create(%tensor)
        : (!mulberry.tensor<?x?xf32>)
            -> !mulberry.list<!mulberry.tensor<?x?xf32>>
    %result = call @use_tensor_list(%list)
        : (!mulberry.list<!mulberry.tensor<?x?xf32>>) -> i64
    return %result : i64
  }

  func.func private @use_tensor_list(
      %xs: !mulberry.list<!mulberry.tensor<?x?xf32>>) -> i64 {
    %size = mulberry.list.size %xs
        : !mulberry.list<!mulberry.tensor<?x?xf32>>
    return %size : i64
  }
}

// CHECK-LABEL: func.func @call_tensor_list
// CHECK-SAME: -> i64
// CHECK: %[[DESC:.*]] = mulberry.list.to_desc %{{[0-9]+}} : !mulberry.list<!mulberry.tensor<?x?xf32>> -> !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
// CHECK: %[[RESULT:.*]] = call @use_tensor_list(%[[DESC]])
// CHECK-SAME: (!mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>) -> i64
// CHECK: return %[[RESULT]] : i64
// CHECK-LABEL: func.func private @use_tensor_list
// CHECK-SAME: %[[XS:.*]]: !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>) -> i64
// CHECK: %[[LENGTH:.*]] = mulberry.list.desc_length %[[XS]]
// CHECK: %[[SIZE:.*]] = arith.index_cast %[[LENGTH]] : index to i64
// CHECK: return %[[SIZE]] : i64
