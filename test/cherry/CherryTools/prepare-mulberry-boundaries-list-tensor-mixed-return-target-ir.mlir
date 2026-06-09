// RUN: cherry-opt --prepare-mulberry-boundaries %s | FileCheck %s

module {
  func.func @make_tensor_list(%n: index, %m: index)
      -> (i64, !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>) {
    %size = arith.constant 1 : i64
    %tensor = mulberry.tensor.alloc(%n, %m) : !mulberry.tensor<?x?xf32>
    %list = mulberry.list.create(%tensor)
        : (!mulberry.tensor<?x?xf32>)
            -> !mulberry.list<!mulberry.tensor<?x?xf32>>
    %desc = mulberry.list.to_desc %list
        : !mulberry.list<!mulberry.tensor<?x?xf32>>
            -> !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
    return %size, %desc
        : i64, !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
  }
}

// CHECK-LABEL: func.func @make_tensor_list
// CHECK-SAME: -> (i64, !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>)
// CHECK: %[[SIZE:.*]] = arith.constant 1 : i64
// CHECK: %[[DESC:.*]] = mulberry.list.to_desc
// CHECK-SAME: !mulberry.list<!mulberry.tensor<?x?xf32>> -> !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
// CHECK: return %[[SIZE]], %[[DESC]] : i64, !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
