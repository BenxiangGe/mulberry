// RUN: cherry-opt --prepare-mulberry-boundaries %s | FileCheck %s

module {
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

// CHECK-LABEL: func.func private @make_tensor_list
// CHECK-SAME: -> (i64, !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>)
// CHECK: %[[SIZE:.*]] = arith.constant 1 : i64
// CHECK: %[[TENSOR:.*]] = mulberry.tensor.alloc
// CHECK: %[[LENGTH:.*]] = arith.constant 1 : index
// CHECK: %[[STORAGE:.*]] = mulberry.list.alloc %[[LENGTH]]
// CHECK-SAME: !mulberry.list_storage<!mulberry.tensor_desc<?x?xf32>>
// CHECK: %[[TENSOR_DESC:.*]] = mulberry.tensor.desc_pack %[[TENSOR]]
// CHECK: mulberry.list.store %[[TENSOR_DESC]], %[[STORAGE]]
// CHECK: %[[ESCAPED:.*]] = mulberry.list.escape_storage %[[STORAGE]], %[[LENGTH]]
// CHECK: %[[DESC:.*]] = mulberry.list.desc_pack %[[LENGTH]], %[[ESCAPED]]
// CHECK: return %[[SIZE]], %[[DESC]] : i64, !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
