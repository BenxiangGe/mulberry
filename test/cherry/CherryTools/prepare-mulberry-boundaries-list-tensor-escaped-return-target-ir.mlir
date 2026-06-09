// RUN: cherry-opt --prepare-mulberry-boundaries %s | FileCheck %s

module {
  func.func @make_tensor_list(%length: index)
      -> !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>> {
    %storage = mulberry.list.alloc %length
        : !mulberry.list_storage<!mulberry.tensor_desc<?x?xf32>>
    %escaped = mulberry.list.escape_storage %storage, %length
        : !mulberry.list_storage<!mulberry.tensor_desc<?x?xf32>>
            -> !mulberry.list_storage<!mulberry.tensor_desc<?x?xf32>>
    %desc = mulberry.list.desc_pack %length, %escaped
        : !mulberry.list_storage<!mulberry.tensor_desc<?x?xf32>>
            -> !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
    return %desc : !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
  }
}

// CHECK-LABEL: func.func @make_tensor_list
// CHECK-SAME: -> !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
// CHECK: %[[STORAGE:.*]] = mulberry.list.alloc %[[LENGTH:.*]] : !mulberry.list_storage<!mulberry.tensor_desc<?x?xf32>>
// CHECK: %[[ESCAPED:.*]] = mulberry.list.escape_storage %[[STORAGE]], %[[LENGTH]]
// CHECK-SAME: !mulberry.list_storage<!mulberry.tensor_desc<?x?xf32>> -> !mulberry.list_storage<!mulberry.tensor_desc<?x?xf32>>
// CHECK: %[[DESC:.*]] = mulberry.list.desc_pack %{{.*}}, %[[ESCAPED]]
// CHECK-SAME: !mulberry.list_storage<!mulberry.tensor_desc<?x?xf32>> -> !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
// CHECK: return %[[DESC]] : !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
