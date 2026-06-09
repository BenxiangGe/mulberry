// RUN: cherry-opt --prepare-mulberry-boundaries %s | FileCheck %s

module {
  func.func @make_tensor_list(%n: index, %m: index)
      -> !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>> {
    %tensor = mulberry.tensor.alloc(%n, %m) : !mulberry.tensor<?x?xf32>
    %list = mulberry.list.create(%tensor)
        : (!mulberry.tensor<?x?xf32>)
            -> !mulberry.list<!mulberry.tensor<?x?xf32>>
    %desc = mulberry.list.to_desc %list
        : !mulberry.list<!mulberry.tensor<?x?xf32>>
            -> !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
    return %desc : !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
  }
}

// CHECK-LABEL: func.func @make_tensor_list
// CHECK-SAME: -> !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
// CHECK: mulberry.list.create
// CHECK-SAME: -> !mulberry.list<!mulberry.tensor<?x?xf32>>
// CHECK: %[[DESC:.*]] = mulberry.list.to_desc
// CHECK-SAME: !mulberry.list<!mulberry.tensor<?x?xf32>> -> !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
// CHECK: return %[[DESC]] : !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
