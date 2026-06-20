// RUN: cherry-opt --prepare-mulberry-boundaries %s | FileCheck %s

module {
  func.func @use_tensor_list_desc(
      %xs: !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>) -> i64 {
    %index = arith.constant 0 : index
    %size = mulberry.list.desc_length %xs
        : !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
    %desc = mulberry.list.desc_get %xs[%index]
        : !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
            -> !mulberry.tensor_desc<?x?xf32>
    %tensor = mulberry.tensor.desc_unpack %desc
        : !mulberry.tensor_desc<?x?xf32>
            -> !mulberry.tensor<?x?xf32>
    %result = arith.constant 0 : i64
    return %result : i64
  }
}

// CHECK-LABEL: func.func @use_tensor_list_desc
// CHECK-SAME: !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
// CHECK: mulberry.list.desc_length
// CHECK: mulberry.list.desc_get
// CHECK-SAME: -> !mulberry.tensor_desc<?x?xf32>
// CHECK: mulberry.tensor.desc_unpack
// CHECK-SAME: -> !mulberry.tensor<?x?xf32>
