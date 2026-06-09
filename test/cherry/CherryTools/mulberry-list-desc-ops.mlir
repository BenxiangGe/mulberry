// RUN: cherry-opt %s | FileCheck %s

module {
  func.func @list_desc_ops(%length: index) -> index {
    %storage = mulberry.list.alloc %length : !mulberry.list_storage<i64>
    %desc = mulberry.list.desc_pack %length, %storage
        : !mulberry.list_storage<i64> -> !mulberry.list_desc<i64>
    %size = mulberry.list.desc_length %desc : !mulberry.list_desc<i64>
    %data = mulberry.list.desc_data %desc
        : !mulberry.list_desc<i64> -> !mulberry.list_storage<i64>
    %abi = mulberry.list.desc_to_abi %desc
        : !mulberry.list_desc<i64> -> !llvm.struct<(i64, ptr)>
    return %size : index
  }

  func.func @tensor_list_desc_ops(%length: index) -> index {
    %storage = mulberry.list.alloc %length
        : !mulberry.list_storage<memref<?x?xf32>>
    %desc = mulberry.list.desc_pack %length, %storage
        : !mulberry.list_storage<memref<?x?xf32>>
            -> !mulberry.list_desc<memref<?x?xf32>>
    %size = mulberry.list.desc_length %desc
        : !mulberry.list_desc<memref<?x?xf32>>
    %data = mulberry.list.desc_data %desc
        : !mulberry.list_desc<memref<?x?xf32>>
            -> !mulberry.list_storage<memref<?x?xf32>>
    %index = arith.constant 0 : index
    %element = mulberry.list.desc_get %desc[%index]
        : !mulberry.list_desc<memref<?x?xf32>> -> memref<?x?xf32>
    return %size : index
  }

  func.func @list_to_desc_ops(%a: !mulberry.tensor<2x2xf32>,
                              %b: !mulberry.tensor<2x2xf32>) {
    %list = mulberry.list.create(%a, %b)
        : (!mulberry.tensor<2x2xf32>, !mulberry.tensor<2x2xf32>)
            -> !mulberry.list<!mulberry.tensor<2x2xf32>>
    %desc = mulberry.list.to_desc %list
        : !mulberry.list<!mulberry.tensor<2x2xf32>>
            -> !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
    return
  }
}

// CHECK-LABEL: func.func @list_desc_ops
// CHECK: mulberry.list.desc_pack
// CHECK-SAME: !mulberry.list_storage<i64> -> !mulberry.list_desc<i64>
// CHECK: mulberry.list.desc_length
// CHECK-SAME: !mulberry.list_desc<i64>
// CHECK: mulberry.list.desc_data
// CHECK-SAME: !mulberry.list_desc<i64> -> !mulberry.list_storage<i64>
// CHECK: mulberry.list.desc_to_abi
// CHECK-SAME: !mulberry.list_desc<i64> -> !llvm.struct<(i64, ptr)>
// CHECK-LABEL: func.func @tensor_list_desc_ops
// CHECK: mulberry.list.desc_pack
// CHECK-SAME: !mulberry.list_storage<memref<?x?xf32>> -> !mulberry.list_desc<memref<?x?xf32>>
// CHECK: mulberry.list.desc_length
// CHECK-SAME: !mulberry.list_desc<memref<?x?xf32>>
// CHECK: mulberry.list.desc_data
// CHECK-SAME: !mulberry.list_desc<memref<?x?xf32>> -> !mulberry.list_storage<memref<?x?xf32>>
// CHECK: mulberry.list.desc_get
// CHECK-SAME: !mulberry.list_desc<memref<?x?xf32>> -> memref<?x?xf32>
// CHECK-LABEL: func.func @list_to_desc_ops
// CHECK: mulberry.list.to_desc
// CHECK-SAME: !mulberry.list<!mulberry.tensor<2x2xf32>> -> !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
