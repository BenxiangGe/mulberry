// RUN: not cherry-opt %s 2>&1 | FileCheck %s

module {
  func.func @bad_desc_pack(%length: index) {
    %storage = mulberry.list.alloc %length : !mulberry.list_storage<i64>
    %desc = mulberry.list.desc_pack %length, %storage
        : !mulberry.list_storage<i64> -> !mulberry.list_desc<i1>
    return
  }

  func.func @bad_desc_to_abi(%length: index) {
    %storage = mulberry.list.alloc %length : !mulberry.list_storage<i64>
    %desc = mulberry.list.desc_pack %length, %storage
        : !mulberry.list_storage<i64> -> !mulberry.list_desc<i64>
    %abi = mulberry.list.desc_to_abi %desc
        : !mulberry.list_desc<i64> -> i64
    return
  }

  func.func @bad_list_to_desc(%value: i64) {
    %list = mulberry.list.create(%value)
        : (i64) -> !mulberry.list<i64>
    %desc = mulberry.list.to_desc %list
        : !mulberry.list<i64> -> !mulberry.list_desc<i64>
    return
  }

  func.func @bad_desc_get(%length: index, %index: index) {
    %storage = mulberry.list.alloc %length : !mulberry.list_storage<i64>
    %desc = mulberry.list.desc_pack %length, %storage
        : !mulberry.list_storage<i64> -> !mulberry.list_desc<i64>
    %element = mulberry.list.desc_get %desc[%index]
        : !mulberry.list_desc<i64> -> i1
    return
  }
}

// CHECK: error: 'mulberry.list.desc_pack' op descriptor element type must match list storage element type
// CHECK: error: 'mulberry.list.desc_to_abi' op result type must be a list ABI record `{i64, !llvm.ptr}`
// CHECK: error: 'mulberry.list.to_desc' op only List<Tensor> to list_desc<TensorDesc> is supported
// CHECK: error: 'mulberry.list.desc_get' op result type must match descriptor element type
