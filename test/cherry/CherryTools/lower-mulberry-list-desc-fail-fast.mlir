// RUN: cherry-opt --lower-mulberry %s | FileCheck %s

module {
  func.func private @use_list_desc_arg(!mulberry.list_desc<i64>)

  func.func @call_list_desc(%length: index) {
    %storage = mulberry.list.alloc %length : !mulberry.list_storage<i64>
    %desc = mulberry.list.desc_pack %length, %storage
        : !mulberry.list_storage<i64> -> !mulberry.list_desc<i64>
    call @use_list_desc_arg(%desc) : (!mulberry.list_desc<i64>) -> ()
    return
  }
}

// CHECK: func.func private @use_list_desc_arg(!llvm.struct<(i64, ptr)>)
// CHECK-LABEL: func.func @call_list_desc
// CHECK: memref.alloc
// CHECK: llvm.insertvalue
// CHECK: call @use_list_desc_arg
// CHECK-SAME: !llvm.struct<(i64, ptr)>
// CHECK-NOT: mulberry.
