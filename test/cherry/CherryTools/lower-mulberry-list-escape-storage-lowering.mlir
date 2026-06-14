// RUN: cherry-opt --lower-mulberry %s | FileCheck %s

module {
  func.func @escape_scalar_storage(%length: index, %index: index)
      -> !llvm.struct<(i64, ptr)> {
    %storage = mulberry.list.alloc %length : !mulberry.list_storage<i64>
    %value = arith.constant 42 : i64
    mulberry.list.store %value, %storage[%index]
        : i64, !mulberry.list_storage<i64>
    %escaped = mulberry.list.escape_storage %storage, %length
        : !mulberry.list_storage<i64> -> !mulberry.list_storage<i64>
    %desc = mulberry.list.desc_pack %length, %escaped
        : !mulberry.list_storage<i64> -> !mulberry.list_desc<i64>
    %abi = mulberry.list.desc_to_abi %desc
        : !mulberry.list_desc<i64> -> !llvm.struct<(i64, ptr)>
    return %abi : !llvm.struct<(i64, ptr)>
  }
}

// CHECK: llvm.func @mulberry_boehm_malloc(i64) -> !llvm.ptr
// CHECK-LABEL: func.func @escape_scalar_storage
// CHECK: memref.alloc
// CHECK: llvm.call @mulberry_boehm_malloc
// CHECK: scf.for
// CHECK: memref.load
// CHECK: llvm.store
// CHECK: llvm.insertvalue
// CHECK: return
// CHECK-NOT: mulberry.list.escape_storage
// CHECK-NOT: !mulberry.list_desc
