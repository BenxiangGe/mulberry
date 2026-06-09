// RUN: cherry-opt --lower-mulberry %s | FileCheck %s

module {
  func.func @list_desc_to_abi(%length: index) -> !llvm.struct<(i64, ptr)> {
    %storage = mulberry.list.alloc %length : !mulberry.list_storage<i64>
    %desc = mulberry.list.desc_pack %length, %storage
        : !mulberry.list_storage<i64> -> !mulberry.list_desc<i64>
    %abi = mulberry.list.desc_to_abi %desc
        : !mulberry.list_desc<i64> -> !llvm.struct<(i64, ptr)>
    return %abi : !llvm.struct<(i64, ptr)>
  }
}

// CHECK-LABEL: func.func @list_desc_to_abi
// CHECK: %[[STORAGE:.*]] = memref.alloc(%{{.*}}) : memref<?xi64>
// CHECK: %[[PTR_INDEX:.*]] = memref.extract_aligned_pointer_as_index %[[STORAGE]]
// CHECK: %[[PTR_INT:.*]] = arith.index_cast %[[PTR_INDEX]] : index to i64
// CHECK: %[[DATA:.*]] = llvm.inttoptr %[[PTR_INT]] : i64 to !llvm.ptr
// CHECK: %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
// CHECK: %[[WITH_LENGTH:.*]] = llvm.insertvalue %{{.*}}, %[[UNDEF]][0]
// CHECK: %[[ABI:.*]] = llvm.insertvalue %[[DATA]], %[[WITH_LENGTH]][1]
// CHECK: return %[[ABI]] : !llvm.struct<(i64, ptr)>
// CHECK-NOT: mulberry.list.desc_pack
// CHECK-NOT: mulberry.list.desc_to_abi
