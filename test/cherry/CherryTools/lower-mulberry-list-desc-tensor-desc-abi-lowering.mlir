// RUN: cherry-opt --lower-mulberry %s | FileCheck %s

module {
  func.func @tensor_desc_list_desc_to_abi(%length: index, %n: index, %m: index)
      -> !llvm.struct<(i64, ptr)> {
    %tensor = mulberry.tensor.alloc(%n, %m)
        : !mulberry.tensor<?x?xf32>
    %tensorDesc = mulberry.tensor.desc_pack %tensor
        : !mulberry.tensor<?x?xf32> -> !mulberry.tensor_desc<?x?xf32>
    %storage = mulberry.list.alloc %length
        : !mulberry.list_storage<!mulberry.tensor_desc<?x?xf32>>
    %index = arith.constant 0 : index
    mulberry.list.store %tensorDesc, %storage[%index]
        : !mulberry.tensor_desc<?x?xf32>,
          !mulberry.list_storage<!mulberry.tensor_desc<?x?xf32>>
    %listDesc = mulberry.list.desc_pack %length, %storage
        : !mulberry.list_storage<!mulberry.tensor_desc<?x?xf32>>
            -> !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
    %abi = mulberry.list.desc_to_abi %listDesc
        : !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
            -> !llvm.struct<(i64, ptr)>
    return %abi : !llvm.struct<(i64, ptr)>
  }
}

// CHECK-LABEL: func.func @tensor_desc_list_desc_to_abi
// CHECK-SAME: -> !llvm.struct<(i64, ptr)>
// CHECK: %[[STORAGE:.*]] = llvm.alloca %{{.*}} x !llvm.struct<(ptr, array<2 x i64>, array<2 x i64>)>
// CHECK: llvm.store %{{.*}}, %{{.*}}
// CHECK: %[[LIST_DESC:.*]] = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
// CHECK: %[[WITH_LENGTH:.*]] = llvm.insertvalue %{{.*}}, %[[LIST_DESC]][0]
// CHECK: %[[ABI:.*]] = llvm.insertvalue %[[STORAGE]], %[[WITH_LENGTH]][1]
// CHECK: return %[[ABI]] : !llvm.struct<(i64, ptr)>
// CHECK-NOT: memref.extract_aligned_pointer_as_index %[[STORAGE]]
// CHECK-NOT: !mulberry.tensor_desc
// CHECK-NOT: !mulberry.list_desc
