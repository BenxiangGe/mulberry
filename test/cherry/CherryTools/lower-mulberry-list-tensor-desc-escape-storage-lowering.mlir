// RUN: cherry-opt --lower-mulberry %s | FileCheck %s

module {
  func.func @escape_tensor_desc_storage(%length: index, %index: index,
                                        %n: index, %m: index)
      -> !llvm.struct<(i64, ptr)> {
    %tensor = mulberry.tensor.alloc(%n, %m)
        : !mulberry.tensor<?x?xf32>
    %tensorDesc = mulberry.tensor.desc_pack %tensor
        : !mulberry.tensor<?x?xf32> -> !mulberry.tensor_desc<?x?xf32>
    %storage = mulberry.list.alloc %length
        : !mulberry.list_storage<!mulberry.tensor_desc<?x?xf32>>
    mulberry.list.store %tensorDesc, %storage[%index]
        : !mulberry.tensor_desc<?x?xf32>,
          !mulberry.list_storage<!mulberry.tensor_desc<?x?xf32>>
    %escaped = mulberry.list.escape_storage %storage, %length
        : !mulberry.list_storage<!mulberry.tensor_desc<?x?xf32>>
            -> !mulberry.list_storage<!mulberry.tensor_desc<?x?xf32>>
    %desc = mulberry.list.desc_pack %length, %escaped
        : !mulberry.list_storage<!mulberry.tensor_desc<?x?xf32>>
            -> !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
    %abi = mulberry.list.desc_to_abi %desc
        : !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
            -> !llvm.struct<(i64, ptr)>
    return %abi : !llvm.struct<(i64, ptr)>
  }
}

// CHECK: llvm.func @malloc(i64) -> !llvm.ptr
// CHECK-LABEL: func.func @escape_tensor_desc_storage
// CHECK: %[[LOCAL:.*]] = llvm.alloca %{{.*}} x !llvm.struct<(!ptr.ptr<#llvm.address_space<0>>, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK: %[[SIZE_PTR:.*]] = llvm.getelementptr %[[NULL]][1]
// CHECK-SAME: -> !llvm.ptr, !llvm.struct<(!ptr.ptr<#llvm.address_space<0>>, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[ELEM_BYTES:.*]] = llvm.ptrtoint %[[SIZE_PTR]] : !llvm.ptr to i64
// CHECK: %[[TOTAL_BYTES:.*]] = arith.muli %{{.*}}, %[[ELEM_BYTES]] : i64
// CHECK: %[[HEAP:.*]] = llvm.call @malloc(%[[TOTAL_BYTES]]) : (i64) -> !llvm.ptr
// CHECK: scf.for
// CHECK: llvm.load
// CHECK: llvm.store
// CHECK: llvm.insertvalue
// CHECK: return
// CHECK-NOT: mulberry.list.escape_storage
// CHECK-NOT: !mulberry.tensor_desc
