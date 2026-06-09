// RUN: cherry-opt --lower-mulberry %s | FileCheck %s

module {
  func.func @escape_tensor_desc_storage(%length: index, %index: index,
                                        %n: index, %m: index)
      -> !llvm.struct<(!ptr.ptr<#llvm.address_space<0>>, array<2 x i64>, array<2 x i64>)> {
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
    %loaded = mulberry.list.load %escaped[%index]
        : !mulberry.list_storage<!mulberry.tensor_desc<?x?xf32>>
            -> !mulberry.tensor_desc<?x?xf32>
    %abi = mulberry.tensor.desc_to_abi %loaded
        : !mulberry.tensor_desc<?x?xf32>
            -> !llvm.struct<(!ptr.ptr<#llvm.address_space<0>>, array<2 x i64>, array<2 x i64>)>
    return %abi : !llvm.struct<(!ptr.ptr<#llvm.address_space<0>>, array<2 x i64>, array<2 x i64>)>
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
// CHECK: %[[LOAD_PTR:.*]] = llvm.getelementptr %[[HEAP]]
// CHECK: %[[LOADED:.*]] = llvm.load %[[LOAD_PTR]]
// CHECK: return %[[LOADED]] : !llvm.struct<(!ptr.ptr<#llvm.address_space<0>>, array<2 x i64>, array<2 x i64>)>
// CHECK-NOT: mulberry.list.escape_storage
// CHECK-NOT: !mulberry.tensor_desc
