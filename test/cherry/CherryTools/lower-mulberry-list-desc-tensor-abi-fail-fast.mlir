// RUN: not cherry-opt --lower-mulberry %s 2>&1 | FileCheck %s

// List<Tensor> must not expose its local memref-handle storage as the function
// ABI. The real ABI needs data to point at Tensor descriptor storage.
module {
  func.func @tensor_list_desc_to_abi(%length: index) -> !llvm.struct<(i64, ptr)> {
    %storage = mulberry.list.alloc %length
        : !mulberry.list_storage<memref<2xf32>>
    %desc = mulberry.list.desc_pack %length, %storage
        : !mulberry.list_storage<memref<2xf32>>
            -> !mulberry.list_desc<memref<2xf32>>
    %abi = mulberry.list.desc_to_abi %desc
        : !mulberry.list_desc<memref<2xf32>> -> !llvm.struct<(i64, ptr)>
    return %abi : !llvm.struct<(i64, ptr)>
  }
}

// CHECK: failed to legalize operation 'mulberry.list.desc_pack'
