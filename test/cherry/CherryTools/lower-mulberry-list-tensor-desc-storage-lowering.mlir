// RUN: cherry-opt --lower-mulberry %s | FileCheck %s

module {
  func.func @tensor_desc_list_storage(%length: index, %index: index,
                                      %n: index, %m: index)
      -> !llvm.struct<(ptr, array<2 x i64>, array<2 x i64>)> {
    %tensor = mulberry.tensor.alloc(%n, %m)
        : !mulberry.tensor<?x?xf32>
    %desc = mulberry.tensor.desc_pack %tensor
        : !mulberry.tensor<?x?xf32> -> !mulberry.tensor_desc<?x?xf32>
    %storage = mulberry.list.alloc %length
        : !mulberry.list_storage<!mulberry.tensor_desc<?x?xf32>>
    mulberry.list.store %desc, %storage[%index]
        : !mulberry.tensor_desc<?x?xf32>,
          !mulberry.list_storage<!mulberry.tensor_desc<?x?xf32>>
    %loaded = mulberry.list.load %storage[%index]
        : !mulberry.list_storage<!mulberry.tensor_desc<?x?xf32>>
            -> !mulberry.tensor_desc<?x?xf32>
    %abi = mulberry.tensor.desc_to_abi %loaded
        : !mulberry.tensor_desc<?x?xf32>
            -> !llvm.struct<(ptr, array<2 x i64>, array<2 x i64>)>
    return %abi : !llvm.struct<(ptr, array<2 x i64>, array<2 x i64>)>
  }
}

// CHECK-LABEL: func.func @tensor_desc_list_storage
// CHECK-SAME: -> !llvm.struct<(ptr, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[STORAGE:.*]] = llvm.alloca %{{.*}} x !llvm.struct<(ptr, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[STORE_PTR:.*]] = llvm.getelementptr %[[STORAGE]]
// CHECK-SAME: -> !llvm.ptr, !llvm.struct<(ptr, array<2 x i64>, array<2 x i64>)>
// CHECK: llvm.store %{{.*}}, %[[STORE_PTR]]
// CHECK: %[[LOAD_PTR:.*]] = llvm.getelementptr %[[STORAGE]]
// CHECK-SAME: -> !llvm.ptr, !llvm.struct<(ptr, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[LOADED:.*]] = llvm.load %[[LOAD_PTR]]
// CHECK: return %[[LOADED]] : !llvm.struct<(ptr, array<2 x i64>, array<2 x i64>)>
// CHECK-NOT: !mulberry.tensor_desc
// CHECK-NOT: !mulberry.list_storage
