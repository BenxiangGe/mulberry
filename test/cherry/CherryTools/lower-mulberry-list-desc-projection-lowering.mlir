// RUN: cherry-opt --lower-mulberry %s | FileCheck %s

module {
  func.func @tensor_desc_list_desc_projection(%length: index, %index: index,
                                              %n: index, %m: index)
      -> !llvm.struct<(ptr, array<2 x i64>, array<2 x i64>)> {
    %tensor = mulberry.tensor.alloc(%n, %m)
        : !mulberry.tensor<?x?xf32>
    %tensorDesc = mulberry.tensor.desc_pack %tensor
        : !mulberry.tensor<?x?xf32> -> !mulberry.tensor_desc<?x?xf32>
    %storage = mulberry.list.alloc %length
        : !mulberry.list_storage<!mulberry.tensor_desc<?x?xf32>>
    mulberry.list.store %tensorDesc, %storage[%index]
        : !mulberry.tensor_desc<?x?xf32>,
          !mulberry.list_storage<!mulberry.tensor_desc<?x?xf32>>
    %listDesc = mulberry.list.desc_pack %length, %storage
        : !mulberry.list_storage<!mulberry.tensor_desc<?x?xf32>>
            -> !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
    %size = mulberry.list.desc_length %listDesc
        : !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
    %data = mulberry.list.desc_data %listDesc
        : !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
            -> !mulberry.list_storage<!mulberry.tensor_desc<?x?xf32>>
    %loaded = mulberry.list.load %data[%size]
        : !mulberry.list_storage<!mulberry.tensor_desc<?x?xf32>>
            -> !mulberry.tensor_desc<?x?xf32>
    %abi = mulberry.tensor.desc_to_abi %loaded
        : !mulberry.tensor_desc<?x?xf32>
            -> !llvm.struct<(ptr, array<2 x i64>, array<2 x i64>)>
    return %abi : !llvm.struct<(ptr, array<2 x i64>, array<2 x i64>)>
  }
}

// CHECK-LABEL: func.func @tensor_desc_list_desc_projection
// CHECK-SAME: -> !llvm.struct<(ptr, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[LIST_DESC:.*]] = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
// CHECK: %[[WITH_LENGTH:.*]] = llvm.insertvalue %{{.*}}, %[[LIST_DESC]][0]
// CHECK: %[[ABI:.*]] = llvm.insertvalue %{{.*}}, %[[WITH_LENGTH]][1]
// CHECK: %[[LENGTH_I64:.*]] = llvm.extractvalue %[[ABI]][0] : !llvm.struct<(i64, ptr)>
// CHECK: %[[LENGTH:.*]] = arith.index_cast %[[LENGTH_I64]] : i64 to index
// CHECK: %[[DATA:.*]] = llvm.extractvalue %[[ABI]][1] : !llvm.struct<(i64, ptr)>
// CHECK: %[[INDEX:.*]] = arith.index_cast %[[LENGTH]] : index to i64
// CHECK: %[[LOAD_PTR:.*]] = llvm.getelementptr %[[DATA]][%[[INDEX]]]
// CHECK-SAME: -> !llvm.ptr, !llvm.struct<(ptr, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[LOADED:.*]] = llvm.load %[[LOAD_PTR]]
// CHECK-SAME: -> !llvm.struct<(ptr, array<2 x i64>, array<2 x i64>)>
// CHECK: return %[[LOADED]] : !llvm.struct<(ptr, array<2 x i64>, array<2 x i64>)>
// CHECK-NOT: !mulberry.tensor_desc
// CHECK-NOT: !mulberry.list_desc
// CHECK-NOT: !mulberry.list_storage
