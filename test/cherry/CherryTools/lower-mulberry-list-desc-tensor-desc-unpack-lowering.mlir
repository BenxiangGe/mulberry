// RUN: cherry-opt --lower-mulberry %s | FileCheck %s

module {
  func.func @use_tensor_desc_list_desc(
      %desc: !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>)
      -> !mulberry.tensor<?x?xf32> {
    %index = arith.constant 0 : index
    %loaded = mulberry.list.desc_get %desc[%index]
        : !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
            -> !mulberry.tensor_desc<?x?xf32>
    %tensor = mulberry.tensor.desc_unpack %loaded
        : !mulberry.tensor_desc<?x?xf32> -> !mulberry.tensor<?x?xf32>
    return %tensor : !mulberry.tensor<?x?xf32>
  }
}

// CHECK-LABEL: func.func @use_tensor_desc_list_desc
// CHECK-SAME: (%[[DESC:.*]]: !llvm.struct<(i64, ptr)>)
// CHECK: %[[DATA_LIST:.*]] = llvm.extractvalue %[[DESC]][1]
// CHECK: %[[LOAD_PTR:.*]] = llvm.getelementptr %[[DATA_LIST]]
// CHECK: %[[TENSOR_DESC:.*]] = llvm.load %[[LOAD_PTR]]
// CHECK: %[[DATA:.*]] = llvm.extractvalue %[[TENSOR_DESC]][0]
// CHECK: %[[MEMREF_DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
// CHECK: llvm.insertvalue %[[DATA]], %[[MEMREF_DESC]][0]
// CHECK: builtin.unrealized_conversion_cast
// CHECK-SAME: to memref<f32, #llvm.address_space<0>>
// CHECK: memref.reinterpret_cast
// CHECK: memref.memory_space_cast
// CHECK: return
// CHECK-NOT: mulberry.tensor.desc_unpack
