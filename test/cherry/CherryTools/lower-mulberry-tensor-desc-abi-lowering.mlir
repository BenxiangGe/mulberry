// RUN: cherry-opt --lower-mulberry %s | FileCheck %s

module {
  func.func @tensor_desc_to_abi(%n: index, %m: index)
      -> !llvm.struct<(!ptr.ptr<#llvm.address_space<0>>, array<2 x i64>, array<2 x i64>)> {
    %tensor = mulberry.tensor.alloc(%n, %m) : !mulberry.tensor<?x?xf32>
    %desc = mulberry.tensor.desc_pack %tensor
        : !mulberry.tensor<?x?xf32> -> !mulberry.tensor_desc<?x?xf32>
    %abi = mulberry.tensor.desc_to_abi %desc
        : !mulberry.tensor_desc<?x?xf32>
            -> !llvm.struct<(!ptr.ptr<#llvm.address_space<0>>, array<2 x i64>, array<2 x i64>)>
    return %abi : !llvm.struct<(!ptr.ptr<#llvm.address_space<0>>, array<2 x i64>, array<2 x i64>)>
  }
}

// CHECK-LABEL: func.func @tensor_desc_to_abi
// CHECK: %[[TENSOR:.*]] = memref.alloc
// CHECK: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(!ptr.ptr<#llvm.address_space<0>>, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[DATA_MEMREF:.*]] = memref.memory_space_cast %[[TENSOR]]
// CHECK-SAME: memref<?x?xf32> to memref<?x?xf32, #llvm.address_space<0>>
// CHECK: %[[DATA:.*]] = ptr.to_ptr %[[DATA_MEMREF]]
// CHECK-SAME: memref<?x?xf32, #llvm.address_space<0>> -> <#llvm.address_space<0>>
// CHECK: %[[WITH_DATA:.*]] = llvm.insertvalue %[[DATA]], %[[DESC]][0]
// CHECK: %[[DIM0:.*]] = memref.dim %[[TENSOR]]
// CHECK: %[[SIZE0:.*]] = arith.index_cast %[[DIM0]] : index to i64
// CHECK: %[[DIM1:.*]] = memref.dim %[[TENSOR]]
// CHECK: %[[SIZE1:.*]] = arith.index_cast %[[DIM1]] : index to i64
// CHECK: %[[WITH_SIZE1:.*]] = llvm.insertvalue %[[SIZE1]], %[[WITH_DATA]][1, 1]
// CHECK: %[[WITH_STRIDE1:.*]] = llvm.insertvalue {{.*}}, %[[WITH_SIZE1]][2, 1]
// CHECK: %[[STRIDE0:.*]] = arith.muli {{.*}}, %[[SIZE1]] : i64
// CHECK: %[[WITH_SIZE0:.*]] = llvm.insertvalue %[[SIZE0]], %[[WITH_STRIDE1]][1, 0]
// CHECK: %[[ABI:.*]] = llvm.insertvalue %[[STRIDE0]], %[[WITH_SIZE0]][2, 0]
// CHECK: return %[[ABI]] : !llvm.struct<(!ptr.ptr<#llvm.address_space<0>>, array<2 x i64>, array<2 x i64>)>
// CHECK-NOT: mulberry.tensor.desc_pack
// CHECK-NOT: mulberry.tensor.desc_to_abi
