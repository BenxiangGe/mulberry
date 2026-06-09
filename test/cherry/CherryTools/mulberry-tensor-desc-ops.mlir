// RUN: cherry-opt %s | FileCheck %s

module {
  func.func @tensor_desc_ops() -> !mulberry.tensor<?x3xf32> {
    %tensor = mulberry.tensor.alloc() : !mulberry.tensor<2x3xf32>
    %desc = mulberry.tensor.desc_pack %tensor
        : !mulberry.tensor<2x3xf32> -> !mulberry.tensor_desc<2x3xf32>
    %unpacked = mulberry.tensor.desc_unpack %desc
        : !mulberry.tensor_desc<2x3xf32> -> !mulberry.tensor<?x3xf32>
    %abi = mulberry.tensor.desc_to_abi %desc
        : !mulberry.tensor_desc<2x3xf32>
            -> !llvm.struct<(!ptr.ptr<#llvm.address_space<0>>, array<2 x i64>, array<2 x i64>)>
    return %unpacked : !mulberry.tensor<?x3xf32>
  }
}

// CHECK-LABEL: func.func @tensor_desc_ops
// CHECK: mulberry.tensor.desc_pack
// CHECK-SAME: !mulberry.tensor<2x3xf32> -> !mulberry.tensor_desc<2x3xf32>
// CHECK: mulberry.tensor.desc_unpack
// CHECK-SAME: !mulberry.tensor_desc<2x3xf32> -> !mulberry.tensor<?x3xf32>
// CHECK: mulberry.tensor.desc_to_abi
// CHECK-SAME: !mulberry.tensor_desc<2x3xf32> -> !llvm.struct<(!ptr.ptr<#llvm.address_space<0>>, array<2 x i64>, array<2 x i64>)>
