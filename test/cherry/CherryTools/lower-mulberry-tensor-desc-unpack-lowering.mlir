// RUN: cherry-opt --lower-mulberry %s | FileCheck %s

module {
  func.func @local_tensor_desc_unpack(%n: index, %m: index) -> index {
    %tensor = mulberry.tensor.alloc(%n, %m)
        : !mulberry.tensor<?x?xf32>
    %desc = mulberry.tensor.desc_pack %tensor
        : !mulberry.tensor<?x?xf32> -> !mulberry.tensor_desc<?x?xf32>
    %unpacked = mulberry.tensor.desc_unpack %desc
        : !mulberry.tensor_desc<?x?xf32> -> !mulberry.tensor<?x?xf32>
    %zero = arith.constant 0 : index
    %dim = mulberry.tensor.dim %unpacked, %zero
        : !mulberry.tensor<?x?xf32>
    return %dim : index
  }
}

// CHECK-LABEL: func.func @local_tensor_desc_unpack
// CHECK: ptr.to_ptr
// CHECK: %[[DATA:.*]] = llvm.extractvalue
// CHECK-SAME: [0]
// CHECK: ptr.from_ptr %[[DATA]]
// CHECK-SAME: -> memref<f32, #llvm.address_space<0>>
// CHECK: memref.reinterpret_cast
// CHECK: memref.memory_space_cast
// CHECK: memref.dim
// CHECK: return
// CHECK-NOT: mulberry.tensor.desc_unpack
