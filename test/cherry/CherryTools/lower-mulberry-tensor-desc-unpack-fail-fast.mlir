// RUN: not cherry-opt --lower-mulberry %s 2>&1 | FileCheck %s

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

// CHECK: failed to legalize operation 'mulberry.tensor.desc_unpack'
// CHECK: Tensor descriptor unpack lowering needs explicit Tensor ABI reconstruction support
