// RUN: not cherry-opt --lower-mulberry %s 2>&1 | FileCheck %s

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

// CHECK: failed to legalize operation 'mulberry.tensor.desc_unpack'
// CHECK: Tensor descriptor unpack lowering needs explicit Tensor ABI reconstruction support
