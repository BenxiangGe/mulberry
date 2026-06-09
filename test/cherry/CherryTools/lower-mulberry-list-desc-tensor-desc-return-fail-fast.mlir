// RUN: not cherry-opt --lower-mulberry %s 2>&1 | FileCheck %s

module {
  func.func @return_tensor_desc_list_desc(%length: index, %n: index, %m: index)
      -> !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>> {
    %tensor = mulberry.tensor.alloc(%n, %m)
        : !mulberry.tensor<?x?xf32>
    %tensorDesc = mulberry.tensor.desc_pack %tensor
        : !mulberry.tensor<?x?xf32> -> !mulberry.tensor_desc<?x?xf32>
    %storage = mulberry.list.alloc %length
        : !mulberry.list_storage<!mulberry.tensor_desc<?x?xf32>>
    %index = arith.constant 0 : index
    mulberry.list.store %tensorDesc, %storage[%index]
        : !mulberry.tensor_desc<?x?xf32>,
          !mulberry.list_storage<!mulberry.tensor_desc<?x?xf32>>
    %desc = mulberry.list.desc_pack %length, %storage
        : !mulberry.list_storage<!mulberry.tensor_desc<?x?xf32>>
            -> !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
    return %desc : !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
  }
}

// CHECK: failed to legalize operation 'func.return'
