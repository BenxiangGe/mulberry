// RUN: cherry-opt --lower-mulberry %s | FileCheck %s

module {
  func.func @first(
      %desc: !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>)
      -> !llvm.struct<(ptr, array<2 x i64>, array<2 x i64>)> {
    %index = arith.constant 0 : index
    %loaded = mulberry.list.desc_get %desc[%index]
        : !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
            -> !mulberry.tensor_desc<?x?xf32>
    %abi = mulberry.tensor.desc_to_abi %loaded
        : !mulberry.tensor_desc<?x?xf32>
            -> !llvm.struct<(ptr, array<2 x i64>, array<2 x i64>)>
    return %abi : !llvm.struct<(ptr, array<2 x i64>, array<2 x i64>)>
  }

  func.func @call_first(%length: index, %n: index, %m: index)
      -> !llvm.struct<(ptr, array<2 x i64>, array<2 x i64>)> {
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
    %result = call @first(%desc)
        : (!mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>)
            -> !llvm.struct<(ptr, array<2 x i64>, array<2 x i64>)>
    return %result : !llvm.struct<(ptr, array<2 x i64>, array<2 x i64>)>
  }
}

// CHECK-LABEL: func.func @first
// CHECK-SAME: (%[[DESC:.*]]: !llvm.struct<(i64, ptr)>)
// CHECK: %[[DATA:.*]] = llvm.extractvalue %[[DESC]][1] : !llvm.struct<(i64, ptr)>
// CHECK: %[[LOAD_PTR:.*]] = llvm.getelementptr %[[DATA]]
// CHECK-SAME: -> !llvm.ptr, !llvm.struct<(ptr, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[LOADED:.*]] = llvm.load %[[LOAD_PTR]]
// CHECK: return %[[LOADED]] : !llvm.struct<(ptr, array<2 x i64>, array<2 x i64>)>

// CHECK-LABEL: func.func @call_first
// CHECK-SAME: -> !llvm.struct<(ptr, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[STORAGE:.*]] = llvm.alloca %{{.*}} x !llvm.struct<(ptr, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[LIST_DESC:.*]] = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
// CHECK: %[[WITH_LENGTH:.*]] = llvm.insertvalue %{{.*}}, %[[LIST_DESC]][0]
// CHECK: %[[ABI_DESC:.*]] = llvm.insertvalue %[[STORAGE]], %[[WITH_LENGTH]][1]
// CHECK: %[[RESULT:.*]] = call @first(%[[ABI_DESC]])
// CHECK-SAME: (!llvm.struct<(i64, ptr)>) -> !llvm.struct<(ptr, array<2 x i64>, array<2 x i64>)>
// CHECK: return %[[RESULT]] : !llvm.struct<(ptr, array<2 x i64>, array<2 x i64>)>
// CHECK-NOT: !mulberry.tensor_desc
// CHECK-NOT: !mulberry.list_desc
// CHECK-NOT: !mulberry.list_storage
