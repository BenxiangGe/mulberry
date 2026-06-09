// RUN: cherry-opt --lower-mulberry %s | FileCheck %s

module {
  func.func @first(
      %desc: !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>)
      -> !llvm.struct<(ptr, array<2 x i64>, array<2 x i64>)> {
    %index = arith.constant 0 : index
    %data = mulberry.list.desc_data %desc
        : !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
            -> !mulberry.list_storage<!mulberry.tensor_desc<?x?xf32>>
    %loaded = mulberry.list.load %data[%index]
        : !mulberry.list_storage<!mulberry.tensor_desc<?x?xf32>>
            -> !mulberry.tensor_desc<?x?xf32>
    %abi = mulberry.tensor.desc_to_abi %loaded
        : !mulberry.tensor_desc<?x?xf32>
            -> !llvm.struct<(ptr, array<2 x i64>, array<2 x i64>)>
    return %abi : !llvm.struct<(ptr, array<2 x i64>, array<2 x i64>)>
  }

  func.func @call_first(%n: index, %m: index)
      -> !llvm.struct<(ptr, array<2 x i64>, array<2 x i64>)> {
    %a = mulberry.tensor.alloc(%n, %m) : !mulberry.tensor<?x?xf32>
    %b = mulberry.tensor.alloc(%n, %m) : !mulberry.tensor<?x?xf32>
    %list = mulberry.list.create(%a, %b)
        : (!mulberry.tensor<?x?xf32>, !mulberry.tensor<?x?xf32>)
            -> !mulberry.list<!mulberry.tensor<?x?xf32>>
    %desc = mulberry.list.to_desc %list
        : !mulberry.list<!mulberry.tensor<?x?xf32>>
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
// CHECK: llvm.load

// CHECK-LABEL: func.func @call_first
// CHECK-SAME: -> !llvm.struct<(ptr, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[LENGTH:.*]] = arith.constant 2 : index
// CHECK: %[[STORAGE:.*]] = llvm.alloca %{{.*}} x !llvm.struct<(ptr, array<2 x i64>, array<2 x i64>)>
// CHECK: llvm.store %{{.*}}, %{{.*}}
// CHECK: llvm.store %{{.*}}, %{{.*}}
// CHECK: %[[LIST_DESC:.*]] = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
// CHECK: %[[WITH_LENGTH:.*]] = llvm.insertvalue %{{.*}}, %[[LIST_DESC]][0]
// CHECK: %[[ABI_DESC:.*]] = llvm.insertvalue %[[STORAGE]], %[[WITH_LENGTH]][1]
// CHECK: call @first(%[[ABI_DESC]])
// CHECK-NOT: mulberry.list.to_desc
// CHECK-NOT: !mulberry.list_desc
// CHECK-NOT: !mulberry.list_storage
