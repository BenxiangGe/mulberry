// RUN: cherry-opt --prepare-mulberry-boundaries --lower-mulberry %s | FileCheck %s

module {
  func.func @call_tensor_list_result(%n: index, %m: index) -> i64 {
    %list = call @make_tensor_list(%n, %m)
        : (index, index) -> !mulberry.list<!mulberry.tensor<?x?xf32>>
    %size = mulberry.list.size %list
        : !mulberry.list<!mulberry.tensor<?x?xf32>>
    return %size : i64
  }

  func.func private @make_tensor_list(%n: index, %m: index)
      -> !mulberry.list<!mulberry.tensor<?x?xf32>> {
    %tensor = mulberry.tensor.alloc(%n, %m) : !mulberry.tensor<?x?xf32>
    %list = mulberry.list.create(%tensor)
        : (!mulberry.tensor<?x?xf32>)
            -> !mulberry.list<!mulberry.tensor<?x?xf32>>
    return %list : !mulberry.list<!mulberry.tensor<?x?xf32>>
  }
}

// CHECK: llvm.func @malloc(i64) -> !llvm.ptr

// CHECK-LABEL: func.func @call_tensor_list_result
// CHECK: %[[DESC:.*]] = call @make_tensor_list
// CHECK-SAME: (index, index) -> !llvm.struct<(i64, ptr)>
// CHECK: %[[LENGTH_I64:.*]] = llvm.extractvalue %[[DESC]][0]
// CHECK-SAME: !llvm.struct<(i64, ptr)>
// CHECK: %[[LENGTH:.*]] = arith.index_cast %[[LENGTH_I64]] : i64 to index
// CHECK: %[[SIZE:.*]] = arith.index_cast %[[LENGTH]] : index to i64
// CHECK: return %[[SIZE]] : i64

// CHECK-LABEL: func.func private @make_tensor_list
// CHECK-SAME: -> !llvm.struct<(i64, ptr)>
// CHECK: %[[LOCAL:.*]] = llvm.alloca %{{.*}} x !llvm.struct<(!ptr.ptr<#llvm.address_space<0>>, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[HEAP:.*]] = llvm.call @malloc
// CHECK-SAME: (i64) -> !llvm.ptr
// CHECK: scf.for
// CHECK: llvm.load
// CHECK: llvm.store
// CHECK: %[[DESC_UNDEF:.*]] = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
// CHECK: %[[WITH_LENGTH:.*]] = llvm.insertvalue %{{.*}}, %[[DESC_UNDEF]][0]
// CHECK: %[[DESC:.*]] = llvm.insertvalue %[[HEAP]], %[[WITH_LENGTH]][1]
// CHECK: return %[[DESC]] : !llvm.struct<(i64, ptr)>
