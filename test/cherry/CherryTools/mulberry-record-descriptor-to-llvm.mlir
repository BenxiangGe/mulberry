// RUN: cherry-opt --convert-cherry-to-llvm %s | FileCheck %s

module {
  func.func @tensor_descriptor() -> i64 {
    %descriptor = mulberry.alloca !mulberry.record<TensorDescriptor {allocated: !mulberry.ptr<i64>, aligned: !mulberry.ptr<i64>, offset: i64, sizes: !mulberry.record<TensorSizes2D {rows: i64, cols: i64}>, strides: !mulberry.record<TensorStrides2D {rows: i64, cols: i64}>}> : !mulberry.ptr<!mulberry.record<TensorDescriptor {allocated: !mulberry.ptr<i64>, aligned: !mulberry.ptr<i64>, offset: i64, sizes: !mulberry.record<TensorSizes2D {rows: i64, cols: i64}>, strides: !mulberry.record<TensorStrides2D {rows: i64, cols: i64}>}>>
    %allocated = mulberry.record.get_field %descriptor["allocated"] : !mulberry.ptr<!mulberry.record<TensorDescriptor {allocated: !mulberry.ptr<i64>, aligned: !mulberry.ptr<i64>, offset: i64, sizes: !mulberry.record<TensorSizes2D {rows: i64, cols: i64}>, strides: !mulberry.record<TensorStrides2D {rows: i64, cols: i64}>}>> -> !mulberry.ptr<!mulberry.ptr<i64>>
    %sizes = mulberry.record.get_field %descriptor["sizes"] : !mulberry.ptr<!mulberry.record<TensorDescriptor {allocated: !mulberry.ptr<i64>, aligned: !mulberry.ptr<i64>, offset: i64, sizes: !mulberry.record<TensorSizes2D {rows: i64, cols: i64}>, strides: !mulberry.record<TensorStrides2D {rows: i64, cols: i64}>}>> -> !mulberry.ptr<!mulberry.record<TensorSizes2D {rows: i64, cols: i64}>>
    %rows = mulberry.record.get_field %sizes["rows"] : !mulberry.ptr<!mulberry.record<TensorSizes2D {rows: i64, cols: i64}>> -> !mulberry.ptr<i64>
    %value = arith.constant 0 : i64
    return %value : i64
  }
}

// CHECK-LABEL: llvm.func @tensor_descriptor()
// CHECK: llvm.alloca {{.*}} x !llvm.struct<(ptr, ptr, i64, struct<(i64, i64)>, struct<(i64, i64)>)>
// CHECK: llvm.getelementptr {{.*}}[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, struct<(i64, i64)>, struct<(i64, i64)>)>
// CHECK: llvm.getelementptr {{.*}}[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, struct<(i64, i64)>, struct<(i64, i64)>)>
// CHECK: llvm.getelementptr {{.*}}[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i64)>
// CHECK: llvm.return
// CHECK-NOT: mulberry.
