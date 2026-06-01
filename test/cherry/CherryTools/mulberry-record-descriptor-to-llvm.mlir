// RUN: cherry-opt --convert-cherry-to-llvm %s | FileCheck %s

module {
  func.func @tensor_descriptor() -> i64 {
    %descriptor = mulberry.alloca !mulberry.record<TensorDescriptor {data: !mulberry.ptr<i64>, shape: !mulberry.record<TensorShape2D {rows: i64, cols: i64}>}> : !mulberry.ptr<!mulberry.record<TensorDescriptor {data: !mulberry.ptr<i64>, shape: !mulberry.record<TensorShape2D {rows: i64, cols: i64}>}>>
    %data = mulberry.record.get_field %descriptor["data"] : !mulberry.ptr<!mulberry.record<TensorDescriptor {data: !mulberry.ptr<i64>, shape: !mulberry.record<TensorShape2D {rows: i64, cols: i64}>}>> -> !mulberry.ptr<!mulberry.ptr<i64>>
    %shape = mulberry.record.get_field %descriptor["shape"] : !mulberry.ptr<!mulberry.record<TensorDescriptor {data: !mulberry.ptr<i64>, shape: !mulberry.record<TensorShape2D {rows: i64, cols: i64}>}>> -> !mulberry.ptr<!mulberry.record<TensorShape2D {rows: i64, cols: i64}>>
    %rows = mulberry.record.get_field %shape["rows"] : !mulberry.ptr<!mulberry.record<TensorShape2D {rows: i64, cols: i64}>> -> !mulberry.ptr<i64>
    %value = arith.constant 0 : i64
    return %value : i64
  }
}

// CHECK-LABEL: llvm.func @tensor_descriptor()
// CHECK: llvm.alloca {{.*}} x !llvm.struct<(ptr, struct<(i64, i64)>)>
// CHECK: llvm.getelementptr {{.*}}[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, struct<(i64, i64)>)>
// CHECK: llvm.getelementptr {{.*}}[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, struct<(i64, i64)>)>
// CHECK: llvm.getelementptr {{.*}}[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i64)>
// CHECK: llvm.return
// CHECK-NOT: mulberry.
