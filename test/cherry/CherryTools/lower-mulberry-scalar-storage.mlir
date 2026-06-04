// RUN: cherry-opt --lower-mulberry %s | FileCheck %s

module {
  func.func @main() -> i64 {
    %ptr = mulberry.alloca i64 : !mulberry.ptr<i64>
    %value = arith.constant 42 : i64
    mulberry.store %value, %ptr : i64, !mulberry.ptr<i64>
    %loaded = mulberry.load %ptr : !mulberry.ptr<i64> -> i64
    return %loaded : i64
  }
}

// CHECK-LABEL: func.func @main
// CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[PTR:.*]] = llvm.alloca %[[ONE]] x i64
// CHECK: arith.constant 42 : i64
// CHECK: llvm.store {{.*}}, %[[PTR]] : i64, !llvm.ptr
// CHECK: %[[LOADED:.*]] = llvm.load %[[PTR]] : !llvm.ptr -> i64
// CHECK: return %[[LOADED]] : i64
// CHECK-NOT: mulberry.
