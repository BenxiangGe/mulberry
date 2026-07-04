// RUN: mulberry-opt --lower-mulberry %s | FileCheck %s

module {
  func.func @main() -> i64 {
    %ptr = mulberry_core.alloca i64 : !mulberry_core.ptr<i64>
    %value = arith.constant 42 : i64
    mulberry_core.store %value, %ptr : i64, !mulberry_core.ptr<i64>
    %loaded = mulberry_core.load %ptr : !mulberry_core.ptr<i64> -> i64
    return %loaded : i64
  }
}

// CHECK-LABEL: func.func @main
// CHECK: %[[ONE:.*]] = arith.constant 1 : i64
// CHECK: %[[PTR:.*]] = llvm.alloca %[[ONE]] x i64
// CHECK: arith.constant 42 : i64
// CHECK: llvm.store {{.*}}, %[[PTR]] : i64, !llvm.ptr
// CHECK: %[[LOADED:.*]] = llvm.load %[[PTR]] : !llvm.ptr -> i64
// CHECK: return %[[LOADED]] : i64
// CHECK-NOT: mulberry.
