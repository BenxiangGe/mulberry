// RUN: cherry-opt --lower-mulberry %s | FileCheck %s

module {
  func.func @identity(%ptr: !mulberry.ptr<i64>) -> !mulberry.ptr<i64> {
    return %ptr : !mulberry.ptr<i64>
  }

  func.func @main() -> i64 {
    %ptr = mulberry.alloca i64 : !mulberry.ptr<i64>
    %value = arith.constant 42 : i64
    mulberry.store %value, %ptr : i64, !mulberry.ptr<i64>
    %same = call @identity(%ptr)
        : (!mulberry.ptr<i64>) -> !mulberry.ptr<i64>
    %loaded = mulberry.load %same : !mulberry.ptr<i64> -> i64
    return %loaded : i64
  }
}

// CHECK-LABEL: func.func @identity
// CHECK-SAME: (%[[ARG:.*]]: !llvm.ptr) -> !llvm.ptr
// CHECK: return %[[ARG]] : !llvm.ptr
// CHECK-LABEL: func.func @main
// CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[PTR:.*]] = llvm.alloca %[[ONE]] x i64
// CHECK: %[[SAME:.*]] = call @identity(%[[PTR]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK: %[[LOADED:.*]] = llvm.load %[[SAME]] : !llvm.ptr -> i64
// CHECK: return %[[LOADED]] : i64
// CHECK-NOT: mulberry.
