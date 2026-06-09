// RUN: cherry-opt --prepare-mulberry-boundaries %s | FileCheck %s

module {
  func.func @main(%value: i64) -> i64 {
    return %value : i64
  }
}

// CHECK-LABEL: func.func @main
// CHECK-SAME: (%[[VALUE:.*]]: i64) -> i64
// CHECK: return %[[VALUE]] : i64
