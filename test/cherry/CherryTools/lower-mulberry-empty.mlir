// RUN: cherry-opt --lower-mulberry %s | FileCheck %s

module {
  func.func @main() -> i64 {
    %value = arith.constant 0 : i64
    return %value : i64
  }
}

// CHECK-LABEL: func.func @main
// CHECK: arith.constant 0 : i64
