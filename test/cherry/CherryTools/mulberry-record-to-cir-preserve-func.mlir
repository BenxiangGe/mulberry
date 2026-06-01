// RUN: cherry-opt --convert-mulberry-record-to-cir %s | FileCheck %s

module {
  func.func @plain(%value: i64) -> i64 {
    return %value : i64
  }
}

// CHECK-LABEL: func.func @plain
// CHECK-SAME: (%{{.*}}: i64) -> i64
// CHECK: return {{.*}} : i64
// CHECK-NOT: cir.func
