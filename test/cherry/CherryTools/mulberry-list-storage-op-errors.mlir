// RUN: not cherry-opt %s 2>&1 | FileCheck %s

module {
  func.func @main(%length: index, %index: index, %value: i1) {
    %storage = mulberry.list.alloc %length : !mulberry.list_storage<i64>
    mulberry.list.store %value, %storage[%index] : i1,
        !mulberry.list_storage<i64>
    return
  }
}

// CHECK: error: 'mulberry.list.store' op value type must match list storage element type
