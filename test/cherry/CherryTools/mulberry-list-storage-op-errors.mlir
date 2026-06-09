// RUN: not cherry-opt %s 2>&1 | FileCheck %s

module {
  func.func @main(%length: index, %index: index, %value: i1) {
    %storage = mulberry.list.alloc %length : !mulberry.list_storage<i64>
    mulberry.list.store %value, %storage[%index] : i1,
        !mulberry.list_storage<i64>
    return
  }

  func.func @bad_escape(%length: index) {
    %storage = mulberry.list.alloc %length : !mulberry.list_storage<i64>
    %escaped = mulberry.list.escape_storage %storage, %length
        : !mulberry.list_storage<i64> -> !mulberry.list_storage<i1>
    return
  }
}

// CHECK: error: 'mulberry.list.store' op value type must match list storage element type
// CHECK: error: 'mulberry.list.escape_storage' op result storage element type must match input storage element type
