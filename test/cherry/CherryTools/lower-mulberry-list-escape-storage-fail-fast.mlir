// RUN: not cherry-opt --lower-mulberry %s 2>&1 | FileCheck %s

module {
  func.func @escape_storage(%length: index) {
    %storage = mulberry.list.alloc %length : !mulberry.list_storage<i64>
    %escaped = mulberry.list.escape_storage %storage, %length
        : !mulberry.list_storage<i64> -> !mulberry.list_storage<i64>
    return
  }
}

// CHECK: failed to legalize operation 'mulberry.list.escape_storage': only List<TensorDesc> escaping storage has heap lowering today
