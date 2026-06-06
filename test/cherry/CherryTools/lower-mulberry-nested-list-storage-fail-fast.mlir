// RUN: not cherry-opt --lower-mulberry %s 2>&1 | FileCheck %s

module {
  func.func @nested_list() {
    %inner = mulberry.list.create()
        : () -> !mulberry.list<i64>
    %list = mulberry.list.create(%inner)
        : (!mulberry.list<i64>)
            -> !mulberry.list<!mulberry.list<i64>>
    return
  }
}

// CHECK: failed to legalize operation
