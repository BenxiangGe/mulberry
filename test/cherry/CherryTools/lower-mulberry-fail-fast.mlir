// RUN: not cherry-opt --lower-mulberry %s 2>&1 | FileCheck %s

module {
  func.func @main() {
    %list = mulberry.list.create() : () -> !mulberry.list<i64>
    return
  }
}

// CHECK: failed to legalize operation 'mulberry.list.create'
