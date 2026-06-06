// RUN: not cherry-opt --lower-mulberry %s 2>&1 | FileCheck %s

module {
  func.func @main() {
    %ptr = mulberry.alloca !mulberry.list<i64>
        : !mulberry.ptr<!mulberry.list<i64>>
    return
  }
}

// CHECK: failed to legalize operation 'mulberry.alloca'
