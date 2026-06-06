// RUN: not cherry-opt --lower-mulberry %s 2>&1 | FileCheck %s

module {
  func.func private @use_list(!mulberry.list<i64>)
}

// CHECK: failed to legalize operation 'func.func'
