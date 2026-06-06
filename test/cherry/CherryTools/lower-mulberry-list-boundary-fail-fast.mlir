// RUN: not cherry-opt --lower-mulberry %s 2>&1 | FileCheck %s

module {
  func.func private @use_nested_list(
      !mulberry.list<!mulberry.list<i64>>)
}

// CHECK: failed to legalize operation 'func.func'
