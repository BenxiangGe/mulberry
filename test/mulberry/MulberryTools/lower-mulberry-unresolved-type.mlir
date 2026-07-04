// RUN: mulberry-opt --lower-mulberry %s | FileCheck %s

module {
  func.func private @use_record(
      !mulberry_core.record<Person {age: i64, active: i1}>)
}

// CHECK: func.func private @use_record(!llvm.struct<(i64, i1)>)
