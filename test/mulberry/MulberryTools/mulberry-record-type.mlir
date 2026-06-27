// RUN: mulberry-opt %s | FileCheck %s

module {
  func.func private @use_record(
      %arg0: !mulberry.record<Person {age: i64, active: i1}>)
}

// CHECK: func.func private @use_record(!mulberry.record<Person {age: i64, active: i1}>)
