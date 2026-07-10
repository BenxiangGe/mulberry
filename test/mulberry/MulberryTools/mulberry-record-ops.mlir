// RUN: mulberry-opt %s | FileCheck %s

module {
  func.func @record_ops() -> i64 {
    %record = mulberry_core.alloca !mulberry_core.record<Person {age: i64, active: i1}> : !mulberry_core.ptr<!mulberry_core.record<Person {age: i64, active: i1}>>
    %age = mulberry_core.record.get_field %record["age"] : !mulberry_core.ptr<!mulberry_core.record<Person {age: i64, active: i1}>> -> !mulberry_core.ptr<i64>
    %value = arith.constant 42 : i64
    mulberry_core.store %value, %age : i64, !mulberry_core.ptr<i64>
    %loaded = mulberry_core.load %age : !mulberry_core.ptr<i64> -> i64
    return %loaded : i64
  }
}

// CHECK-LABEL: func.func @record_ops
// CHECK: mulberry_core.alloca !mulberry_core.record<Person {age: i64, active: i1}> : !mulberry_core.ptr<!mulberry_core.record<Person {age: i64, active: i1}>>
// CHECK: mulberry_core.record.get_field {{.*}}["age"] : !mulberry_core.ptr<!mulberry_core.record<Person {age: i64, active: i1}>> -> !mulberry_core.ptr<i64>
// CHECK: mulberry_core.store
// CHECK: mulberry_core.load
