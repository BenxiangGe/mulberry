// RUN: cherry-opt %s | FileCheck %s

module {
  func.func @record_ops() -> i64 {
    %active = arith.constant true
    %initialAge = arith.constant 7 : i64
    %created = mulberry.record.create %initialAge, %active : (i64, i1) -> !mulberry.record<Person {age: i64, active: i1}>
    %record = mulberry.alloca !mulberry.record<Person {age: i64, active: i1}> : !mulberry.ptr<!mulberry.record<Person {age: i64, active: i1}>>
    %age = mulberry.record.get_field %record["age"] : !mulberry.ptr<!mulberry.record<Person {age: i64, active: i1}>> -> !mulberry.ptr<i64>
    %value = arith.constant 42 : i64
    mulberry.store %value, %age : i64, !mulberry.ptr<i64>
    %loaded = mulberry.load %age : !mulberry.ptr<i64> -> i64
    return %loaded : i64
  }
}

// CHECK-LABEL: func.func @record_ops
// CHECK: mulberry.record.create {{.*}} : (i64, i1) -> !mulberry.record<Person {age: i64, active: i1}>
// CHECK: mulberry.alloca !mulberry.record<Person {age: i64, active: i1}> : !mulberry.ptr<!mulberry.record<Person {age: i64, active: i1}>>
// CHECK: mulberry.record.get_field {{.*}}["age"] : !mulberry.ptr<!mulberry.record<Person {age: i64, active: i1}>> -> !mulberry.ptr<i64>
// CHECK: mulberry.store
// CHECK: mulberry.load
