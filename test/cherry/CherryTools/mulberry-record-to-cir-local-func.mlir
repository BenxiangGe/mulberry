// RUN: cherry-opt --convert-mulberry-record-to-cir %s | FileCheck %s

module {
  func.func @record_body() -> !cir.int<u, 64> {
    %record = mulberry.alloca !mulberry.record<Person {age: !cir.int<u, 64>}> : !mulberry.ptr<!mulberry.record<Person {age: !cir.int<u, 64>}>>
    %age = mulberry.record.get_field %record["age"] : !mulberry.ptr<!mulberry.record<Person {age: !cir.int<u, 64>}>> -> !mulberry.ptr<!cir.int<u, 64>>
    %value = cir.const #cir.int<42> : !cir.int<u, 64>
    mulberry.store %value, %age : !cir.int<u, 64>, !mulberry.ptr<!cir.int<u, 64>>
    return %value : !cir.int<u, 64>
  }
}

// CHECK: !rec_Person = !cir.record<struct "Person" {!u64i}>
// CHECK-LABEL: cir.func @record_body()
// CHECK-SAME: -> !u64i
// CHECK: cir.alloca !rec_Person, !cir.ptr<!rec_Person>
// CHECK: cir.get_member {{.*}}[0] {name = "age"}
// CHECK: cir.store
// CHECK: cir.return {{.*}} : !u64i
// CHECK-NOT: mulberry.
// CHECK-NOT: func.
