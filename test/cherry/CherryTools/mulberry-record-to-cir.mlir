// RUN: cherry-opt --convert-mulberry-record-to-cir %s | FileCheck %s

module {
  cir.func @main() {
    %initialAge = cir.const #cir.int<7> : !cir.int<u, 64>
    %initialActive = cir.const #cir.bool<true> : !cir.bool
    %created = mulberry.record.create %initialAge, %initialActive : (!cir.int<u, 64>, !cir.bool) -> !mulberry.record<Person {age: !cir.int<u, 64>, active: !cir.bool}>
    %record = mulberry.alloca !mulberry.record<Person {age: !cir.int<u, 64>, active: !cir.bool}> : !mulberry.ptr<!mulberry.record<Person {age: !cir.int<u, 64>, active: !cir.bool}>>
    %age = mulberry.record.get_field %record["age"] : !mulberry.ptr<!mulberry.record<Person {age: !cir.int<u, 64>, active: !cir.bool}>> -> !mulberry.ptr<!cir.int<u, 64>>
    %value = cir.const #cir.int<42> : !cir.int<u, 64>
    mulberry.store %value, %age : !cir.int<u, 64>, !mulberry.ptr<!cir.int<u, 64>>
    %loaded = mulberry.load %age : !mulberry.ptr<!cir.int<u, 64>> -> !cir.int<u, 64>
    cir.return
  }

  func.func @make_person(%ageValue: !cir.int<u, 64>, %activeValue: !cir.bool) -> !mulberry.record<Person {age: !cir.int<u, 64>, active: !cir.bool}> {
    %record = mulberry.alloca !mulberry.record<Person {age: !cir.int<u, 64>, active: !cir.bool}> : !mulberry.ptr<!mulberry.record<Person {age: !cir.int<u, 64>, active: !cir.bool}>>
    %age = mulberry.record.get_field %record["age"] : !mulberry.ptr<!mulberry.record<Person {age: !cir.int<u, 64>, active: !cir.bool}>> -> !mulberry.ptr<!cir.int<u, 64>>
    mulberry.store %ageValue, %age : !cir.int<u, 64>, !mulberry.ptr<!cir.int<u, 64>>
    %active = mulberry.record.get_field %record["active"] : !mulberry.ptr<!mulberry.record<Person {age: !cir.int<u, 64>, active: !cir.bool}>> -> !mulberry.ptr<!cir.bool>
    mulberry.store %activeValue, %active : !cir.bool, !mulberry.ptr<!cir.bool>
    %value = mulberry.load %record : !mulberry.ptr<!mulberry.record<Person {age: !cir.int<u, 64>, active: !cir.bool}>> -> !mulberry.record<Person {age: !cir.int<u, 64>, active: !cir.bool}>
    return %value : !mulberry.record<Person {age: !cir.int<u, 64>, active: !cir.bool}>
  }

  cir.func @call_make_person() {
    %age = cir.const #cir.int<7> : !cir.int<u, 64>
    %active = cir.const #cir.bool<true> : !cir.bool
    %person = func.call @make_person(%age, %active) : (!cir.int<u, 64>, !cir.bool) -> !mulberry.record<Person {age: !cir.int<u, 64>, active: !cir.bool}>
    %record = mulberry.alloca !mulberry.record<Person {age: !cir.int<u, 64>, active: !cir.bool}> : !mulberry.ptr<!mulberry.record<Person {age: !cir.int<u, 64>, active: !cir.bool}>>
    mulberry.store %person, %record : !mulberry.record<Person {age: !cir.int<u, 64>, active: !cir.bool}>, !mulberry.ptr<!mulberry.record<Person {age: !cir.int<u, 64>, active: !cir.bool}>>
    cir.return
  }
}

// CHECK: !rec_Person = !cir.record<struct "Person" {!u64i, !cir.bool}>
// CHECK-LABEL: cir.func @main()
// CHECK: cir.alloca !rec_Person, !cir.ptr<!rec_Person>
// CHECK: cir.get_member {{.*}}[0] {name = "age"}
// CHECK: cir.get_member {{.*}}[1] {name = "active"}
// CHECK: cir.load
// CHECK: cir.alloca !rec_Person, !cir.ptr<!rec_Person>
// CHECK: cir.get_member {{.*}}[0] {name = "age"}
// CHECK: cir.store
// CHECK: cir.load
// CHECK-LABEL: cir.func @make_person
// CHECK-SAME: (%{{.*}}: !u64i, %{{.*}}: !cir.bool) -> !rec_Person
// CHECK: cir.return {{.*}} : !rec_Person
// CHECK-LABEL: cir.func @call_make_person()
// CHECK: cir.call @make_person
// CHECK-SAME: : (!u64i, !cir.bool) -> !rec_Person
// CHECK-NOT: mulberry.
// CHECK-NOT: func.
