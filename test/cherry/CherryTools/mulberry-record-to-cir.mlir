// RUN: cherry-opt --convert-mulberry-record-to-cir %s | FileCheck %s

module {
  cir.func @main() {
    %record = mulberry.alloca !mulberry.record<Person {age: !cir.int<u, 64>, active: !cir.bool}> : !mulberry.ptr<!mulberry.record<Person {age: !cir.int<u, 64>, active: !cir.bool}>>
    %age = mulberry.record.get_field %record["age"] : !mulberry.ptr<!mulberry.record<Person {age: !cir.int<u, 64>, active: !cir.bool}>> -> !mulberry.ptr<!cir.int<u, 64>>
    %value = cir.const #cir.int<42> : !cir.int<u, 64>
    mulberry.store %value, %age : !cir.int<u, 64>, !mulberry.ptr<!cir.int<u, 64>>
    %loaded = mulberry.load %age : !mulberry.ptr<!cir.int<u, 64>> -> !cir.int<u, 64>
    cir.return
  }
}

// CHECK: !rec_Person = !cir.record<struct "Person" {!u64i, !cir.bool}>
// CHECK-LABEL: cir.func @main()
// CHECK: cir.alloca !rec_Person, !cir.ptr<!rec_Person>
// CHECK: cir.get_member {{.*}}[0] {name = "age"}
// CHECK: cir.store
// CHECK: cir.load
// CHECK-NOT: mulberry.
