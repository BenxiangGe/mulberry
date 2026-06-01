// RUN: cherry-opt --convert-cherry-to-llvm %s | FileCheck %s

module {
  func.func @record_ops() -> i64 {
    %record = mulberry.alloca !mulberry.record<Person {age: i64, active: i1}> : !mulberry.ptr<!mulberry.record<Person {age: i64, active: i1}>>
    %age = mulberry.record.get_field %record["age"] : !mulberry.ptr<!mulberry.record<Person {age: i64, active: i1}>> -> !mulberry.ptr<i64>
    %value = arith.constant 42 : i64
    mulberry.store %value, %age : i64, !mulberry.ptr<i64>
    %loaded = mulberry.load %age : !mulberry.ptr<i64> -> i64
    return %loaded : i64
  }

  func.func @make_person(%ageValue: i64, %activeValue: i1) -> !mulberry.record<Person {age: i64, active: i1}> {
    %record = mulberry.alloca !mulberry.record<Person {age: i64, active: i1}> : !mulberry.ptr<!mulberry.record<Person {age: i64, active: i1}>>
    %age = mulberry.record.get_field %record["age"] : !mulberry.ptr<!mulberry.record<Person {age: i64, active: i1}>> -> !mulberry.ptr<i64>
    mulberry.store %ageValue, %age : i64, !mulberry.ptr<i64>
    %active = mulberry.record.get_field %record["active"] : !mulberry.ptr<!mulberry.record<Person {age: i64, active: i1}>> -> !mulberry.ptr<i1>
    mulberry.store %activeValue, %active : i1, !mulberry.ptr<i1>
    %value = mulberry.load %record : !mulberry.ptr<!mulberry.record<Person {age: i64, active: i1}>> -> !mulberry.record<Person {age: i64, active: i1}>
    return %value : !mulberry.record<Person {age: i64, active: i1}>
  }
}

// CHECK-LABEL: llvm.func @record_ops
// CHECK: llvm.alloca {{.*}} x !llvm.struct<(i64, i1)>
// CHECK: llvm.getelementptr {{.*}}[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i1)>
// CHECK: llvm.store
// CHECK: llvm.load {{.*}} : !llvm.ptr -> i64
// CHECK: llvm.return {{.*}} : i64

// CHECK-LABEL: llvm.func @make_person
// CHECK-SAME: (%{{.*}}: i64, %{{.*}}: i1) -> !llvm.struct<(i64, i1)>
// CHECK: llvm.getelementptr {{.*}}[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i1)>
// CHECK: llvm.getelementptr {{.*}}[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i1)>
// CHECK: llvm.load {{.*}} : !llvm.ptr -> !llvm.struct<(i64, i1)>
// CHECK: llvm.return {{.*}} : !llvm.struct<(i64, i1)>
// CHECK-NOT: mulberry.
