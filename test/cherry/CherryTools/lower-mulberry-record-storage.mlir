// RUN: cherry-opt --lower-mulberry %s | FileCheck %s

module {
  func.func @main() -> i1 {
    %record = mulberry.alloca !mulberry.record<Person {age: i64, active: i1}> : !mulberry.ptr<!mulberry.record<Person {age: i64, active: i1}>>
    %age = mulberry.record.get_field %record["age"] : !mulberry.ptr<!mulberry.record<Person {age: i64, active: i1}>> -> !mulberry.ptr<i64>
    %value = arith.constant 42 : i64
    mulberry.store %value, %age : i64, !mulberry.ptr<i64>
    %recordValue = mulberry.load %record : !mulberry.ptr<!mulberry.record<Person {age: i64, active: i1}>> -> !mulberry.record<Person {age: i64, active: i1}>
    %active = mulberry.record.extract %recordValue["active"] : !mulberry.record<Person {age: i64, active: i1}> -> i1
    return %active : i1
  }
}

// CHECK-LABEL: func.func @main
// CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[RECORD:.*]] = llvm.alloca %[[ONE]] x !llvm.struct<(i64, i1)>
// CHECK: %[[AGE:.*]] = llvm.getelementptr %[[RECORD]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i1)>
// CHECK: arith.constant 42 : i64
// CHECK: llvm.store {{.*}}, %[[AGE]] : i64, !llvm.ptr
// CHECK: %[[VALUE:.*]] = llvm.load %[[RECORD]] : !llvm.ptr -> !llvm.struct<(i64, i1)>
// CHECK: %[[ACTIVE:.*]] = llvm.extractvalue %[[VALUE]][1] : !llvm.struct<(i64, i1)>
// CHECK: return %[[ACTIVE]] : i1
// CHECK-NOT: mulberry.
