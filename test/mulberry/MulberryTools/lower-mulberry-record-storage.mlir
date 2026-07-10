// RUN: mulberry-opt --lower-mulberry %s | FileCheck %s

module {
  func.func @main() -> i1 {
    %record = mulberry_core.alloca !mulberry_core.record<Person {age: i64, active: i1}> : !mulberry_core.ptr<!mulberry_core.record<Person {age: i64, active: i1}>>
    %age = mulberry_core.record.get_field %record["age"] : !mulberry_core.ptr<!mulberry_core.record<Person {age: i64, active: i1}>> -> !mulberry_core.ptr<i64>
    %value = arith.constant 42 : i64
    mulberry_core.store %value, %age : i64, !mulberry_core.ptr<i64>
    %active = mulberry_core.record.get_field %record["active"] : !mulberry_core.ptr<!mulberry_core.record<Person {age: i64, active: i1}>> -> !mulberry_core.ptr<i1>
    %true = arith.constant true
    mulberry_core.store %true, %active : i1, !mulberry_core.ptr<i1>
    %loaded = mulberry_core.load %active : !mulberry_core.ptr<i1> -> i1
    return %loaded : i1
  }
}

// CHECK-LABEL: func.func @main
// CHECK: %[[ONE:.*]] = arith.constant 1 : i64
// CHECK: %[[RECORD:.*]] = llvm.alloca %[[ONE]] x !llvm.struct<(i64, i1)>
// CHECK: %[[AGE:.*]] = llvm.getelementptr %[[RECORD]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i1)>
// CHECK: arith.constant 42 : i64
// CHECK: llvm.store {{.*}}, %[[AGE]] : i64, !llvm.ptr
// CHECK: %[[ACTIVE:.*]] = llvm.getelementptr %[[RECORD]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i1)>
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: llvm.store %[[TRUE]], %[[ACTIVE]] : i1, !llvm.ptr
// CHECK: %[[LOADED:.*]] = llvm.load %[[ACTIVE]] : !llvm.ptr -> i1
// CHECK: return %[[LOADED]] : i1
// CHECK-NOT: mulberry.
