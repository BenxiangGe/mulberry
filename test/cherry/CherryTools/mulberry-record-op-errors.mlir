// RUN: not cherry-opt %s 2>&1 | FileCheck %s

module {
  func.func @bad_field(
      %record: !mulberry.ptr<!mulberry.record<Person {age: i64}>>) {
    %field = mulberry.record.get_field %record["missing"] : !mulberry.ptr<!mulberry.record<Person {age: i64}>> -> !mulberry.ptr<i64>
    return
  }
}

// CHECK: error: 'mulberry.record.get_field' op unknown record field `missing`
