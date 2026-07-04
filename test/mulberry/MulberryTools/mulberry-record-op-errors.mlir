// RUN: not mulberry-opt %s 2>&1 | FileCheck %s

module {
  func.func @bad_field(
      %record: !mulberry_core.ptr<!mulberry_core.record<Person {age: i64}>>) {
    %field = mulberry_core.record.get_field %record["missing"] : !mulberry_core.ptr<!mulberry_core.record<Person {age: i64}>> -> !mulberry_core.ptr<i64>
    return
  }
}

// CHECK: error: 'mulberry_core.record.get_field' op unknown record field `missing`
