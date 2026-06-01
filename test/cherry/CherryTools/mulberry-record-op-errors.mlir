// RUN: not cherry-opt -split-input-file %s 2>&1 | FileCheck %s

module {
  func.func @bad_create_field_count() {
    %value = arith.constant 42 : i64
    %record = mulberry.record.create %value : (i64) -> !mulberry.record<Person {age: i64, active: i1}>
    return
  }
}

// CHECK: error: 'mulberry.record.create' op field count must match record type

// -----

module {
  func.func @bad_create_field_type() {
    %age = arith.constant 42 : i64
    %active = arith.constant 7 : i64
    %record = mulberry.record.create %age, %active : (i64, i64) -> !mulberry.record<Person {age: i64, active: i1}>
    return
  }
}

// CHECK: error: 'mulberry.record.create' op field `active` type must match record type

// -----

module {
  func.func @bad_field(
      %record: !mulberry.ptr<!mulberry.record<Person {age: i64}>>) {
    %field = mulberry.record.get_field %record["missing"] : !mulberry.ptr<!mulberry.record<Person {age: i64}>> -> !mulberry.ptr<i64>
    return
  }
}

// CHECK: error: 'mulberry.record.get_field' op unknown record field `missing`

// -----

module {
  func.func @bad_alloca() {
    %record = mulberry.alloca !mulberry.record<Person {age: i64}> : !mulberry.ptr<i64>
    return
  }
}

// CHECK: error: 'mulberry.alloca' op result pointer element type must match alloca type

// -----

module {
  func.func @bad_load(%ptr: !mulberry.ptr<i64>) {
    %value = mulberry.load %ptr : !mulberry.ptr<i64> -> i1
    return
  }
}

// CHECK: error: 'mulberry.load' op result type must match pointer element type

// -----

module {
  func.func @bad_store(%ptr: !mulberry.ptr<i64>) {
    %value = arith.constant true
    mulberry.store %value, %ptr : i1, !mulberry.ptr<i64>
    return
  }
}

// CHECK: error: 'mulberry.store' op value type must match pointer element type

// -----

module {
  func.func @bad_get_field_input(%ptr: !mulberry.ptr<i64>) {
    %field = mulberry.record.get_field %ptr["age"] : !mulberry.ptr<i64> -> !mulberry.ptr<i64>
    return
  }
}

// CHECK: error: 'mulberry.record.get_field' op input must be a pointer to a Mulberry record

// -----

module {
  func.func @bad_get_field_result(
      %record: !mulberry.ptr<!mulberry.record<Person {age: i64}>>) {
    %field = mulberry.record.get_field %record["age"] : !mulberry.ptr<!mulberry.record<Person {age: i64}>> -> !mulberry.ptr<i1>
    return
  }
}

// CHECK: error: 'mulberry.record.get_field' op result pointer element type must match field type
