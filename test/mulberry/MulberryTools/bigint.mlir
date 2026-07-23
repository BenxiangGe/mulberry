// RUN: mulberry-opt %s | FileCheck %s

module {
  func.func @bigint_semantics() -> i1 {
    %a = bigint.constant "0x1234_5678_0123" : !bigint.int
    %raw = arith.constant 42 : i64
    %b = bigint.from_uint64 %raw : i64 -> !bigint.int
    %sum = bigint.add %a, %b : !bigint.int -> !bigint.int
    %difference = bigint.sub %sum, %b : !bigint.int -> !bigint.int
    %product = bigint.mul %difference, %b : !bigint.int -> !bigint.int
    %masked = bigint.and %product, %a : !bigint.int -> !bigint.int
    %joined = bigint.or %masked, %b : !bigint.int -> !bigint.int
    %changed = bigint.xor %joined, %a : !bigint.int -> !bigint.int
    %leftShift = bigint.shift_left %changed, %raw : !bigint.int, i64 -> !bigint.int
    %rightShift = bigint.shift_right %leftShift, %raw : !bigint.int, i64 -> !bigint.int
    %negative = bigint.neg %product : !bigint.int -> !bigint.int
    %equal = bigint.cmp "eq", %negative, %product : !bigint.int, !bigint.int -> i1
    %notEqual = bigint.cmp "ne", %negative, %product : !bigint.int, !bigint.int -> i1
    %less = bigint.cmp "lt", %negative, %product : !bigint.int, !bigint.int -> i1
    %lessEqual = bigint.cmp "le", %negative, %product : !bigint.int, !bigint.int -> i1
    %greater = bigint.cmp "gt", %negative, %product : !bigint.int, !bigint.int -> i1
    %greaterEqual = bigint.cmp "ge", %negative, %product : !bigint.int, !bigint.int -> i1
    return %less : i1
  }
}

// CHECK-LABEL: func.func @bigint_semantics
// CHECK: %[[A:.*]] = bigint.constant "0x1234_5678_0123" : !bigint.int
// CHECK: %[[RAW:.*]] = arith.constant 42 : i64
// CHECK: %[[B:.*]] = bigint.from_uint64 %[[RAW]] : i64 -> !bigint.int
// CHECK: %[[SUM:.*]] = bigint.add %[[A]], %[[B]] : !bigint.int -> !bigint.int
// CHECK: %[[DIFF:.*]] = bigint.sub %[[SUM]], %[[B]] : !bigint.int -> !bigint.int
// CHECK: %[[PRODUCT:.*]] = bigint.mul %[[DIFF]], %[[B]] : !bigint.int -> !bigint.int
// CHECK: %[[MASKED:.*]] = bigint.and %[[PRODUCT]], %[[A]] : !bigint.int -> !bigint.int
// CHECK: %[[JOINED:.*]] = bigint.or %[[MASKED]], %[[B]] : !bigint.int -> !bigint.int
// CHECK: %[[CHANGED:.*]] = bigint.xor %[[JOINED]], %[[A]] : !bigint.int -> !bigint.int
// CHECK: %[[LEFT:.*]] = bigint.shift_left %[[CHANGED]], %[[RAW]] : !bigint.int, i64 -> !bigint.int
// CHECK: bigint.shift_right %[[LEFT]], %[[RAW]] : !bigint.int, i64 -> !bigint.int
// CHECK: %[[NEGATIVE:.*]] = bigint.neg %[[PRODUCT]] : !bigint.int -> !bigint.int
// CHECK: bigint.cmp "eq", %[[NEGATIVE]], %[[PRODUCT]] : !bigint.int, !bigint.int -> i1
// CHECK: bigint.cmp "ne", %[[NEGATIVE]], %[[PRODUCT]] : !bigint.int, !bigint.int -> i1
// CHECK: bigint.cmp "lt", %[[NEGATIVE]], %[[PRODUCT]] : !bigint.int, !bigint.int -> i1
// CHECK: bigint.cmp "le", %[[NEGATIVE]], %[[PRODUCT]] : !bigint.int, !bigint.int -> i1
// CHECK: bigint.cmp "gt", %[[NEGATIVE]], %[[PRODUCT]] : !bigint.int, !bigint.int -> i1
// CHECK: bigint.cmp "ge", %[[NEGATIVE]], %[[PRODUCT]] : !bigint.int, !bigint.int -> i1
