// RUN: not cherry-opt -split-input-file %s 2>&1 | FileCheck %s

func.func @bad_cast() -> i64 {
  %0 = arith.constant 1 : i32
  // CHECK: 'cherry.cast' op only supports bool -> UInt64 and i64 <-> !cir.int<u,64> casts for now
  %1 = cherry.cast %0 : i32 to i64
  return %1 : i64
}
