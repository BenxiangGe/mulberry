// UNSUPPORTED: target={{.*}}
// The current static plugin links LLVM/MLIR support already present in mlir-opt,
// which can duplicate global command-line option registration on load.
// RUN: mlir-opt %s --load-pass-plugin=%cherry_libs/CherryPlugin%shlibext --pass-pipeline="builtin.module(convert-cherry-to-arith-cf-func)" | FileCheck %s

module {
  func.func @main() -> i64 {
    // CHECK-LABEL: arith.constant 10 : i64
    %c10_i64 = arith.constant 10 : i64
    return %c10_i64 : i64
  }
}
