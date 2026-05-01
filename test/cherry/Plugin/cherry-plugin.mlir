// RUN: mlir-opt %s --load-dialect-plugin=%cherry_libs/CherryPlugin%shlibext --pass-pipeline="builtin.module(convert-cherry-to-arith-cf-func)" | FileCheck %s

module  {
  func.func @main() -> i64 {
    // CHECK-LABEL:  arith.constant 10 : i64
    %0 = arith.constant 10 : i64
    func.return %0 : i64
  }
}
