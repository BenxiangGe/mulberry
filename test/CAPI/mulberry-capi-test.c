//===--- mulberry-capi-test.c - Simple demo of C-API ----------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

// RUN: mulberry-capi-test 2>&1 | FileCheck %s

#include "mlir-c/IR.h"
#include "mlir-c/RegisterEverything.h"
#include <stdio.h>

static void registerAllUpstreamDialects(MlirContext ctx) {
  MlirDialectRegistry registry = mlirDialectRegistryCreate();
  mlirRegisterAllDialects(registry);
  mlirContextAppendDialectRegistry(ctx, registry);
  mlirDialectRegistryDestroy(registry);
}

int main(int argc, char **argv) {
  MlirContext ctx = mlirContextCreate();
  registerAllUpstreamDialects(ctx);

  MlirModule module = mlirModuleCreateParse(
      ctx, mlirStringRefCreateFromCString("llvm.func @printU64(i64)\n"
                                          "func.func @main() -> i64 {\n"
                                          "  %0 = arith.constant 10 : i64\n"
                                          "  llvm.call @printU64(%0) : (i64) -> ()\n"
                                          "  %1 = arith.constant 0 : i64\n"
                                          "  func.return %1 : i64\n"
                                          "}"));
  if (mlirModuleIsNull(module)) {
    printf("ERROR: Could not parse.\n");
    mlirContextDestroy(ctx);
    return 1;
  }
  MlirOperation op = mlirModuleGetOperation(module);

  // CHECK: %[[C:.*]] = arith.constant 10 : i64
  // CHECK: llvm.call @printU64(%[[C]]) : (i64) -> ()
  mlirOperationDump(op);

  mlirModuleDestroy(module);
  mlirContextDestroy(ctx);
  return 0;
}
