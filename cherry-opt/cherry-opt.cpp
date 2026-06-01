//===--- cherry-opt.cpp - Cherry optimiser ----------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/Conversion/CherryPasses.h"
#include "cherry/MLIRGen/IR/CherryDialect.h"
#include "cherry/MLIRGen/IR/CherryNNDialect.h"
#include "cherry/MLIRGen/IR/MulberryDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::cherry::registerCherryConversionPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, cir::CIRDialect,
                  mlir::cf::ControlFlowDialect, mlir::cherry::CherryDialect,
                  mlir::cherry_nn::CherryNNDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::LLVM::LLVMDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::mulberry::MulberryDialect, mlir::scf::SCFDialect>();
  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  // registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Cherry optimizer driver\n", registry));
}
