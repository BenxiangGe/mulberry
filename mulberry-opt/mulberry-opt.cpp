//===--- mulberry-opt.cpp - Mulberry optimizer ------------------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "mulberry/MLIRGen/Conversion/MulberryPasses.h"
#include "mulberry/MLIRGen/IR/MulberryDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

namespace LLVM = mlir::LLVM;
namespace arith = mlir::arith;
namespace cf = mlir::cf;
namespace func = mlir::func;
namespace linalg = mlir::linalg;
namespace math = mlir::math;
namespace memref = mlir::memref;
namespace mulberry_core = mlir::mulberry_core;
namespace ptr = mlir::ptr;
namespace scf = mlir::scf;

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mulberry_core::registerMulberryConversionPasses();

  mlir::DialectRegistry registry;
  registry.insert<arith::ArithDialect, cf::ControlFlowDialect,
                  func::FuncDialect, linalg::LinalgDialect,
                  LLVM::LLVMDialect, math::MathDialect,
                  memref::MemRefDialect, mulberry_core::MulberryDialect,
                  ptr::PtrDialect, scf::SCFDialect>();
  arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
  cf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  scf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  // registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Mulberry optimizer driver\n", registry));
}
