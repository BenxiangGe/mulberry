//===--- Compilation.cpp - Compilation Task Data Structure ----------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/Driver/Compilation.h"
#include "cherry/MLIRGen/Conversion/CherryPasses.h"
#include "cherry/MLIRGen/IR/CherryDialect.h"
#include "cherry/MLIRGen/IR/CherryNNDialect.h"
#include "cherry/MLIRGen/IR/MulberryDialect.h"
#include "cherry/MLIRGen/MLIRGen.h"
#include "cherry/Parse/Lexer.h"
#include "cherry/Parse/Parser.h"
#include "cherry/Sema/Sema.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace cherry;

static auto makeContext() -> mlir::MLIRContext {
  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);
  return mlir::MLIRContext(registry);
}

Compilation::Compilation() : _mlirContext{makeContext()} {}

auto Compilation::make(llvm::StringRef filename,
                       bool enableOpt) -> std::unique_ptr<Compilation> {
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(filename);

  if (auto ec = fileOrErr.getError()) {
    llvm::errs() << "error: " << ec.message() << ": '" << filename << "'\n";
    return {};
  }

  auto compilation = std::make_unique<Compilation>();
  compilation->_mlirContext.getOrLoadDialect<mlir::cherry::CherryDialect>();
  compilation->_mlirContext
      .getOrLoadDialect<mlir::cherry_nn::CherryNNDialect>();
  compilation->_mlirContext.getOrLoadDialect<mlir::mulberry::MulberryDialect>();
  compilation->_mlirContext.getOrLoadDialect<mlir::arith::ArithDialect>();
  compilation->_mlirContext.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  compilation->_mlirContext.getOrLoadDialect<mlir::func::FuncDialect>();
  compilation->_mlirContext.getOrLoadDialect<mlir::memref::MemRefDialect>();
  compilation->_mlirContext.getOrLoadDialect<mlir::scf::SCFDialect>();
  compilation->_mlirContext.getOrLoadDialect<mlir::linalg::LinalgDialect>();
  compilation->_mlirContext.getOrLoadDialect<mlir::math::MathDialect>();

  compilation->sourceManager().AddNewSourceBuffer(std::move(fileOrErr.get()),
                                                  llvm::SMLoc());
  compilation->_enableOpt = enableOpt;
  return compilation;
}

auto Compilation::parse(std::unique_ptr<Module> &module) -> CherryResult {
  auto lexer = std::make_unique<Lexer>(_sourceManager);
  auto parser = Parser{std::move(lexer), _sourceManager};
  return parser.parseModule(module);
}

auto Compilation::genMLIR(mlir::OwningOpRef<mlir::ModuleOp> &module,
                          Lowering lowering) -> CherryResult {
  std::unique_ptr<Module> moduleAST;
  if (parse(moduleAST))
    return failure();

  if (cherry::sema(_sourceManager, *moduleAST.get()) ||
      mlirGen(_sourceManager, _mlirContext, *moduleAST, module))
    return failure();

  mlir::PassManager pm(module.get()->getName());
  mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
  if (_enableOpt)
    optPM.addPass(mlir::createCanonicalizerPass());

  if (lowering >= Lowering::SCF)
    optPM.addPass(mlir::cherry::createConvertCherryToSCF());

  if (lowering >= Lowering::ArithCfFunc)
    optPM.addPass(mlir::cherry::createConvertCherryToArithCfFunc());

  if (lowering >= Lowering::Linalg)
    pm.addPass(mlir::cherry::createConvertCherryNNToLinalg());

  return pm.run(*module);
}

auto Compilation::typecheck() -> int {
  std::unique_ptr<Module> module;
  if (parse(module) || cherry::sema(_sourceManager, *module.get()))
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}

auto Compilation::jit() -> int {
  // TODO: Re-enable this after the high-level Mulberry IR has a real lowering
  // pipeline. The old JIT path was tied to the removed CIR/LLVM bridge.
  llvm::errs() << "error: JIT is temporarily disabled while Mulberry lowering "
                  "is redesigned\n";
  return EXIT_FAILURE;
}

auto Compilation::genObjectFile(const char *outputFileName) -> int {
  (void)outputFileName;
  llvm::errs() << "error: object file generation is temporarily disabled "
                  "while Mulberry lowering is redesigned\n";
  return EXIT_FAILURE;
}

auto Compilation::dumpTokens() -> int {
  auto lexer = std::make_unique<Lexer>(_sourceManager);
  Lexer::tokenize(_sourceManager, *lexer);
  return EXIT_SUCCESS;
}

auto Compilation::dumpParse() -> int {
  std::unique_ptr<Module> module;
  if (parse(module))
    return EXIT_FAILURE;

  cherry::dumpAST(_sourceManager, *module);
  return EXIT_SUCCESS;
}

auto Compilation::dumpAST() -> int {
  std::unique_ptr<Module> module;
  if (parse(module) || cherry::sema(_sourceManager, *module.get()))
    return EXIT_FAILURE;

  cherry::dumpAST(_sourceManager, *module);
  return EXIT_SUCCESS;
}

auto Compilation::dumpMLIR(Lowering lowering) -> int {
  if (lowering >= Lowering::LLVM) {
    // TODO: Replace this with a real Mulberry-to-LLVM lowering pipeline.
    llvm::errs() << "error: LLVM lowering is temporarily disabled while "
                    "Mulberry lowering is redesigned\n";
    return EXIT_FAILURE;
  }

  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (genMLIR(module, lowering))
    return EXIT_FAILURE;

  module->dump();
  return EXIT_SUCCESS;
}

auto Compilation::dumpLLVM() -> int {
  llvm::errs() << "error: LLVM lowering is temporarily disabled while "
                  "Mulberry lowering is redesigned\n";
  return EXIT_FAILURE;
}
