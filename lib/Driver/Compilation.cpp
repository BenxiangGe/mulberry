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
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <string>
#include <vector>

using namespace cherry;

namespace {

auto getRuntimeLibPath() -> std::string {
#ifdef CHERRY_MLIR_C_RUNNER_UTILS
  return CHERRY_MLIR_C_RUNNER_UTILS;
#else
  return {};
#endif
}

auto registerLLVMTranslations(mlir::ModuleOp module) -> void {
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());
}

auto createTargetMachine()
    -> llvm::Expected<std::unique_ptr<llvm::TargetMachine>> {
  auto builder = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!builder)
    return builder.takeError();

  return builder->createTargetMachine();
}

} // namespace

static auto makeContext() -> mlir::MLIRContext {
  mlir::DialectRegistry registry;
  // `convert-to-llvm` finds lowering patterns through dialect interfaces.
  // Loading dialects alone is not enough; their ConvertToLLVM extensions must
  // also be registered before the context starts loading dialects.
  mlir::registerAllExtensions(registry);
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
  compilation->_mlirContext.getOrLoadDialect<mlir::ptr::PtrDialect>();

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

  if (lowering >= Lowering::Mulberry) {
    pm.addPass(mlir::cherry::createPrepareMulberryBoundaries());
    pm.addPass(mlir::cherry::createLowerMulberry());
  }

  if (lowering >= Lowering::LLVM) {
    mlir::ConvertToLLVMPassOptions llvmOptions;
    llvmOptions.filterDialects = {"arith", "cf", "func", "math", "memref"};
    pm.addPass(mlir::cherry::createLowerCherryRuntime());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::createConvertLinalgToLoopsPass());
    pm.addPass(mlir::createSCFToControlFlowPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::LLVM::createLLVMRequestCWrappersPass());
    pm.addPass(mlir::createConvertToLLVMPass(std::move(llvmOptions)));
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  }

  return pm.run(*module);
}

auto Compilation::typecheck() -> int {
  std::unique_ptr<Module> module;
  if (parse(module) || cherry::sema(_sourceManager, *module.get()))
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}

auto Compilation::jit() -> int {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (genMLIR(module, Lowering::LLVM))
    return EXIT_FAILURE;

  registerLLVMTranslations(*module);

  auto targetMachine = createTargetMachine();
  if (!targetMachine) {
    llvm::errs() << "error: failed to create target machine: "
                 << llvm::toString(targetMachine.takeError()) << "\n";
    return EXIT_FAILURE;
  }

  auto transformer = mlir::makeOptimizingTransformer(
      _enableOpt ? 3 : 0, 0, targetMachine->get());

  mlir::ExecutionEngineOptions options;
  options.transformer = transformer;
  options.jitCodeGenOptLevel =
      _enableOpt ? llvm::CodeGenOptLevel::Aggressive
                 : llvm::CodeGenOptLevel::None;

  auto runtimeLibPath = getRuntimeLibPath();
  std::vector<llvm::StringRef> sharedLibPaths;
  if (!runtimeLibPath.empty())
    sharedLibPaths.push_back(runtimeLibPath);
  options.sharedLibPaths = sharedLibPaths;

  auto engine = mlir::ExecutionEngine::create(*module, options,
                                              std::move(*targetMachine));
  if (!engine) {
    llvm::errs() << "error: failed to create execution engine: "
                 << llvm::toString(engine.takeError()) << "\n";
    return EXIT_FAILURE;
  }

  (*engine)->initialize();

  uint64_t result = 0;
  auto error =
      (*engine)->invoke("main", mlir::ExecutionEngine::result(result));
  if (error) {
    llvm::errs() << "error: JIT invocation failed: "
                 << llvm::toString(std::move(error)) << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
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
  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (genMLIR(module, lowering))
    return EXIT_FAILURE;

  module->dump();
  return EXIT_SUCCESS;
}

auto Compilation::dumpLLVM() -> int {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (genMLIR(module, Lowering::LLVM))
    return EXIT_FAILURE;

  registerLLVMTranslations(*module);

  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "error: failed to translate MLIR LLVM dialect to LLVM IR\n";
    return EXIT_FAILURE;
  }

  auto targetMachine = createTargetMachine();
  if (!targetMachine) {
    llvm::errs() << "error: failed to create target machine: "
                 << llvm::toString(targetMachine.takeError()) << "\n";
    return EXIT_FAILURE;
  }

  mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(),
                                                        targetMachine->get());

  auto transformer = mlir::makeOptimizingTransformer(
      _enableOpt ? 3 : 0, 0, targetMachine->get());
  if (auto error = transformer(llvmModule.get())) {
    llvm::errs() << "error: failed to optimize LLVM IR: "
                 << llvm::toString(std::move(error)) << "\n";
    return EXIT_FAILURE;
  }

  llvm::outs() << *llvmModule << "\n";
  return EXIT_SUCCESS;
}
