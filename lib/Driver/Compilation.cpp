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
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
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

auto getCherryRuntimeLibPath() -> std::string {
#ifdef CHERRY_RUNTIME_LIB
  return CHERRY_RUNTIME_LIB;
#else
  return {};
#endif
}

auto getDefaultStdlibPath() -> std::string {
#ifdef CHERRY_STDLIB_DIR
  return CHERRY_STDLIB_DIR;
#else
  return "stdlib";
#endif
}

auto importAlias(std::string_view moduleName) -> std::string {
  auto dot = moduleName.rfind('.');
  if (dot == std::string_view::npos)
    return std::string(moduleName);
  return std::string(moduleName.substr(dot + 1));
}

auto isStdlibPackage(std::string_view packageName) -> bool {
  return packageName == "std" || packageName.rfind("std.", 0) == 0;
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

  compilation->_inputFilename = std::string(filename);
  compilation->_enableOpt = enableOpt;
  return compilation;
}

auto Compilation::parse(std::unique_ptr<Module> &module) -> CherryResult {
  _importAliases.clear();
  _loadedModules.clear();

  if (parseFile(_inputFilename, llvm::SMLoc(), module))
    return failure();

  if (!isStdlibPackage(module->packageName()) && loadPrelude(*module))
    return failure();

  return loadImports(*module);
}

auto Compilation::parseFile(const std::string &filename,
                            llvm::SMLoc includeLocation,
                            std::unique_ptr<Module> &module) -> CherryResult {
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (auto ec = fileOrErr.getError()) {
    llvm::errs() << "error: " << ec.message() << ": '" << filename << "'\n";
    return failure();
  }

  auto bufferId = _sourceManager.AddNewSourceBuffer(std::move(fileOrErr.get()),
                                                    includeLocation);
  auto lexer = std::make_unique<Lexer>(_sourceManager, bufferId);
  auto parser = Parser{std::move(lexer), _sourceManager};
  return parser.parseModule(module);
}

auto Compilation::resolveStdlibPath(std::string_view relativePath)
    -> std::string {
  std::string path;
  if (const char *envPath = std::getenv("CHERRY_STDLIB_PATH")) {
    path = envPath;
  } else {
    path = getDefaultStdlibPath();
  }

  llvm::SmallString<256> fullPath(path);
  llvm::sys::path::append(fullPath, relativePath);
  return std::string(fullPath.str());
}

auto Compilation::resolveImportPath(std::string_view moduleName)
    -> std::string {
  if (moduleName.rfind("std.", 0) != 0) {
    llvm::errs() << "error: only stdlib imports are supported now: '"
                 << moduleName << "'\n";
    return {};
  }

  std::string relativePath(moduleName);
  std::replace(relativePath.begin(), relativePath.end(), '.', '/');
  return resolveStdlibPath(relativePath + ".cherry");
}

auto Compilation::loadPrelude(Module &module) -> CherryResult {
  std::unique_ptr<Module> preludeModule;
  auto preludePath = resolveStdlibPath("prelude.cherry");
  if (parseFile(preludePath, module.location(), preludeModule) ||
      loadImports(*preludeModule))
    return failure();

  VectorUniquePtr<Decl> declarations = preludeModule->takeDeclarations();
  for (auto &decl : module.takeDeclarations())
    declarations.push_back(std::move(decl));

  module.setDeclarations(std::move(declarations));
  return success();
}

auto Compilation::loadImports(Module &module) -> CherryResult {
  VectorUniquePtr<Decl> mergedDeclarations;
  for (auto &decl : module.takeDeclarations()) {
    if (auto *importDecl = llvm::dyn_cast<ImportDecl>(decl.get())) {
      auto moduleName = std::string(importDecl->moduleName());
      _importAliases[importAlias(moduleName)] = moduleName;
      if (_loadedModules.insert(moduleName).second) {
        std::unique_ptr<Module> importedModule;
        auto importPath = resolveImportPath(moduleName);
        if (importPath.empty())
          return failure();
        if (parseFile(importPath, importDecl->location(), importedModule) ||
            loadImports(*importedModule))
          return failure();

        for (auto &importedDecl : importedModule->takeDeclarations())
          mergedDeclarations.push_back(std::move(importedDecl));
      }

      continue;
    }

    mergedDeclarations.push_back(std::move(decl));
  }

  module.setDeclarations(std::move(mergedDeclarations));
  return success();
}

auto Compilation::genMLIR(mlir::OwningOpRef<mlir::ModuleOp> &module,
                          Lowering lowering) -> CherryResult {
  std::unique_ptr<Module> moduleAST;
  if (parse(moduleAST))
    return failure();

  if (cherry::sema(_sourceManager, *moduleAST.get(), _importAliases) ||
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
  if (parse(module) ||
      cherry::sema(_sourceManager, *module.get(), _importAliases))
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
  auto cherryRuntimeLibPath = getCherryRuntimeLibPath();
  std::vector<llvm::StringRef> sharedLibPaths;
  if (!runtimeLibPath.empty())
    sharedLibPaths.push_back(runtimeLibPath);
  if (!cherryRuntimeLibPath.empty())
    sharedLibPaths.push_back(cherryRuntimeLibPath);
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
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(_inputFilename);
  if (auto ec = fileOrErr.getError()) {
    llvm::errs() << "error: " << ec.message() << ": '" << _inputFilename
                 << "'\n";
    return EXIT_FAILURE;
  }

  auto bufferId = _sourceManager.AddNewSourceBuffer(std::move(fileOrErr.get()),
                                                    llvm::SMLoc());
  auto lexer = std::make_unique<Lexer>(_sourceManager, bufferId);
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
  if (parse(module) ||
      cherry::sema(_sourceManager, *module.get(), _importAliases))
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
