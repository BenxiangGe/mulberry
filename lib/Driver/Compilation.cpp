//===--- Compilation.cpp - Compilation Task Data Structure ----------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "mulberry/Driver/Compilation.h"
#include "mulberry/MLIRGen/Conversion/MulberryPasses.h"
#include "mulberry/MLIRGen/IR/MulberryDialect.h"
#include "mulberry/MLIRGen/MLIRGen.h"
#include "mulberry/Parse/Lexer.h"
#include "mulberry/Parse/Parser.h"
#include "mulberry/Sema/Sema.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

using namespace mulberry;

namespace LLVM = mlir::LLVM;
namespace arith = mlir::arith;
namespace bufferization = mlir::bufferization;
namespace cf = mlir::cf;
namespace func = mlir::func;
namespace linalg = mlir::linalg;
namespace math = mlir::math;
namespace memref = mlir::memref;
namespace mulberry_core = mlir::mulberry_core;
namespace ptr = mlir::ptr;
namespace scf = mlir::scf;

namespace {

auto getRuntimeLibPath() -> std::string {
#ifdef MULBERRY_MLIR_C_RUNNER_UTILS
  return MULBERRY_MLIR_C_RUNNER_UTILS;
#else
  return {};
#endif
}

auto splitRuntimeLibPaths(std::string_view paths) -> std::vector<std::string> {
  std::vector<std::string> result;
  size_t start = 0;
  while (start <= paths.size()) {
    auto separator = paths.find('|', start);
    auto item = separator == std::string_view::npos
                    ? paths.substr(start)
                    : paths.substr(start, separator - start);
    if (!item.empty())
      result.push_back(std::string(item));
    if (separator == std::string_view::npos)
      break;
    start = separator + 1;
  }
  return result;
}

auto getMLIRRuntimeLibPaths() -> std::vector<std::string> {
#ifdef MULBERRY_MLIR_RUNTIME_LIBS
  return splitRuntimeLibPaths(MULBERRY_MLIR_RUNTIME_LIBS);
#else
  auto runtimeLibPath = getRuntimeLibPath();
  if (runtimeLibPath.empty())
    return {};
  return {runtimeLibPath};
#endif
}

auto getMulberryRuntimeLibPath() -> std::string {
#ifdef MULBERRY_RUNTIME_LIB
  return MULBERRY_RUNTIME_LIB;
#else
  return {};
#endif
}

auto getBoehmLinkLibPath() -> std::string {
#ifdef MULBERRY_BDWGC_LINK_LIB
  return MULBERRY_BDWGC_LINK_LIB;
#else
  return {};
#endif
}

auto getBoehmRuntimeLibPaths() -> std::vector<std::string> {
#ifdef MULBERRY_BDWGC_RUNTIME_LIBS
  return splitRuntimeLibPaths(MULBERRY_BDWGC_RUNTIME_LIBS);
#else
  return {};
#endif
}

auto getBoehmRuntimeLibDir() -> std::string {
#ifdef MULBERRY_BDWGC_LIBRARY_DIR
  return MULBERRY_BDWGC_LIBRARY_DIR;
#else
  return {};
#endif
}

auto getBundledPackageRegistry() -> std::string {
#ifdef MULBERRY_BUNDLED_PACKAGE_REGISTRY
  return MULBERRY_BUNDLED_PACKAGE_REGISTRY;
#else
  return {};
#endif
}

auto getDefaultStdlibPath() -> std::string {
#ifdef MULBERRY_STDLIB_DIR
  return MULBERRY_STDLIB_DIR;
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

auto isBundledPackage(std::string_view packageName) -> bool {
  return packageName == "std" || packageName.rfind("std.", 0) == 0 ||
         packageName == "mulberry" ||
         packageName.rfind("mulberry.", 0) == 0;
}

auto normalizeBundledImportName(std::string_view importName) -> std::string {
  if (isBundledPackage(importName))
    return std::string(importName);

  std::string normalizedName = "std.";
  normalizedName += importName;
  return normalizedName;
}

auto isInternalBundledImport(std::string_view importName) -> bool {
  return importName == "std.internal" ||
         importName.rfind("std.internal.", 0) == 0;
}

auto isInternalSourceLocation(const llvm::SourceMgr &sourceManager,
                              llvm::SMLoc location) -> bool {
  if (!location.isValid())
    return false;

  auto bufferId = sourceManager.FindBufferContainingLoc(location);
  if (bufferId == 0)
    return false;

  auto path = std::string(
      sourceManager.getMemoryBuffer(bufferId)->getBufferIdentifier());
  return path.rfind("stdlib/", 0) == 0 ||
         path.find("/stdlib/") != std::string::npos;
}

struct BundledPackageSpec {
  std::string moduleName;
  std::string libraryPath;
  std::string preCorePipeline;
  std::string postCorePipeline;
  bool runSymbolDCEAfterPostCore = false;
};

auto splitRegistryFields(std::string_view line) -> std::vector<std::string> {
  std::vector<std::string> fields;
  size_t start = 0;
  while (start <= line.size()) {
    auto separator = line.find('|', start);
    if (separator == std::string_view::npos) {
      fields.push_back(std::string(line.substr(start)));
      break;
    }
    fields.push_back(std::string(line.substr(start, separator - start)));
    start = separator + 1;
  }
  return fields;
}

auto parseBundledPackageSpec(std::string_view line)
    -> std::optional<BundledPackageSpec> {
  auto fields = splitRegistryFields(line);
  if (fields.size() != 5)
    return std::nullopt;

  return BundledPackageSpec{
      std::move(fields[0]),
      std::move(fields[1]),
      std::move(fields[2]),
      std::move(fields[3]),
      fields[4] == "1",
  };
}

auto parseBundledPackageRegistry(std::string_view registry)
    -> std::vector<BundledPackageSpec> {
  std::vector<BundledPackageSpec> specs;
  size_t start = 0;
  while (start <= registry.size()) {
    auto newline = registry.find('\n', start);
    auto line = newline == std::string_view::npos
                    ? registry.substr(start)
                    : registry.substr(start, newline - start);
    if (!line.empty()) {
      if (auto spec = parseBundledPackageSpec(line))
        specs.push_back(std::move(*spec));
    }
    if (newline == std::string_view::npos)
      break;
    start = newline + 1;
  }
  return specs;
}

auto bundledPackageSpecs() -> const std::vector<BundledPackageSpec> & {
  static const auto specs =
      parseBundledPackageRegistry(getBundledPackageRegistry());
  return specs;
}

auto findBundledPackageSpec(std::string_view moduleName)
    -> const BundledPackageSpec* {
  for (auto& spec : bundledPackageSpecs()) {
    if (spec.moduleName == moduleName)
      return &spec;
  }
  return nullptr;
}

auto splitQualifiedName(std::string_view name)
    -> std::vector<std::string_view> {
  std::vector<std::string_view> segments;
  size_t start = 0;
  while (start <= name.size()) {
    auto dot = name.find('.', start);
    if (dot == std::string_view::npos) {
      segments.push_back(name.substr(start));
      break;
    }
    segments.push_back(name.substr(start, dot - start));
    start = dot + 1;
  }
  return segments;
}

auto joinQualifiedName(const std::vector<std::string_view> &segments,
                       size_t begin, size_t end) -> std::string {
  std::string name;
  for (size_t index = begin; index < end; ++index) {
    if (index != begin)
      name += '.';
    name += segments[index];
  }
  return name;
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

auto initializeMulberryRuntime(mlir::ExecutionEngine &engine)
    -> llvm::Error {
  auto runtimeInit = engine.lookup("mulberry_runtime_init");
  if (!runtimeInit)
    return runtimeInit.takeError();

  reinterpret_cast<void (*)()>(*runtimeInit)();
  return llvm::Error::success();
}

auto genLLVMModule(mlir::ModuleOp module, bool enableOpt,
                   std::unique_ptr<llvm::TargetMachine> &targetMachine,
                   llvm::LLVMContext &llvmContext)
    -> std::unique_ptr<llvm::Module> {
  registerLLVMTranslations(module);

  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "error: failed to translate MLIR LLVM dialect to LLVM IR\n";
    return nullptr;
  }

  mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(),
                                                        targetMachine.get());

  auto transformer =
      mlir::makeOptimizingTransformer(enableOpt ? 3 : 0, 0,
                                      targetMachine.get());
  if (auto error = transformer(llvmModule.get())) {
    llvm::errs() << "error: failed to optimize LLVM IR: "
                 << llvm::toString(std::move(error)) << "\n";
    return nullptr;
  }

  return llvmModule;
}

auto createLLVMObjectFile(llvm::Module &module,
                          llvm::TargetMachine &targetMachine,
                          llvm::StringRef outputFileName) -> bool {
  std::error_code ec;
  llvm::raw_fd_ostream output(outputFileName, ec, llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << "error: failed to open object file '" << outputFileName
                 << "': " << ec.message() << "\n";
    return true;
  }

  llvm::legacy::PassManager pm;
  if (targetMachine.addPassesToEmitFile(pm, output, nullptr,
                                        llvm::CodeGenFileType::ObjectFile)) {
    llvm::errs() << "error: target machine cannot emit object files\n";
    return true;
  }

  pm.run(module);
  output.flush();
  return false;
}

auto addRuntimeMainWrapper(llvm::Module &module) -> bool {
  auto *userMain = module.getFunction("main");
  if (!userMain) {
    llvm::errs() << "error: AOT executable requires a main function\n";
    return true;
  }

  if (!userMain->getFunctionType()->getReturnType()->isIntegerTy(64) ||
      userMain->arg_size() != 0) {
    llvm::errs() << "error: AOT executable main must have type () -> UInt64\n";
    return true;
  }

  auto &context = module.getContext();
  userMain->setName("__mulberry_user_main");

  auto *voidType = llvm::Type::getVoidTy(context);
  auto *i64Type = llvm::Type::getInt64Ty(context);
  auto *runtimeInitType = llvm::FunctionType::get(voidType, false);
  auto runtimeInit = module.getOrInsertFunction("mulberry_runtime_init",
                                                runtimeInitType);
  auto *mainType = llvm::FunctionType::get(i64Type, false);
  auto *wrapperMain =
      llvm::Function::Create(mainType, llvm::GlobalValue::ExternalLinkage,
                             "main", module);
  auto *entry = llvm::BasicBlock::Create(context, "entry", wrapperMain);

  llvm::IRBuilder<> builder(entry);
  builder.CreateCall(runtimeInit);
  auto *result = builder.CreateCall(userMain);
  builder.CreateRet(result);
  return false;
}

auto findProgram(std::string_view programName) -> std::string {
  auto program = llvm::sys::findProgramByName(programName);
  if (!program)
    return {};
  return std::string(*program);
}

auto parentDir(llvm::StringRef filePath) -> std::string {
  llvm::SmallString<256> path(filePath);
  llvm::sys::path::remove_filename(path);
  return std::string(path.str());
}

auto basename(llvm::StringRef filePath) -> std::string {
  llvm::SmallString<256> path(filePath);
  return std::string(llvm::sys::path::filename(path));
}

auto appendPath(llvm::StringRef dir, llvm::StringRef filename)
    -> std::string {
  llvm::SmallString<256> path(dir);
  llvm::sys::path::append(path, filename);
  return std::string(path.str());
}

auto copyRuntimeLibrary(llvm::StringRef sourcePath,
                        llvm::StringRef outputDir) -> bool {
  auto targetPath = appendPath(outputDir, basename(sourcePath));
  if (sourcePath == targetPath)
    return false;

  llvm::sys::fs::remove(targetPath);
  if (auto ec = llvm::sys::fs::copy_file(sourcePath, targetPath)) {
    llvm::errs() << "error: failed to copy runtime library '" << sourcePath
                 << "' to '" << targetPath << "': " << ec.message() << "\n";
    return true;
  }
  return false;
}

auto copyBundledRuntimeLibraries(llvm::StringRef outputFileName,
                                 llvm::StringRef mulberryRuntimeLibPath,
                                 const std::vector<std::string> &mlirLibPaths,
                                 const std::vector<std::string> &boehmLibPaths)
    -> bool {
  auto outputDir = parentDir(outputFileName);
  if (outputDir.empty())
    outputDir = ".";

  if (copyRuntimeLibrary(mulberryRuntimeLibPath, outputDir))
    return true;

  for (auto &libPath : mlirLibPaths) {
    if (copyRuntimeLibrary(libPath, outputDir))
      return true;
  }

  for (auto &libPath : boehmLibPaths) {
    if (copyRuntimeLibrary(libPath, outputDir))
      return true;
  }

  return false;
}

auto addRPath(std::vector<std::string> &argStorage,
              std::vector<std::string> &rpaths,
              llvm::StringRef path) -> void {
  if (path.empty())
    return;
  std::string pathString(path);
  if (std::find(rpaths.begin(), rpaths.end(), pathString) != rpaths.end())
    return;
  rpaths.push_back(pathString);
  argStorage.push_back("-Wl,-rpath," + pathString);
}

auto linkExecutable(llvm::StringRef objectFileName,
                    llvm::StringRef outputFileName,
                    bool bundleRuntime) -> int {
  auto linker = findProgram("clang");
  if (linker.empty())
    linker = findProgram("cc");
  if (linker.empty()) {
    llvm::errs() << "error: unable to find clang or cc for executable linking\n";
    return EXIT_FAILURE;
  }

  auto mulberryRuntimeLibPath = getMulberryRuntimeLibPath();
  if (mulberryRuntimeLibPath.empty()) {
    llvm::errs() << "error: Mulberry runtime library path is not configured\n";
    return EXIT_FAILURE;
  }

  auto mlirRuntimeLibPath = getRuntimeLibPath();
  if (mlirRuntimeLibPath.empty()) {
    llvm::errs() << "error: MLIR runtime library path is not configured\n";
    return EXIT_FAILURE;
  }
  auto mlirRuntimeLibPaths = getMLIRRuntimeLibPaths();
  if (mlirRuntimeLibPaths.empty()) {
    llvm::errs() << "error: MLIR runtime library list is not configured\n";
    return EXIT_FAILURE;
  }

  auto runtimeDir = parentDir(mulberryRuntimeLibPath);
  auto mlirRuntimeDir = parentDir(mlirRuntimeLibPath);

  auto boehmLinkLibPath = getBoehmLinkLibPath();
  auto boehmRuntimeLibDir = getBoehmRuntimeLibDir();
  auto boehmRuntimeLibPaths = getBoehmRuntimeLibPaths();

  std::vector<std::string> argStorage = {
      linker,
      std::string(objectFileName),
      mulberryRuntimeLibPath,
  };
  if (bundleRuntime) {
    argStorage.push_back("-Wl,--no-as-needed");
    for (auto &libPath : mlirRuntimeLibPaths)
      argStorage.push_back(libPath);
    if (!boehmLinkLibPath.empty())
      argStorage.push_back(boehmLinkLibPath);
    argStorage.push_back("-Wl,--as-needed");
  } else {
    for (auto &libPath : mlirRuntimeLibPaths)
      argStorage.push_back(libPath);
    if (!boehmLinkLibPath.empty())
      argStorage.push_back(boehmLinkLibPath);
  }
  argStorage.push_back("-lm");
  if (bundleRuntime) {
    // Use old-style RPATH here so $ORIGIN also applies to runtime libraries'
    // transitive dependencies when the executable is moved with its bundle.
    argStorage.push_back("-Wl,-rpath,$ORIGIN");
    argStorage.push_back("-Wl,--disable-new-dtags");
  } else {
    std::vector<std::string> rpaths;
    addRPath(argStorage, rpaths, runtimeDir);
    addRPath(argStorage, rpaths, mlirRuntimeDir);
    for (auto &libPath : mlirRuntimeLibPaths)
      addRPath(argStorage, rpaths, parentDir(libPath));
    addRPath(argStorage, rpaths, boehmRuntimeLibDir);
    for (auto &libPath : boehmRuntimeLibPaths)
      addRPath(argStorage, rpaths, parentDir(libPath));
  }
  argStorage.push_back("-o");
  argStorage.push_back(std::string(outputFileName));

  std::vector<llvm::StringRef> args;
  for (auto &arg : argStorage)
    args.push_back(arg);

  int result = llvm::sys::ExecuteAndWait(linker, args);
  if (result != 0) {
    llvm::errs() << "error: executable linker failed with exit code " << result
                 << "\n";
    return EXIT_FAILURE;
  }

  if (bundleRuntime &&
      copyBundledRuntimeLibraries(outputFileName, mulberryRuntimeLibPath,
                                  mlirRuntimeLibPaths, boehmRuntimeLibPaths))
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}

} // namespace

static auto makeContext() -> mlir::MLIRContext {
  mlir::DialectRegistry registry;
  // `convert-to-llvm` finds lowering patterns through dialect interfaces.
  // Loading dialects alone is not enough; their ConvertToLLVM extensions must
  // also be registered before the context starts loading dialects.
  mlir::registerAllExtensions(registry);
  arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
  cf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  scf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  return mlir::MLIRContext(registry);
}

Compilation::Compilation() : _mlirContext{makeContext()} {}

auto Compilation::make(llvm::StringRef filename,
                       bool enableOpt) -> std::unique_ptr<Compilation> {
  auto compilation = std::make_unique<Compilation>();
  compilation->_mlirContext.getOrLoadDialect<mulberry_core::MulberryDialect>();
  compilation->_mlirContext.getOrLoadDialect<arith::ArithDialect>();
  compilation->_mlirContext.getOrLoadDialect<cf::ControlFlowDialect>();
  compilation->_mlirContext.getOrLoadDialect<func::FuncDialect>();
  compilation->_mlirContext.getOrLoadDialect<memref::MemRefDialect>();
  compilation->_mlirContext.getOrLoadDialect<scf::SCFDialect>();
  compilation->_mlirContext.getOrLoadDialect<linalg::LinalgDialect>();
  compilation->_mlirContext.getOrLoadDialect<math::MathDialect>();
  compilation->_mlirContext.getOrLoadDialect<LLVM::LLVMDialect>();
  compilation->_mlirContext.getOrLoadDialect<ptr::PtrDialect>();

  compilation->_inputFilename = std::string(filename);
  compilation->_enableOpt = enableOpt;
  return compilation;
}

auto Compilation::parse(std::unique_ptr<Module> &module) -> MulberryResult {
  _importAliases.clear();
  _loadedModules.clear();
  _usedBundledPackages.clear();

  if (parseFile(_inputFilename, llvm::SMLoc(), module))
    return failure();

  if (!isBundledPackage(module->packageName()) && loadPrelude(*module))
    return failure();

  return loadImports(*module);
}

auto Compilation::parseFile(const std::string &filename,
                            llvm::SMLoc includeLocation,
                            std::unique_ptr<Module> &module) -> MulberryResult {
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
  if (const char *envPath = std::getenv("MULBERRY_STDLIB_PATH")) {
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
  auto normalizedName = normalizeBundledImportName(moduleName);
  std::string relativePath(normalizedName);
  std::replace(relativePath.begin(), relativePath.end(), '.', '/');
  auto path = resolveStdlibPath(relativePath + ".mulberry");
  if (!llvm::sys::fs::exists(path))
    return {};
  return path;
}

auto Compilation::loadPrelude(Module &module) -> MulberryResult {
  std::unique_ptr<Module> preludeModule;
  auto preludePath = resolveStdlibPath("prelude.mulberry");
  if (parseFile(preludePath, module.location(), preludeModule) ||
      loadImports(*preludeModule))
    return failure();

  VectorUniquePtr<Decl> declarations = preludeModule->takeDeclarations();
  for (auto &decl : module.takeDeclarations())
    declarations.push_back(std::move(decl));

  module.setDeclarations(std::move(declarations));
  return success();
}

auto Compilation::loadImports(Module &module) -> MulberryResult {
  VectorUniquePtr<Decl> mergedDeclarations;
  for (auto &decl : module.takeDeclarations()) {
    if (auto *importDecl = llvm::dyn_cast<ImportDecl>(decl.get())) {
      auto importName =
          normalizeBundledImportName(importDecl->moduleName());
      if (isInternalBundledImport(importName) &&
          !isInternalSourceLocation(_sourceManager, importDecl->location())) {
        _sourceManager.PrintMessage(
            importDecl->location(), llvm::SourceMgr::DiagKind::DK_Error,
            "internal package is not available to user code");
        return failure();
      }

      auto segments = splitQualifiedName(importName);
      std::string moduleName;
      std::string importPath;
      std::string importedName;

      for (size_t moduleSize = segments.size(); moduleSize > 0; --moduleSize) {
        auto candidateModuleName = joinQualifiedName(segments, 0, moduleSize);
        auto candidatePath = resolveImportPath(candidateModuleName);
        if (candidatePath.empty())
          continue;

        moduleName = std::move(candidateModuleName);
        importPath = std::move(candidatePath);
        if (moduleSize < segments.size())
          importedName =
              joinQualifiedName(segments, moduleSize, segments.size());
        break;
      }

      if (moduleName.empty()) {
        llvm::errs() << "error: unable to resolve bundled import: '"
                     << importDecl->moduleName() << "'\n";
        return failure();
      }

      _importAliases[importAlias(moduleName)] = moduleName;
      if (!importedName.empty())
        _importAliases[importedName] = moduleName + "." + importedName;
      if (findBundledPackageSpec(moduleName))
        _usedBundledPackages.insert(moduleName);

      if (_loadedModules.insert(moduleName).second) {
        std::unique_ptr<Module> importedModule;
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

auto Compilation::loadBundledPackage(std::string_view moduleName)
    -> MulberryResult {
  auto spec = findBundledPackageSpec(moduleName);
  if (!spec)
    return success();

  std::string packageName(moduleName);
  if (_loadedBundledPackages.count(packageName))
    return success();

  auto packagePath = spec->libraryPath;
  if (packagePath.empty()) {
    llvm::errs() << "error: bundled package path is not configured for '"
                 << packageName << "'\n";
    return failure();
  }

  auto dialectPlugin = mlir::DialectPlugin::load(packagePath);
  if (!dialectPlugin) {
    llvm::errs() << "error: failed to load bundled dialect package '"
                 << packageName << "': "
                 << llvm::toString(dialectPlugin.takeError()) << "\n";
    return failure();
  }

  mlir::DialectRegistry registry;
  dialectPlugin->registerDialectRegistryCallbacks(registry);
  _mlirContext.appendDialectRegistry(registry);

  auto passPlugin = mlir::PassPlugin::load(packagePath);
  if (!passPlugin) {
    llvm::errs() << "error: failed to load bundled pass package '"
                 << packageName << "': "
                 << llvm::toString(passPlugin.takeError()) << "\n";
    return failure();
  }

  passPlugin->registerPassRegistryCallbacks();
  _loadedBundledPackages.insert(packageName);
  return success();
}

auto Compilation::loadUsedBundledPackages() -> MulberryResult {
  for (auto &moduleName : _usedBundledPackages) {
    if (loadBundledPackage(moduleName))
      return failure();
  }
  return success();
}

auto Compilation::addBundledPackagePreCorePipelines(mlir::PassManager &pm)
    -> MulberryResult {
  for (auto &moduleName : _usedBundledPackages) {
    auto spec = findBundledPackageSpec(moduleName);
    if (!spec || spec->preCorePipeline.empty())
      continue;

    if (addPassPipeline(pm, spec->preCorePipeline))
      return failure();
  }
  return success();
}

auto Compilation::addBundledPackagePostCorePipelines(mlir::PassManager &pm)
    -> MulberryResult {
  for (auto &moduleName : _usedBundledPackages) {
    auto spec = findBundledPackageSpec(moduleName);
    if (!spec || spec->postCorePipeline.empty())
      continue;

    if (addPassPipeline(pm, spec->postCorePipeline))
      return failure();
    if (spec->runSymbolDCEAfterPostCore)
      pm.addPass(mlir::createSymbolDCEPass());
  }
  return success();
}

auto Compilation::addPassPipeline(mlir::PassManager &pm,
                                  llvm::StringRef pipeline) -> MulberryResult {
  if (mlir::failed(mlir::parsePassPipeline(pipeline, pm, llvm::errs())))
    return failure();
  return success();
}

auto Compilation::genMLIR(mlir::OwningOpRef<mlir::ModuleOp> &module,
                          Lowering lowering) -> MulberryResult {
  std::unique_ptr<Module> moduleAST;
  if (parse(moduleAST))
    return failure();

  if (mulberry::sema(_sourceManager, *moduleAST.get(), _importAliases) ||
      mlirGen(_sourceManager, _mlirContext, *moduleAST, module))
    return failure();

  mlir::PassManager pm(module.get()->getName());
  mlir::OpPassManager &optPM = pm.nest<func::FuncOp>();
  if (_enableOpt)
    optPM.addPass(mlir::createCanonicalizerPass());

  if (lowering >= Lowering::Mulberry && loadUsedBundledPackages())
    return failure();

  if (lowering >= Lowering::Mulberry &&
      addBundledPackagePreCorePipelines(pm))
    return failure();

  if (lowering >= Lowering::Mulberry)
    pm.addPass(mulberry_core::createLowerResultTry());

  if (lowering >= Lowering::Mulberry)
    pm.addPass(mulberry_core::createLowerMulberry());

  if (lowering >= Lowering::Mulberry &&
      addBundledPackagePostCorePipelines(pm))
    return failure();

  if (lowering >= Lowering::Mulberry && !_usedBundledPackages.empty())
    bufferization::buildBufferDeallocationPipeline(pm);

  if (lowering >= Lowering::Mulberry)
    pm.addPass(mulberry_core::createFinalizeMulberryTensorStorage());

  if (lowering >= Lowering::LLVM) {
    pm.addNestedPass<func::FuncOp>(
        mlir::createConvertLinalgToLoopsPass());
    pm.addPass(mlir::createLowerAffinePass());
    pm.addPass(mlir::createSCFToControlFlowPass());
    pm.addNestedPass<func::FuncOp>(LLVM::createLLVMRequestCWrappersPass());
    pm.addPass(mulberry_core::createConvertMulberryToLLVM());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  }

  return pm.run(*module);
}

auto Compilation::typecheck() -> int {
  std::unique_ptr<Module> module;
  if (parse(module) ||
      mulberry::sema(_sourceManager, *module.get(), _importAliases))
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
  auto mulberryRuntimeLibPath = getMulberryRuntimeLibPath();
  std::vector<llvm::StringRef> sharedLibPaths;
  if (!runtimeLibPath.empty())
    sharedLibPaths.push_back(runtimeLibPath);
  if (!mulberryRuntimeLibPath.empty())
    sharedLibPaths.push_back(mulberryRuntimeLibPath);
  options.sharedLibPaths = sharedLibPaths;

  auto engine = mlir::ExecutionEngine::create(*module, options,
                                              std::move(*targetMachine));
  if (!engine) {
    llvm::errs() << "error: failed to create execution engine: "
                 << llvm::toString(engine.takeError()) << "\n";
    return EXIT_FAILURE;
  }

  (*engine)->initialize();
  if (auto error = initializeMulberryRuntime(**engine)) {
    llvm::errs() << "error: failed to initialize Mulberry runtime: "
                 << llvm::toString(std::move(error)) << "\n";
    return EXIT_FAILURE;
  }

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
  return genObjectFile(outputFileName, false);
}

auto Compilation::genObjectFile(const char *outputFileName,
                                bool addExecutableWrapper) -> int {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (genMLIR(module, Lowering::LLVM))
    return EXIT_FAILURE;

  auto targetMachine = createTargetMachine();
  if (!targetMachine) {
    llvm::errs() << "error: failed to create target machine: "
                 << llvm::toString(targetMachine.takeError()) << "\n";
    return EXIT_FAILURE;
  }

  llvm::LLVMContext llvmContext;
  auto llvmModule =
      genLLVMModule(*module, _enableOpt, *targetMachine, llvmContext);
  if (!llvmModule)
    return EXIT_FAILURE;

  if (addExecutableWrapper && addRuntimeMainWrapper(*llvmModule))
    return EXIT_FAILURE;

  if (createLLVMObjectFile(*llvmModule, **targetMachine, outputFileName))
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}

auto Compilation::genExecutable(const char *outputFileName,
                                bool bundleRuntime) -> int {
  llvm::SmallString<128> objectFileName;
  int objectFd = -1;
  if (auto ec =
          llvm::sys::fs::createTemporaryFile("mulberry-aot", "o", objectFd,
                                             objectFileName)) {
    llvm::errs() << "error: failed to create temporary object file: "
                 << ec.message() << "\n";
    return EXIT_FAILURE;
  }
  llvm::raw_fd_ostream objectStream(objectFd, true);
  objectStream.close();
  llvm::FileRemover removeObjectFile(objectFileName);

  if (genObjectFile(objectFileName.c_str(), true))
    return EXIT_FAILURE;

  return linkExecutable(objectFileName, outputFileName, bundleRuntime);
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

  mulberry::dumpAST(_sourceManager, *module);
  return EXIT_SUCCESS;
}

auto Compilation::dumpAST() -> int {
  std::unique_ptr<Module> module;
  if (parse(module) ||
      mulberry::sema(_sourceManager, *module.get(), _importAliases))
    return EXIT_FAILURE;

  mulberry::dumpAST(_sourceManager, *module);
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
  llvm::InitializeNativeTargetAsmParser();

  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (genMLIR(module, Lowering::LLVM))
    return EXIT_FAILURE;

  auto targetMachine = createTargetMachine();
  if (!targetMachine) {
    llvm::errs() << "error: failed to create target machine: "
                 << llvm::toString(targetMachine.takeError()) << "\n";
    return EXIT_FAILURE;
  }

  llvm::LLVMContext llvmContext;
  auto llvmModule =
      genLLVMModule(*module, _enableOpt, *targetMachine, llvmContext);
  if (!llvmModule)
    return EXIT_FAILURE;

  llvm::outs() << *llvmModule << "\n";
  return EXIT_SUCCESS;
}
