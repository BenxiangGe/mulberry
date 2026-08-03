//===--- JitSession.cpp - Persistent Mulberry JIT Session ----------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "mulberry/Driver/JitSession.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/TargetSelect.h"
#include <cstddef>
#include <string_view>
#include <utility>

namespace {

auto splitRuntimeLibraryPaths(std::string_view paths)
    -> std::vector<std::string> {
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

auto runtimeLibraryPaths() -> std::vector<std::string> {
#ifdef MULBERRY_MLIR_RUNTIME_LIBS
  return splitRuntimeLibraryPaths(MULBERRY_MLIR_RUNTIME_LIBS);
#else
  return {};
#endif
}

} // namespace

namespace mulberry {

JitSession::JitSession(std::unique_ptr<llvm::orc::LLJIT> jit,
                       std::unique_ptr<llvm::TargetMachine> targetMachine,
                       bool enableOpt,
                       std::vector<std::string> sharedLibraryPaths)
    : _jit{std::move(jit)}, _targetMachine{std::move(targetMachine)},
      _enableOpt{enableOpt},
      _sharedLibraryPaths{std::move(sharedLibraryPaths)} {}

JitSession::~JitSession() = default;

auto JitSession::create(bool enableOpt)
    -> llvm::Expected<std::unique_ptr<JitSession>> {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  auto targetMachineBuilder = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!targetMachineBuilder)
    return targetMachineBuilder.takeError();

  targetMachineBuilder->setCodeGenOptLevel(
      enableOpt ? llvm::CodeGenOptLevel::Aggressive
                : llvm::CodeGenOptLevel::None);
  auto targetMachine = targetMachineBuilder->createTargetMachine();
  if (!targetMachine)
    return targetMachine.takeError();

  auto jit = llvm::orc::LLJITBuilder()
                 .setJITTargetMachineBuilder(std::move(*targetMachineBuilder))
                 .setNumCompileThreads(0)
                 .create();
  if (!jit)
    return jit.takeError();

  auto session = std::unique_ptr<JitSession>(new JitSession(
      std::move(*jit), std::move(*targetMachine), enableOpt,
      runtimeLibraryPaths()));
  if (auto error = session->addRuntimeGenerators(
          session->_jit->getMainJITDylib()))
    return std::move(error);

  return session;
}

auto JitSession::addRuntimeGenerators(llvm::orc::JITDylib &jitDylib)
    -> llvm::Error {
  auto globalPrefix = _jit->getDataLayout().getGlobalPrefix();

  // The driver embeds Mulberry runtime objects and exports them from the host
  // process. ORC must be told to search that process for generated externs.
  auto processSymbols =
      llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          globalPrefix);
  if (!processSymbols)
    return processSymbols.takeError();
  jitDylib.addGenerator(std::move(*processSymbols));

  for (const auto &path : _sharedLibraryPaths) {
    auto librarySymbols =
        llvm::orc::DynamicLibrarySearchGenerator::Load(path.c_str(),
                                                       globalPrefix);
    if (!librarySymbols)
      return librarySymbols.takeError();
    jitDylib.addGenerator(std::move(*librarySymbols));
  }

  return llvm::Error::success();
}

auto JitSession::findJitDylib(llvm::StringRef moduleName)
    -> llvm::orc::JITDylib * {
  if (moduleName.empty())
    return &_jit->getMainJITDylib();
  return _jit->getJITDylibByName(moduleName);
}

auto JitSession::discardSubmission(llvm::orc::JITDylib &jitDylib)
    -> llvm::Error {
  if (&jitDylib == &_jit->getMainJITDylib())
    return llvm::Error::success();

  if (_pendingSubmissionDylib == &jitDylib)
    _pendingSubmissionDylib = nullptr;
  for (auto it = _submissionDylibs.begin(); it != _submissionDylibs.end();
       ++it) {
    if (*it != &jitDylib)
      continue;
    _submissionDylibs.erase(it);
    break;
  }

  return _jit->getExecutionSession().removeJITDylib(jitDylib);
}

auto JitSession::addModule(mlir::ModuleOp module, llvm::StringRef moduleName)
    -> llvm::Error {
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  auto context = std::make_unique<llvm::LLVMContext>();
  auto llvmModule = mlir::translateModuleToLLVMIR(module, *context);
  if (!llvmModule)
    return llvm::make_error<llvm::StringError>(
        "could not convert MLIR module to LLVM IR",
        llvm::inconvertibleErrorCode());

  llvmModule->setTargetTriple(_jit->getTargetTriple());
  llvmModule->setDataLayout(_jit->getDataLayout());

  if (_enableOpt) {
    auto transformer = mlir::makeOptimizingTransformer(
        3, 0, _targetMachine.get());
    if (auto error = transformer(llvmModule.get()))
      return error;
  }

  auto *jitDylib = &_jit->getMainJITDylib();
  if (!moduleName.empty()) {
    auto newJitDylib = _jit->createJITDylib(moduleName.str());
    if (!newJitDylib)
      return newJitDylib.takeError();
    jitDylib = &*newJitDylib;
    _pendingSubmissionDylib = jitDylib;
    if (auto error = addRuntimeGenerators(*jitDylib)) {
      auto cleanupError = discardSubmission(*jitDylib);
      if (cleanupError)
        return llvm::joinErrors(std::move(error), std::move(cleanupError));
      return error;
    }

    // Keep each submission's definitions local first, then search older
    // submissions. This is the symbol visibility boundary for the later
    // source-binding/runtime-slot work; it avoids rebuilding old source.
    for (auto *previous : _submissionDylibs)
      jitDylib->addToLinkOrder(*previous);
    jitDylib->addToLinkOrder(_jit->getMainJITDylib());
  }

  auto error = _jit->addIRModule(
      *jitDylib,
      llvm::orc::ThreadSafeModule(std::move(llvmModule), std::move(context)));
  if (error && _pendingSubmissionDylib == jitDylib) {
    auto cleanupError = discardSubmission(*jitDylib);
    if (cleanupError)
      return llvm::joinErrors(std::move(error), std::move(cleanupError));
  }
  return error;
}

auto JitSession::initialize(llvm::StringRef moduleName) -> llvm::Error {
  auto *jitDylib = findJitDylib(moduleName);
  if (jitDylib == nullptr)
    return llvm::make_error<llvm::StringError>(
        "unknown JIT module '" + moduleName.str() + "'",
        llvm::inconvertibleErrorCode());
  auto error = _jit->initialize(*jitDylib);
  if (error) {
    if (_pendingSubmissionDylib == jitDylib) {
      auto cleanupError = discardSubmission(*jitDylib);
      if (cleanupError)
        return llvm::joinErrors(std::move(error), std::move(cleanupError));
    }
    return error;
  }

  if (moduleName.empty() || _pendingSubmissionDylib != jitDylib)
    return llvm::Error::success();

  _submissionDylibs.push_back(jitDylib);
  _pendingSubmissionDylib = nullptr;
  return llvm::Error::success();
}

auto JitSession::invoke(llvm::StringRef moduleName,
                        llvm::StringRef functionName, uint64_t &result)
    -> llvm::Error {
  auto *jitDylib = findJitDylib(moduleName);
  if (jitDylib == nullptr)
    return llvm::make_error<llvm::StringError>(
        "unknown JIT module '" + moduleName.str() + "'",
        llvm::inconvertibleErrorCode());

  auto address = _jit->lookup(*jitDylib, "_mlir_ciface_" + functionName.str());
  if (!address) {
    auto error = address.takeError();
    if (!moduleName.empty()) {
      auto cleanupError = discardSubmission(*jitDylib);
      if (cleanupError)
        return llvm::joinErrors(std::move(error), std::move(cleanupError));
    }
    return error;
  }

  // Scalar UInt64 results use a direct C wrapper in MLIR 22. The packed
  // wrapper ABI is only needed for aggregate results or arguments.
  using Function = uint64_t (*)();
  result = address->toPtr<Function>()();
  return llvm::Error::success();
}

} // namespace mulberry
