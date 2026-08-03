//===--- JitSession.h - Persistent Mulberry JIT Session --------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_JIT_SESSION_H
#define MULBERRY_JIT_SESSION_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace llvm {
namespace orc {
class JITDylib;
class LLJIT;
} // namespace orc
class TargetMachine;
} // namespace llvm

namespace mulberry {

class JitSession {
public:
  static auto create(bool enableOpt)
      -> llvm::Expected<std::unique_ptr<JitSession>>;

  ~JitSession();

  auto addModule(mlir::ModuleOp module,
                 llvm::StringRef moduleName = {}) -> llvm::Error;
  auto initialize(llvm::StringRef moduleName = {}) -> llvm::Error;
  auto invoke(llvm::StringRef moduleName, llvm::StringRef functionName,
              uint64_t &result) -> llvm::Error;

private:
  JitSession(std::unique_ptr<llvm::orc::LLJIT> jit,
             std::unique_ptr<llvm::TargetMachine> targetMachine,
             bool enableOpt, std::vector<std::string> sharedLibraryPaths);

  auto addRuntimeGenerators(llvm::orc::JITDylib &jitDylib) -> llvm::Error;
  auto discardSubmission(llvm::orc::JITDylib &jitDylib) -> llvm::Error;
  auto findJitDylib(llvm::StringRef moduleName)
      -> llvm::orc::JITDylib *;

  std::unique_ptr<llvm::orc::LLJIT> _jit;
  std::unique_ptr<llvm::TargetMachine> _targetMachine;
  bool _enableOpt;
  std::vector<std::string> _sharedLibraryPaths;
  std::vector<llvm::orc::JITDylib *> _submissionDylibs;
  llvm::orc::JITDylib *_pendingSubmissionDylib = nullptr;
};

} // namespace mulberry

#endif // MULBERRY_JIT_SESSION_H
