//===--- Compilation.h - Compilation Task Data Structure --------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_COMPILATION_H
#define MULBERRY_COMPILATION_H

#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include <memory>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace mlir {
template <typename OpTy> class OwningOpRef;
class ModuleOp;
class PassManager;
} // end namespace mlir

namespace llvm {
class MemoryBuffer;
} // end namespace llvm

namespace mulberry {
class Module;
class SemaSession;

class Compilation {
public:
  enum Lowering { None, SCF, ArithCfFunc, Mulberry, LLVM };

  Compilation();
  ~Compilation();

  static auto make(llvm::StringRef filename, bool enableOpt)
      -> std::unique_ptr<Compilation>;

  auto dumpTokens() -> int;
  auto dumpParse() -> int;
  auto dumpAST() -> int;
  auto dumpMLIR(Lowering lowering) -> int;
  auto dumpLLVM() -> int;

  auto compileSource(llvm::StringRef source, llvm::StringRef sourceName,
                     mlir::OwningOpRef<mlir::ModuleOp> &module,
                     Lowering lowering) -> llvm::LogicalResult;
  auto compileReplSource(llvm::StringRef source, llvm::StringRef sourceName,
                         llvm::StringRef functionName,
                         mlir::OwningOpRef<mlir::ModuleOp> &module,
                         Lowering lowering) -> llvm::LogicalResult;
  auto replTypeOf(llvm::StringRef source, llvm::StringRef sourceName,
                  std::string &typeName) -> llvm::LogicalResult;
  auto commitReplSubmission() -> void;
  auto rollbackReplSubmission() -> void;
  auto replCompletionNames() const -> std::vector<std::string>;
  auto replCompletionMembers(std::string_view receiver) const
      -> std::vector<std::string>;

  auto typecheck() -> int;
  auto jit() -> int;
  auto genObjectFile(const char *outputFileName) -> int;
  auto genExecutable(const char *outputFileName, bool bundleRuntime) -> int;

  auto sourceManager() -> llvm::SourceMgr & { return _sourceManager; };

private:
  struct ReplImportState {
    std::map<std::string, std::string> importAliases;
    std::set<std::string> loadedModules;
    std::set<std::string> usedBundledPackages;
    std::string localImportRoot;
    bool replPrepared = false;
  };

  llvm::SourceMgr _sourceManager;
  bool _enableOpt;
  mlir::MLIRContext _mlirContext;
  std::string _inputFilename;
  std::map<std::string, std::string> _importAliases;
  std::set<std::string> _loadedModules;
  std::set<std::string> _usedBundledPackages;
  std::set<std::string> _loadedBundledPackages;
  std::string _localImportRoot;
  std::unique_ptr<SemaSession> _replSema;
  std::optional<ReplImportState> _pendingReplImportState;
  bool _replPrepared = false;

  auto parse(std::unique_ptr<Module> &module) -> llvm::LogicalResult;
  auto parseSource(llvm::StringRef source, llvm::StringRef sourceName,
                   std::unique_ptr<Module> &module) -> llvm::LogicalResult;
  auto parseReplSource(llvm::StringRef source, llvm::StringRef sourceName,
                       llvm::StringRef functionName,
                       std::unique_ptr<Module> &module)
      -> llvm::LogicalResult;
  auto parseFile(const std::string &filename, llvm::SMLoc includeLocation,
                 std::unique_ptr<Module> &module) -> llvm::LogicalResult;
  auto parseBuffer(std::unique_ptr<llvm::MemoryBuffer> buffer,
                   llvm::SMLoc includeLocation,
                   std::unique_ptr<Module> &module) -> llvm::LogicalResult;
  auto prepareModule(std::unique_ptr<Module> &module) -> llvm::LogicalResult;
  auto loadPrelude(Module &module) -> llvm::LogicalResult;
  auto loadImports(Module &module) -> llvm::LogicalResult;
  auto resolveStdlibPath(std::string_view relativePath) -> std::string;
  auto resolveBundledImportPath(std::string_view importName) -> std::string;
  auto resolveLocalImportPath(std::string_view moduleName) -> std::string;
  auto setLocalImportRoot(llvm::SMLoc location) -> void;
  auto loadBundledPackage(std::string_view moduleName) -> llvm::LogicalResult;
  auto loadUsedBundledPackages() -> llvm::LogicalResult;
  auto addBundledPackagePreCorePipelines(mlir::PassManager &pm)
      -> llvm::LogicalResult;
  auto addBundledPackagePostCorePipelines(mlir::PassManager &pm)
      -> llvm::LogicalResult;
  auto addPassPipeline(mlir::PassManager &pm, llvm::StringRef pipeline)
      -> llvm::LogicalResult;
  auto compileModule(Module &module,
                     mlir::OwningOpRef<mlir::ModuleOp> &compiledModule,
                     Lowering lowering) -> llvm::LogicalResult;
  auto genMLIR(mlir::OwningOpRef<mlir::ModuleOp> &module, Lowering lowering)
      -> llvm::LogicalResult;
  auto genObjectFile(const char *outputFileName, bool addExecutableWrapper)
      -> int;
};

} // end namespace mulberry

#endif // MULBERRY_COMPILATION_H
