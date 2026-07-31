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

namespace mlir {
template <typename OpTy> class OwningOpRef;
class ModuleOp;
class PassManager;
} // end namespace mlir

namespace llvm {
} // end namespace llvm

namespace mulberry {
class Module;

class Compilation {
public:
  enum Lowering { None, SCF, ArithCfFunc, Mulberry, LLVM };

  Compilation();

  static auto make(llvm::StringRef filename, bool enableOpt)
      -> std::unique_ptr<Compilation>;

  auto dumpTokens() -> int;
  auto dumpParse() -> int;
  auto dumpAST() -> int;
  auto dumpMLIR(Lowering lowering) -> int;
  auto dumpLLVM() -> int;

  auto typecheck() -> int;
  auto jit() -> int;
  auto genObjectFile(const char *outputFileName) -> int;
  auto genExecutable(const char *outputFileName, bool bundleRuntime) -> int;

  auto sourceManager() -> llvm::SourceMgr & { return _sourceManager; };

private:
  llvm::SourceMgr _sourceManager;
  bool _enableOpt;
  mlir::MLIRContext _mlirContext;
  std::string _inputFilename;
  std::map<std::string, std::string> _importAliases;
  std::set<std::string> _loadedModules;
  std::set<std::string> _usedBundledPackages;
  std::set<std::string> _loadedBundledPackages;

  auto parse(std::unique_ptr<Module> &module) -> llvm::LogicalResult;
  auto parseFile(const std::string &filename, llvm::SMLoc includeLocation,
                 std::unique_ptr<Module> &module) -> llvm::LogicalResult;
  auto loadPrelude(Module &module) -> llvm::LogicalResult;
  auto loadImports(Module &module) -> llvm::LogicalResult;
  auto resolveStdlibPath(std::string_view relativePath) -> std::string;
  auto resolveImportPath(std::string_view moduleName) -> std::string;
  auto loadBundledPackage(std::string_view moduleName) -> llvm::LogicalResult;
  auto loadUsedBundledPackages() -> llvm::LogicalResult;
  auto addBundledPackagePreCorePipelines(mlir::PassManager &pm)
      -> llvm::LogicalResult;
  auto addBundledPackagePostCorePipelines(mlir::PassManager &pm)
      -> llvm::LogicalResult;
  auto addPassPipeline(mlir::PassManager &pm, llvm::StringRef pipeline)
      -> llvm::LogicalResult;
  auto genMLIR(mlir::OwningOpRef<mlir::ModuleOp> &module, Lowering lowering)
      -> llvm::LogicalResult;
  auto genObjectFile(const char *outputFileName, bool addExecutableWrapper)
      -> int;
};

} // end namespace mulberry

#endif // MULBERRY_COMPILATION_H
