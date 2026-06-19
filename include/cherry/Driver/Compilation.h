//===--- Compilation.h - Compilation Task Data Structure --------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_COMPILATION_H
#define CHERRY_COMPILATION_H

#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include <memory>
#include <map>
#include <set>
#include <string>

namespace mlir {
template <typename OpTy> class OwningOpRef;
class ModuleOp;
} // end namespace mlir

namespace llvm {
} // end namespace llvm

namespace cherry {
class Module;
class CherryResult;

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

  auto sourceManager() -> llvm::SourceMgr & { return _sourceManager; };

private:
  llvm::SourceMgr _sourceManager;
  bool _enableOpt;
  mlir::MLIRContext _mlirContext;
  std::string _inputFilename;
  std::map<std::string, std::string> _importAliases;
  std::set<std::string> _loadedModules;

  auto parse(std::unique_ptr<Module> &module) -> CherryResult;
  auto parseFile(const std::string &filename, llvm::SMLoc includeLocation,
                 std::unique_ptr<Module> &module) -> CherryResult;
  auto loadPrelude(Module &module) -> CherryResult;
  auto loadImports(Module &module) -> CherryResult;
  auto resolveStdlibPath(std::string_view relativePath) -> std::string;
  auto resolveImportPath(std::string_view moduleName) -> std::string;
  auto genMLIR(mlir::OwningOpRef<mlir::ModuleOp> &module, Lowering lowering)
      -> CherryResult;
};

} // end namespace cherry

#endif // CHERRY_COMPILATION_H
