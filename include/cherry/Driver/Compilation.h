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
#include "llvm/Support/SourceMgr.h"

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
  enum Lowering { None, SCF, ArithCfFunc, Linalg, LLVM };

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

  auto parse(std::unique_ptr<Module> &module) -> CherryResult;
  auto genMLIR(mlir::OwningOpRef<mlir::ModuleOp> &module, Lowering lowering)
      -> CherryResult;
};

} // end namespace cherry

#endif // CHERRY_COMPILATION_H
