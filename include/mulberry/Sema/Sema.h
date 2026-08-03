//===--- Sema.h - Mulberry Semantic Analysis ----------------------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_SEMA_H
#define MULBERRY_SEMA_H

#include "mulberry/AST/Module.h"
#include "llvm/Support/LogicalResult.h"
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace llvm {
class SourceMgr;
} // end namespace llvm

namespace mulberry {
class Module;
class SemaImpl;

class SemaSession {
public:
  SemaSession(const llvm::SourceMgr &sourceManager,
              const std::map<std::string, std::string> &importAliases);
  ~SemaSession();

  auto sema(Module &module, bool checkReplResult = true)
      -> llvm::LogicalResult;
  auto retainDeclarations(VectorUniquePtr<Decl> declarations) -> void;
  auto commitReplSubmission() -> void;
  auto rollbackReplSubmission() -> void;
  auto completionNames() const -> std::vector<std::string>;
  auto completionMembers(std::string_view receiver) const
      -> std::vector<std::string>;

private:
  struct PendingState;

  std::unique_ptr<SemaImpl> _impl;
  VectorUniquePtr<Decl> _retainedDeclarations;
  std::unique_ptr<PendingState> _pendingState;
};

auto sema(const llvm::SourceMgr &sourceManager, Module &moduleAST)
    -> llvm::LogicalResult;
auto sema(const llvm::SourceMgr &sourceManager, Module &moduleAST,
          const std::map<std::string, std::string> &importAliases)
    -> llvm::LogicalResult;

} // end namespace mulberry

#endif // MULBERRY_SEMA_H
