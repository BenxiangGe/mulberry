//===--- Sema.h - Mulberry Semantic Analysis ----------------------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_SEMA_H
#define MULBERRY_SEMA_H

#include "llvm/Support/LogicalResult.h"
#include <map>
#include <string>
#include <string_view>

namespace llvm {
class SourceMgr;
} // end namespace llvm

namespace mulberry {
class Module;

auto sema(const llvm::SourceMgr &sourceManager, Module &moduleAST)
    -> llvm::LogicalResult;
auto sema(const llvm::SourceMgr &sourceManager, Module &moduleAST,
          const std::map<std::string, std::string> &importAliases)
    -> llvm::LogicalResult;

} // end namespace mulberry

#endif // MULBERRY_SEMA_H
