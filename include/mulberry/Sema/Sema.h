//===--- Sema.h - Mulberry Semantic Analysis ----------------------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_SEMA_H
#define MULBERRY_SEMA_H

#include <map>
#include <string>
#include <string_view>

namespace llvm {
class SourceMgr;
} // end namespace llvm

namespace mulberry {
class Module;
class MulberryResult;

auto sema(const llvm::SourceMgr &sourceManager, Module &moduleAST)
    -> MulberryResult;
auto sema(const llvm::SourceMgr &sourceManager, Module &moduleAST,
          const std::map<std::string, std::string> &importAliases)
    -> MulberryResult;

} // end namespace mulberry

#endif // MULBERRY_SEMA_H
