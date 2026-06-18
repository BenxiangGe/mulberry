//===--- Sema.h - Cherry Semantic Analysis ----------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_SEMA_H
#define CHERRY_SEMA_H

#include <map>
#include <string>
#include <string_view>

namespace llvm {
class SourceMgr;
} // end namespace llvm

namespace cherry {
class Module;
class CherryResult;

auto sema(const llvm::SourceMgr &sourceManager, Module &moduleAST)
    -> CherryResult;
auto sema(const llvm::SourceMgr &sourceManager, Module &moduleAST,
          const std::map<std::string, std::string> &importAliases)
    -> CherryResult;

} // end namespace cherry

#endif // CHERRY_SEMA_H
