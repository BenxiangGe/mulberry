//===--- AST.h - AST nodes and AST Dumper -----------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_AST_H
#define CHERRY_AST_H

#include "cherry/AST/AssignExpr.h"
#include "cherry/AST/BinaryExpr.h"
#include "cherry/AST/Decl.h"
#include "cherry/AST/Expr.h"
#include "cherry/AST/Identifier.h"
#include "cherry/AST/ListExpr.h"
#include "cherry/AST/LiteralExpr.h"
#include "cherry/AST/Module.h"
#include "cherry/AST/Name.h"
#include "cherry/AST/NameExpr.h"
#include "cherry/AST/Stat.h"
#include "cherry/AST/StructLiteralExpr.h"
#include "cherry/AST/TensorExpr.h"
#include "cherry/AST/Type.h"
#include "mlir/IR/Location.h"
#include "llvm/Support/SourceMgr.h"

namespace cherry {

auto dumpAST(const llvm::SourceMgr &sourceManager, const Module &module)
    -> void;
} // end namespace cherry

#endif // CHERRY_AST_H
