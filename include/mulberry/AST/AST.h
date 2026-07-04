//===--- AST.h - AST nodes and AST Dumper -----------------------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_AST_H
#define MULBERRY_AST_H

#include "mulberry/AST/AssignExpr.h"
#include "mulberry/AST/BinaryExpr.h"
#include "mulberry/AST/Decl.h"
#include "mulberry/AST/Expr.h"
#include "mulberry/AST/HeapExpr.h"
#include "mulberry/AST/Identifier.h"
#include "mulberry/AST/LiteralExpr.h"
#include "mulberry/AST/Module.h"
#include "mulberry/AST/Name.h"
#include "mulberry/AST/NameExpr.h"
#include "mulberry/AST/Stat.h"
#include "mulberry/AST/StructLiteralExpr.h"
#include "mulberry/AST/CollectionExpr.h"
#include "mulberry/AST/Type.h"
#include "mulberry/AST/TypeLayoutExpr.h"
#include "mlir/IR/Location.h"
#include "llvm/Support/SourceMgr.h"

namespace mulberry {

auto dumpAST(const llvm::SourceMgr &sourceManager, const Module &module)
    -> void;
} // end namespace mulberry

#endif // MULBERRY_AST_H
