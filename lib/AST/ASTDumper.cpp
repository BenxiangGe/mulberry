//===--- ASTDumper.cpp - Cherry Language AST Dumper
//------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/AST/AST.h"
#include "cherry/Basic/Types.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <string_view>

using namespace cherry;

namespace {
using llvm::cast;
using llvm::dyn_cast;
using llvm::errs;

class Dumper {
public:
  Dumper(const llvm::SourceMgr &sourceManager)
      : _sourceManager{sourceManager} {}

  auto dump(const Module *node) -> void;

private:
  int _curIndent = 0;
  const llvm::SourceMgr &_sourceManager;

  // Declarations
  auto dump(const Decl *node) -> void;
  auto dump(const Prototype *node) -> void;
  auto dump(const FunctionDecl *node) -> void;
  auto dump(const StructDecl *node) -> void;

  // Expressions
  auto dump(const Expr *node) -> void;
  auto dump(const UnitExpr *node) -> void;
  auto dump(const BlockExpr *node, std::string_view string) -> void;
  auto dump(const CallExpr *node) -> void;
  auto dump(const StructLiteralExpr *node) -> void;
  auto dump(const VariableExpr *node) -> void;
  auto dump(const MemberExpr *node) -> void;
  auto dump(const AssignExpr *node) -> void;
  auto dump(const DecimalLiteralExpr *node) -> void;
  auto dump(const FloatLiteralExpr *node) -> void;
  auto dump(const BoolLiteralExpr *node) -> void;
  auto dump(const ListLiteralExpr *node) -> void;
  auto dump(const ListAccessExpr *node) -> void;
  auto dump(const BinaryExpr *node) -> void;
  auto dump(const IfExpr *node) -> void;
  auto dump(const WhileExpr *node) -> void;

  // Statements
  auto dump(const Stat *node) -> void;
  auto dump(const VariableStat *node) -> void;
  auto dump(const ExprStat *node) -> void;

  // Utility
  auto indent() -> void {
    for (int i = 0; i < _curIndent; i++)
      errs() << "  ";
  }

  template <typename T> auto loc(T *node) -> std::string {
    auto [line, col] = _sourceManager.getLineAndColumn(node->location());
    return "loc=" + std::to_string(line) + ":" + std::to_string(col);
  }

  auto formatTypeNode(const TypeNode *node) -> std::string {
    if (dyn_cast<UnitTypeNode>(node))
      return "()";

    if (auto *listType = dyn_cast<ListTypeNode>(node)) {
      std::string result = formatTypeNode(listType->elementTypeNode());
      result += "[";
      std::string separator;
      for (auto dim : listType->shape()) {
        result += separator;
        result += dim < 0 ? "?" : std::to_string(dim);
        separator = ", ";
      }
      result += "]";
      return result;
    }

    auto *namedType = cast<NamedTypeNode>(node);
    return std::string(namedType->name());
  }
};

struct Indent {
  Indent(int &level) : level(level) { ++level; }
  ~Indent() { --level; }
  int &level;
};

#define INDENT()                                                               \
  Indent level_(_curIndent);                                                   \
  indent();

} // end namespace

auto Dumper::dump(const Module *node) -> void {
  for (auto &decl : *node)
    dump(decl.get());
}

auto Dumper::dump(const Decl *node) -> void {
  llvm::TypeSwitch<const Decl *>(node)
      .Case<FunctionDecl, StructDecl>([&](auto *node) { this->dump(node); })
      .Default(
          [&](const Decl *) { llvm_unreachable("Unexpected declaration"); });
}

auto Dumper::dump(const Prototype *node) -> void {
  auto id = node->id().get();
  auto typeNode = node->returnTypeNode();
  INDENT();
  errs() << "Prototype " << loc(node) << " (name=" << id->name() << " "
         << loc(id) << " (type=" << formatTypeNode(typeNode) << " "
         << loc(typeNode) << ")\n";
  for (auto &parameter : node->parameters())
    dump(parameter.get());
}

auto Dumper::dump(const FunctionDecl *node) -> void {
  INDENT();
  errs() << "FunctionDecl " << loc(node) << "\n";
  dump(node->proto().get());
  dump(node->body().get(), "Body:");
}

auto Dumper::dump(const StructDecl *node) -> void {
  INDENT();
  auto id = node->id().get();
  errs() << "StructDecl " << loc(node)
         << " (name=" << id->name() << " " << loc(id) << ")\n";
  for (auto &var : node->variables())
    dump(var.get());
}

auto Dumper::dump(const Expr *node) -> void {
  llvm::TypeSwitch<const Expr *>(node)
      .Case<UnitExpr, CallExpr, StructLiteralExpr, DecimalLiteralExpr,
            FloatLiteralExpr, BoolLiteralExpr,
            ListLiteralExpr, ListAccessExpr, VariableExpr, MemberExpr,
            AssignExpr, IfExpr, WhileExpr, BinaryExpr>(
          [&](auto *node) { this->dump(node); })
      .Default(
          [&](const Expr *) { llvm_unreachable("Unexpected expression"); });
}

auto Dumper::dump(const UnitExpr *node) -> void {
  INDENT();
  errs() << "UnitExpr " << loc(node)
         << " type=" << formatType(node->type()) << "\n";
}

auto Dumper::dump(const BlockExpr *node, std::string_view string) -> void {
  INDENT();
  errs() << string << "\n";
  for (auto &expr : *node)
    dump(expr.get());
  dump(node->expression().get());
}

auto Dumper::dump(const CallExpr *node) -> void {
  INDENT();
  errs() << "CallExpr " << loc(node)
         << " type=" << formatType(node->type())
         << " callee=" << node->name() << "\n";
  for (auto &expr : *node)
    dump(expr.get());
}

auto Dumper::dump(const StructLiteralExpr *node) -> void {
  INDENT();
  errs() << "StructLiteralExpr " << loc(node)
         << " type=" << formatType(node->type())
         << " name=" << node->name() << "\n";
  for (auto &expr : *node)
    dump(expr.get());
}

auto Dumper::dump(const VariableExpr *node) -> void {
  INDENT();
  errs() << "VariableExpr " << loc(node)
         << " type=" << formatType(node->type())
         << " name=" << node->name() << "\n";
}

auto Dumper::dump(const MemberExpr *node) -> void {
  INDENT();
  errs() << "MemberExpr " << loc(node)
         << " type=" << formatType(node->type())
         << " field=" << node->fieldName() << "\n";
  dump(node->base().get());
}

auto Dumper::dump(const AssignExpr *node) -> void {
  INDENT();
  errs() << "AssignExpr " << loc(node)
         << " type=" << formatType(node->type()) << "\n";
  dump(node->lhs().get());
  dump(node->rhs().get());
}

auto Dumper::dump(const DecimalLiteralExpr *node) -> void {
  INDENT();
  errs() << "DecimalExpr " << loc(node)
         << " type=" << formatType(node->type())
         << " value=" << node->value() << "\n";
}

auto Dumper::dump(const FloatLiteralExpr *node) -> void {
  INDENT();
  llvm::SmallString<32> value;
  node->value().toString(value);
  errs() << "FloatLiteralExpr " << loc(node)
         << " type=" << formatType(node->type())
         << " value=" << value << "\n";
}

auto Dumper::dump(const BoolLiteralExpr *node) -> void {
  INDENT();
  errs() << "BoolLiteralExpr " << loc(node)
         << " type=" << formatType(node->type())
         << " value=" << node->value() << "\n";
}

auto Dumper::dump(const ListLiteralExpr *node) -> void {
  INDENT();
  errs() << "ListLiteralExpr " << loc(node)
         << " type=" << formatType(node->type()) << "\n";
  for (auto &element : node->getElements())
    dump(element.get());
}

auto Dumper::dump(const ListAccessExpr *node) -> void {
  INDENT();
  errs() << "ListAccessExpr " << loc(node)
         << " type=" << formatType(node->type())
         << " name=" << node->getVarName() << "\n";
  for (auto &index : node->getIndices())
    dump(index.get());
}

auto Dumper::dump(const BinaryExpr *node) -> void {
  INDENT();
  errs() << "BinaryExpr " << loc(node)
         << " type=" << formatType(node->type())
         << " op=`"
         << node->op() << "`\n";
  dump(node->lhs().get());
  dump(node->rhs().get());
}

auto Dumper::dump(const IfExpr *node) -> void {
  INDENT();
  errs() << "IfExpr " << loc(node)
         << " type=" << formatType(node->type()) << "\n";
  dump(node->conditionExpr().get());
  dump(node->thenBlock().get(), "thenBlock:");
  dump(node->elseBlock().get(), "elseBlock:");
}

auto Dumper::dump(const WhileExpr *node) -> void {
  INDENT();
  errs() << "WhileExpr " << loc(node)
         << " type=" << formatType(node->type()) << "\n";
  dump(node->conditionExpr().get());
  dump(node->bodyBlock().get(), "bodyBlock:");
}

auto Dumper::dump(const Stat *node) -> void {
  llvm::TypeSwitch<const Stat *>(node)
      .Case<VariableStat, ExprStat>([&](auto *node) { this->dump(node); })
      .Default([&](const Stat *) { llvm_unreachable("Unexpected statement"); });
}

auto Dumper::dump(const VariableStat *node) -> void {
  auto id = node->variable().get();
  auto typeNode = node->typeNode();
  INDENT();
  errs() << "VariableStat ";
  if (node->isConst())
    errs() << "const ";
  errs() << "(id=" << id->name() << " " << loc(id)
         << ") (type=" << formatTypeNode(typeNode) << " "
         << loc(typeNode) << ")\n";
  if (node->init())
    dump(node->init().get());
}

auto Dumper::dump(const ExprStat *node) -> void {
  dump(node->expression().get());
}

namespace cherry {

auto dumpAST(const llvm::SourceMgr &sourceManager, const Module &module)
    -> void {
  Dumper(sourceManager).dump(&module);
}

} // end namespace cherry
