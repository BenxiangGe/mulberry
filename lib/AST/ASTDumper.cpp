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

auto indexKindName(IndexExpr::IndexKind kind) -> std::string_view {
  switch (kind) {
  case IndexExpr::IndexKind::Unknown:
    return "unknown";
  case IndexExpr::IndexKind::Ptr:
    return "ptr";
  case IndexExpr::IndexKind::Tensor:
    return "tensor";
  case IndexExpr::IndexKind::StdlibList:
    return "stdlibList";
  }
  return "unknown";
}

auto intrinsicKindName(CallExpr::IntrinsicKind kind) -> std::string_view {
  switch (kind) {
  case CallExpr::IntrinsicKind::None:
    return "none";
  case CallExpr::IntrinsicKind::Print:
    return "print";
  case CallExpr::IntrinsicKind::BoolToUInt64:
    return "boolToUInt64";
  case CallExpr::IntrinsicKind::Zeros:
    return "zeros";
  case CallExpr::IntrinsicKind::TensorPack:
    return "tensorPack";
  case CallExpr::IntrinsicKind::TensorView:
    return "tensorView";
  case CallExpr::IntrinsicKind::PtrAsUInt8:
    return "ptrAsUInt8";
  }

  return "unknown";
}

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
  auto dump(const ImportDecl *node) -> void;
  auto dump(const Prototype *node) -> void;
  auto dump(const FunctionDecl *node) -> void;
  auto dump(const StructDecl *node) -> void;
  auto dump(const ComptimeTypeAliasDecl *node) -> void;

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
  auto dump(const StringLiteralExpr *node) -> void;
  auto dump(const TypeLayoutExpr *node) -> void;
  auto dump(const HeapAllocExpr *node) -> void;
  auto dump(const DerefExpr *node) -> void;
  auto dump(const ArrayLiteralExpr *node) -> void;
  auto dump(const IndexExpr *node) -> void;
  auto dump(const BinaryExpr *node) -> void;
  auto dump(const IfExpr *node) -> void;
  auto dump(const WhileExpr *node) -> void;
  auto dump(const ForExpr *node) -> void;

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

    if (auto *tensorType = dyn_cast<TensorTypeNode>(node)) {
      std::string result = formatTypeNode(tensorType->elementTypeNode());
      result += "[";
      std::string separator;
      for (auto dim : tensorType->shape()) {
        result += separator;
        result += dim < 0 ? "?" : std::to_string(dim);
        separator = ", ";
      }
      result += "]";
      return result;
    }

    if (auto *listType = dyn_cast<ListTypeNode>(node))
      return "List<" + formatTypeNode(listType->elementTypeNode()) + ">";

    if (auto *ptrType = dyn_cast<PtrTypeNode>(node))
      return "Ptr<" + formatTypeNode(ptrType->pointeeTypeNode()) + ">";

    if (auto *genericType = dyn_cast<GenericTypeNode>(node)) {
      std::string result = std::string(genericType->name()) + "<";
      std::string separator;
      for (auto &argument : genericType->arguments()) {
        result += separator;
        if (argument.kind() == ComptimeArg::Kind::UInt64)
          result += std::to_string(argument.uint64Value());
        else
          result += formatTypeNode(argument.typeNode());
        separator = ", ";
      }
      result += ">";
      return result;
    }

    if (auto *structType = dyn_cast<StructTypeNode>(node)) {
      std::string result = "struct {";
      std::string separator;
      for (auto &field : structType->fields()) {
        result += separator;
        result += field->variable()->name();
        result += ": ";
        result += formatTypeNode(field->typeNode());
        separator = ", ";
      }
      for (auto &method : structType->methods()) {
        result += separator;
        result += "fn ";
        result += method->proto()->id()->name();
        separator = ", ";
      }
      result += "}";
      return result;
    }

    auto *namedType = cast<NamedTypeNode>(node);
    return std::string(namedType->name());
  }

  auto formatComptimeParams(const std::vector<ComptimeParam> &parameters)
      -> std::string {
    std::string result;
    std::string separator;
    for (auto &parameter : parameters) {
      result += separator;
      result += parameter.name;
      if (parameter.kind == ComptimeParam::Kind::UInt64)
        result += ": UInt64";
      separator = ", ";
    }
    return result;
  }

  auto formatOrigin(const ComptimeAliasOrigin *origin) -> std::string {
    if (!origin)
      return {};

    std::string result = std::string(origin->aliasName()) + "<";
    std::string separator;
    for (auto &argument : origin->arguments()) {
      result += separator;
      if (argument.kind() == ComptimeTypeValue::Kind::Type)
        result += formatType(argument.type());
      else
        result += std::to_string(argument.uint64Value());
      separator = ", ";
    }
    result += ">";
    return result;
  }

  auto originOf(const Type *type) -> const ComptimeAliasOrigin * {
    if (auto *ptrType = getPtrType(type))
      type = ptrType->pointeeType();
    if (auto *structType = getStructType(type))
      return structType->origin();
    return nullptr;
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
  if (!node->packageName().empty()) {
    INDENT();
    errs() << "Package " << loc(node)
           << " name=" << node->packageName() << "\n";
  }
  for (auto &decl : *node)
    dump(decl.get());
}

auto Dumper::dump(const Decl *node) -> void {
  llvm::TypeSwitch<const Decl *>(node)
      .Case<ImportDecl, FunctionDecl, StructDecl, ComptimeTypeAliasDecl>(
          [&](auto *node) { this->dump(node); })
      .Default(
          [&](const Decl *) { llvm_unreachable("Unexpected declaration"); });
}

auto Dumper::dump(const ImportDecl *node) -> void {
  INDENT();
  errs() << "ImportDecl " << loc(node)
         << " module=" << node->moduleName() << "\n";
}

auto Dumper::dump(const Prototype *node) -> void {
  auto id = node->id().get();
  auto typeNode = node->returnTypeNode();
  INDENT();
  errs() << "Prototype " << loc(node) << " (name=" << id->name();
  if (node->isGeneric())
    errs() << " parameter=" << formatComptimeParams(node->comptimeParameters());
  errs() << " "
         << loc(id) << " (type=" << formatTypeNode(typeNode) << " "
         << loc(typeNode) << ")\n";
  for (auto &parameter : node->parameters())
    dump(parameter.get());
}

auto Dumper::dump(const FunctionDecl *node) -> void {
  INDENT();
  errs() << "FunctionDecl " << loc(node);
  if (node->isExtern())
    errs() << " extern";
  errs() << "\n";
  dump(node->proto().get());
  if (node->isExtern())
    return;
  dump(node->body().get(), "Body:");
}

auto Dumper::dump(const StructDecl *node) -> void {
  INDENT();
  auto id = node->id().get();
  errs() << "StructDecl " << loc(node)
         << " (name=" << id->name() << " " << loc(id) << ")\n";
  for (auto &var : node->variables())
    dump(var.get());
  for (auto &method : node->methods())
    dump(method.get());
}

auto Dumper::dump(const ComptimeTypeAliasDecl *node) -> void {
  INDENT();
  errs() << "ComptimeTypeAliasDecl " << loc(node)
         << " (name=" << node->name()
         << " parameter=" << formatComptimeParams(node->parameters())
         << " type=" << formatTypeNode(node->bodyTypeNode()) << ")\n";
}

auto Dumper::dump(const Expr *node) -> void {
  llvm::TypeSwitch<const Expr *>(node)
      .Case<UnitExpr, CallExpr, StructLiteralExpr, DecimalLiteralExpr,
            FloatLiteralExpr, BoolLiteralExpr, StringLiteralExpr,
            TypeLayoutExpr, HeapAllocExpr, DerefExpr,
            ArrayLiteralExpr, IndexExpr, VariableExpr, MemberExpr,
            AssignExpr, IfExpr, WhileExpr, ForExpr, BinaryExpr>(
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
         << " callee=" << node->name();
  if (node->intrinsicKind() != CallExpr::IntrinsicKind::None)
    errs() << " intrinsic=" << intrinsicKindName(node->intrinsicKind());
  errs() << "\n";
  if (node->hasReceiver())
    dump(node->receiver().get());
  for (auto &expr : *node)
    dump(expr.get());
}

auto Dumper::dump(const StructLiteralExpr *node) -> void {
  INDENT();
  errs() << "StructLiteralExpr " << loc(node)
         << " type=" << formatType(node->type())
         << " literalType=" << formatTypeNode(node->typeNode()) << "\n";
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

auto Dumper::dump(const StringLiteralExpr *node) -> void {
  INDENT();
  errs() << "StringLiteralExpr " << loc(node)
         << " type=" << formatType(node->type())
         << " value=\"" << node->value() << "\"\n";
}

auto Dumper::dump(const TypeLayoutExpr *node) -> void {
  INDENT();
  auto query = node->query() == TypeLayoutExpr::Query::SizeOf ? "sizeof"
                                                              : "alignof";
  errs() << "TypeLayoutExpr " << loc(node)
         << " type=" << formatType(node->type())
         << " query=" << query
         << " target=" << formatTypeNode(node->typeNode())
         << " value=" << node->value() << "\n";
}

auto Dumper::dump(const HeapAllocExpr *node) -> void {
  INDENT();
  errs() << "HeapAllocExpr " << loc(node)
         << " type=" << formatType(node->type())
         << " allocated=" << formatType(node->allocatedType()) << "\n";
  if (node->count())
    dump(node->count().get());
}

auto Dumper::dump(const DerefExpr *node) -> void {
  INDENT();
  errs() << "DerefExpr " << loc(node)
         << " type=" << formatType(node->type()) << "\n";
  dump(node->pointer().get());
}

auto Dumper::dump(const ArrayLiteralExpr *node) -> void {
  INDENT();
  errs() << "ArrayLiteralExpr " << loc(node)
         << " type=" << formatType(node->type()) << "\n";
  for (auto &element : node->getElements())
    dump(element.get());
}

auto Dumper::dump(const IndexExpr *node) -> void {
  INDENT();
  errs() << "IndexExpr " << loc(node)
         << " type=" << formatType(node->type())
         << " kind=" << indexKindName(node->indexKind()) << "\n";
  dump(node->base().get());
  for (auto &index : node->indices())
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

auto Dumper::dump(const ForExpr *node) -> void {
  INDENT();
  errs() << "ForExpr " << loc(node)
         << " type=" << formatType(node->type())
         << " variable=" << node->variableName() << "\n";
  dump(node->startExpr().get());
  dump(node->endExpr().get());
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
         << loc(typeNode) << ")";
  if (auto origin = formatOrigin(originOf(node->type())); !origin.empty())
    errs() << " origin=" << origin;
  errs() << "\n";
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
