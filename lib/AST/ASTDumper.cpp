//===--- ASTDumper.cpp - Mulberry Language AST Dumper
//------------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "mulberry/AST/AST.h"
#include "mulberry/Basic/Types.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <string_view>

using namespace mulberry;

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
  case IndexExpr::IndexKind::Array:
    return "array";
  case IndexExpr::IndexKind::StdlibTensor:
    return "stdlibTensor";
  case IndexExpr::IndexKind::StdlibList:
    return "stdlibList";
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
  auto dump(const ParameterDecl *node) -> void;
  auto dump(const FieldDecl *node) -> void;
  auto dump(const FunctionDecl *node) -> void;
  auto dump(const TraitDecl *node) -> void;
  auto dump(const TraitMethodDecl *node) -> void;
  auto dump(const ImplDecl *node) -> void;
  auto dump(const StructDecl *node) -> void;
  auto dump(const DataDecl *node) -> void;
  auto dump(const DataConstructorDecl *node) -> void;
  auto dump(const ComptimeTypeAliasDecl *node) -> void;

  // Expressions
  auto dump(const Expr *node) -> void;
  auto dump(const UnitExpr *node) -> void;
  auto dump(const BlockExpr *node, std::string_view string) -> void;
  auto dump(const LambdaExpr *node) -> void;
  auto dump(const MatchExpr *node) -> void;
  auto dump(const MatchExprArm *node) -> void;
  auto dump(const TryExpr *node) -> void;
  auto dump(const CallExpr *node) -> void;
  auto dump(const DataConstructorExpr *node) -> void;
  auto dump(const StructLiteralExpr *node) -> void;
  auto dump(const VariableExpr *node) -> void;
  auto dump(const MemberExpr *node) -> void;
  auto dump(const AssignExpr *node) -> void;
  auto dump(const DecimalLiteralExpr *node) -> void;
  auto dump(const FloatLiteralExpr *node) -> void;
  auto dump(const BoolLiteralExpr *node) -> void;
  auto dump(const StringLiteralExpr *node) -> void;
  auto dump(const InterpolatedStringExpr *node) -> void;
  auto dump(const ObjectIdentityExpr *node) -> void;
  auto dump(const CharLiteralExpr *node) -> void;
  auto dump(const TypeInfoExpr *node) -> void;
  auto dump(const TypeLayoutExpr *node) -> void;
  auto dump(const HeapAllocExpr *node) -> void;
  auto dump(const ArrayLiteralExpr *node) -> void;
  auto dump(const IndexExpr *node) -> void;
  auto dump(const BinaryExpr *node) -> void;

  // Statements
  auto dump(const Stat *node) -> void;
  auto dump(const VariableStat *node) -> void;
  auto dump(const ExprStat *node) -> void;
  auto dump(const IfStat *node) -> void;
  auto dump(const MatchStat *node) -> void;
  auto dump(const MatchArm *node) -> void;
  auto dump(const DataPattern *node) -> void;
  auto dump(const WhileStat *node) -> void;
  auto dump(const ForStat *node) -> void;
  auto dump(const BreakStat *node) -> void;
  auto dump(const ContinueStat *node) -> void;
  auto dump(const ReturnStat *node) -> void;

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

    if (dyn_cast<ComputedTypeNode>(node))
      return "<computed>";

    if (auto *arrayType = dyn_cast<ArrayTypeNode>(node)) {
      std::string result = formatTypeNode(arrayType->elementTypeNode());
      result += "[";
      std::string separator;
      for (auto dim : arrayType->shape()) {
        result += separator;
        result += dim < 0 ? "?" : std::to_string(dim);
        separator = ", ";
      }
      result += "]";
      return result;
    }

    if (auto *ptrType = dyn_cast<PtrTypeNode>(node))
      return "Ptr<" + formatTypeNode(ptrType->pointeeTypeNode()) + ">";

    if (auto *functionType = dyn_cast<FunctionTypeNode>(node)) {
      std::string result = "fn(";
      std::string separator;
      for (size_t i = 0; i < functionType->parameterTypes().size(); ++i) {
        result += separator;
        if (functionType->parameterCanMutateObject()[i])
          result += "mut ";
        result += formatTypeNode(functionType->parameterTypes()[i].get());
        separator = ", ";
      }
      result += "): ";
      result += formatTypeNode(functionType->returnTypeNode());
      return result;
    }

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
      else if (parameter.hasTraitConstraint()) {
        result += ": ";
        result += parameter.traitName;
      }
      separator = ", ";
    }
    return result;
  }

  auto formatComptimeValue(const ComptimeValue &value) -> std::string {
    switch (value.kind()) {
    case ComptimeValue::Kind::Type:
      return formatSourceType(value.type());
    case ComptimeValue::Kind::Bool:
      return value.boolValue() ? "true" : "false";
    case ComptimeValue::Kind::UInt64:
      return std::to_string(value.uint64Value());
    case ComptimeValue::Kind::String:
      return "\"" + std::string(value.stringValue()) + "\"";
    }
    llvm_unreachable("unexpected comptime value");
  }

  auto formatOrigin(const ComptimeAliasOrigin *origin) -> std::string {
    if (!origin)
      return {};

    std::string result = std::string(origin->aliasName()) + "<";
    std::string separator;
    for (auto &argument : origin->arguments()) {
      result += separator;
      result += formatComptimeValue(argument);
      separator = ", ";
    }
    result += ">";
    return result;
  }

  auto formatSourceType(const Type *type) -> std::string {
    if (!type)
      return "<pending>";

    if (auto *arrayType = getArrayType(type))
      return "Array<" + formatSourceType(arrayType->elementType()) + ", " +
             std::to_string(arrayType->size()) + ">";
    if (auto *ptrType = getPtrType(type))
      return "Ptr<" + formatSourceType(ptrType->pointeeType()) + ">";
    if (auto origin = formatOrigin(originOf(type)); !origin.empty())
      return origin;
    return formatType(type);
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
      .Case<ImportDecl, FunctionDecl, StructDecl, DataDecl,
            ComptimeTypeAliasDecl, TraitDecl, ImplDecl>(
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

auto Dumper::dump(const ParameterDecl *node) -> void {
  auto id = node->variable().get();
  auto typeNode = node->typeNode();
  INDENT();
  errs() << "ParameterDecl ";
  if (!node->canMutateObject())
    errs() << "readonly ";
  errs() << "(id=" << id->name() << " " << loc(id)
         << ") (type=" << formatTypeNode(typeNode) << " "
         << loc(typeNode) << ")";
  if (auto origin = formatOrigin(originOf(node->type())); !origin.empty())
    errs() << " origin=" << origin;
  errs() << "\n";
}

auto Dumper::dump(const FieldDecl *node) -> void {
  auto id = node->variable().get();
  auto typeNode = node->typeNode();
  INDENT();
  errs() << "FieldDecl "
         << "(id=" << id->name() << " " << loc(id)
         << ") (type=" << formatTypeNode(typeNode) << " "
         << loc(typeNode) << ")";
  if (auto origin = formatOrigin(originOf(node->type())); !origin.empty())
    errs() << " origin=" << origin;
  errs() << "\n";
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

auto Dumper::dump(const TraitDecl *node) -> void {
  INDENT();
  errs() << "TraitDecl " << loc(node) << " name=" << node->name() << "\n";
  for (auto &method : node->methods())
    dump(method.get());
}

auto Dumper::dump(const TraitMethodDecl *node) -> void {
  INDENT();
  errs() << "TraitMethodDecl " << loc(node) << " name=" << node->name()
         << " receiver="
         << (node->receiverCanMutateObject() ? "mut" : "readonly")
         << " return=" << formatTypeNode(node->returnTypeNode()) << "\n";
  for (auto &parameter : node->parameters())
    dump(parameter.get());
  if (node->hasDefaultBody())
    dump(node->body().get(), "DefaultBody:");
}

auto Dumper::dump(const ImplDecl *node) -> void {
  INDENT();
  errs() << "ImplDecl " << loc(node) << " trait=" << node->traitName()
         << " target=" << formatTypeNode(node->targetTypeNode());
  if (node->isGeneric())
    errs() << " parameter=" << formatComptimeParams(node->comptimeParameters());
  errs() << "\n";
  if (node->whereCondition()) {
    INDENT();
    errs() << "Where:\n";
    ++_curIndent;
    dump(node->whereCondition());
    --_curIndent;
  }
  for (auto &method : node->methods())
    dump(method.get());
}

auto Dumper::dump(const StructDecl *node) -> void {
  INDENT();
  auto id = node->id().get();
  errs() << "StructDecl " << loc(node)
         << " (name=" << id->name() << " " << loc(id) << ")\n";
  for (auto &field : node->fields())
    dump(field.get());
  for (auto &method : node->methods())
    dump(method.get());
}

auto Dumper::dump(const DataDecl *node) -> void {
  INDENT();
  errs() << "DataDecl " << loc(node)
         << " (name=" << node->name();
  if (node->isGeneric())
    errs() << " parameter=" << formatComptimeParams(node->parameters());
  errs() << ")\n";
  for (auto &constructor : node->constructors())
    dump(constructor.get());
}

auto Dumper::dump(const DataConstructorDecl *node) -> void {
  INDENT();
  errs() << "DataConstructorDecl " << loc(node)
         << " name=" << node->name() << " payloadTypes=";
  std::string separator;
  for (auto &payloadType : node->payloadTypes()) {
    errs() << separator << formatTypeNode(payloadType.get());
    separator = ", ";
  }
  errs() << "\n";
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
      .Case<UnitExpr, CallExpr, DataConstructorExpr, StructLiteralExpr,
            MatchExpr, TryExpr, DecimalLiteralExpr, FloatLiteralExpr,
            BoolLiteralExpr, StringLiteralExpr, InterpolatedStringExpr,
            ObjectIdentityExpr, CharLiteralExpr, TypeInfoExpr, TypeLayoutExpr,
            LambdaExpr, HeapAllocExpr, ArrayLiteralExpr, IndexExpr,
            VariableExpr, MemberExpr, AssignExpr, BinaryExpr>(
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
}

auto Dumper::dump(const LambdaExpr *node) -> void {
  INDENT();
  errs() << "LambdaExpr " << loc(node)
         << " type=" << formatType(node->type())
         << " function=" << node->functionName() << " parameters=";
  std::string separator;
  for (auto &parameter : node->parameters()) {
    errs() << separator << parameter.name << ": "
           << formatType(parameter.type);
    separator = ", ";
  }
  errs() << "\n";
  if (node->body())
    dump(node->body().get());
}

auto Dumper::dump(const MatchExpr *node) -> void {
  INDENT();
  errs() << "MatchExpr " << loc(node)
         << " type=" << formatType(node->type()) << "\n";
  dump(node->value().get());
  for (auto &arm : node->arms())
    dump(arm.get());
}

auto Dumper::dump(const MatchExprArm *node) -> void {
  INDENT();
  errs() << "MatchExprArm " << loc(node) << "\n";
  dump(node->pattern().get());
  if (!node->bodyBlock()->statements().empty())
    dump(node->bodyBlock().get(), "bodyBlock:");
  dump(node->resultExpr().get());
}

auto Dumper::dump(const TryExpr *node) -> void {
  INDENT();
  errs() << "TryExpr " << loc(node)
         << " type=" << formatType(node->type()) << "\n";
  dump(node->value().get());
}

auto Dumper::dump(const CallExpr *node) -> void {
  INDENT();
  errs() << "CallExpr " << loc(node)
         << " type=" << formatType(node->type())
         << " callee=" << node->name();
  if (node->isIndirectCall())
    errs() << " indirect";
  errs() << "\n";
  if (node->hasReceiver())
    dump(node->receiver().get());
  for (auto &expr : *node)
    dump(expr.get());
}

auto Dumper::dump(const DataConstructorExpr *node) -> void {
  INDENT();
  errs() << "DataConstructorExpr " << loc(node)
         << " type=" << formatType(node->type())
         << " constructor=" << node->name()
         << " tag=" << node->constructorIndex() << "\n";
  for (auto &expression : node->expressions())
    dump(expression.get());
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
         << " name=" << node->name();
  if (node->isFunctionValue())
    errs() << " functionValue";
  if (node->comptimeValue())
    errs() << " comptimeValue="
           << formatComptimeValue(*node->comptimeValue());
  errs() << "\n";
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

auto Dumper::dump(const InterpolatedStringExpr *node) -> void {
  INDENT();
  errs() << "InterpolatedStringExpr " << loc(node)
         << " type=" << formatType(node->type())
         << " segments=" << node->segments().size() << "\n";
  for (auto &segment : node->segments())
    dump(segment.get());
}

auto Dumper::dump(const ObjectIdentityExpr *node) -> void {
  INDENT();
  errs() << "ObjectIdentityExpr " << loc(node)
         << " type=" << formatType(node->type())
         << " objectType=" << node->typeName() << "\n";
  dump(node->value().get());
}

auto Dumper::dump(const CharLiteralExpr *node) -> void {
  INDENT();
  errs() << "CharLiteralExpr " << loc(node)
         << " type=" << formatType(node->type())
         << " value=" << static_cast<unsigned>(node->value()) << "\n";
}

auto Dumper::dump(const TypeInfoExpr *node) -> void {
  INDENT();
  errs() << "TypeInfoExpr " << loc(node)
         << " target=" << formatTypeNode(node->typeNode()) << "\n";
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

auto Dumper::dump(const Stat *node) -> void {
  llvm::TypeSwitch<const Stat *>(node)
      .Case<VariableStat, ExprStat, IfStat, MatchStat, WhileStat, ForStat,
            BreakStat, ContinueStat, ReturnStat>(
          [&](auto *node) { this->dump(node); })
      .Default([&](const Stat *) { llvm_unreachable("Unexpected statement"); });
}

auto Dumper::dump(const VariableStat *node) -> void {
  auto id = node->variable().get();
  auto typeNode = node->typeNode();
  INDENT();
  errs() << "VariableStat ";
  if (node->comptimeValue())
    errs() << "comptime ";
  if (node->isConstBinding())
    errs() << "const ";
  else if (!node->canMutateObject())
    errs() << "readonly ";
  errs() << "(id=" << id->name() << " " << loc(id) << ") ";
  if (typeNode) {
    errs() << "(type=" << formatTypeNode(typeNode) << " " << loc(typeNode)
           << ")";
  } else {
    errs() << "(inferredType=" << formatSourceType(node->type()) << ")";
  }
  if (auto origin = formatOrigin(originOf(node->type())); !origin.empty())
    errs() << " origin=" << origin;
  if (node->comptimeValue())
    errs() << " value=" << formatComptimeValue(*node->comptimeValue());
  errs() << "\n";
  if (node->init())
    dump(node->init().get());
}

auto Dumper::dump(const ExprStat *node) -> void {
  dump(node->expression().get());
}

auto Dumper::dump(const IfStat *node) -> void {
  INDENT();
  errs() << "IfStat " << loc(node);
  if (node->comptimeValue())
    errs() << " comptime value="
           << (*node->comptimeValue() ? "true" : "false");
  errs() << "\n";
  dump(node->conditionExpr().get());
  dump(node->thenBlock().get(), "thenBlock:");
  if (node->hasElseBlock())
    dump(node->elseBlock().get(), "elseBlock:");
}

auto Dumper::dump(const MatchStat *node) -> void {
  INDENT();
  errs() << "MatchStat " << loc(node) << "\n";
  dump(node->value().get());
  for (auto &arm : node->arms())
    dump(arm.get());
}

auto Dumper::dump(const MatchArm *node) -> void {
  INDENT();
  errs() << "MatchArm " << loc(node) << "\n";
  dump(node->pattern().get());
  dump(node->bodyBlock().get(), "bodyBlock:");
}

auto Dumper::dump(const DataPattern *node) -> void {
  INDENT();
  errs() << "DataPattern " << loc(node)
         << " constructor=" << node->constructorName()
         << " tag=" << node->constructorIndex() << "\n";
  for (auto &binding : node->bindings())
    dump(binding.get());
}

auto Dumper::dump(const WhileStat *node) -> void {
  INDENT();
  errs() << "WhileStat " << loc(node) << "\n";
  dump(node->conditionExpr().get());
  dump(node->bodyBlock().get(), "bodyBlock:");
}

auto Dumper::dump(const ForStat *node) -> void {
  INDENT();
  errs() << "ForStat " << loc(node)
         << " variable=" << node->variableName() << "\n";
  dump(node->startExpr().get());
  dump(node->endExpr().get());
  dump(node->bodyBlock().get(), "bodyBlock:");
}

auto Dumper::dump(const BreakStat *node) -> void {
  INDENT();
  errs() << "BreakStat " << loc(node) << "\n";
}

auto Dumper::dump(const ContinueStat *node) -> void {
  INDENT();
  errs() << "ContinueStat " << loc(node) << "\n";
}

auto Dumper::dump(const ReturnStat *node) -> void {
  INDENT();
  errs() << "ReturnStat " << loc(node) << "\n";
  if (node->hasExpression())
    dump(node->expression().get());
}

namespace mulberry {

auto dumpAST(const llvm::SourceMgr &sourceManager, const Module &module)
    -> void {
  Dumper(sourceManager).dump(&module);
}

} // end namespace mulberry
