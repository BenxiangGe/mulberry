//===--- Sema.cpp - Mulberry Semantic Analysis ------------------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "mulberry/Sema/Sema.h"
#include "Symbols.h"
#include "mulberry/AST/AST.h"
#include "mulberry/Sema/DiagnosticsSema.h"
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using namespace mulberry;
using llvm::cast;
using llvm::dyn_cast;

using NameSet = std::set<std::string, std::less<>>;

struct TypeSubstitution {
  std::string parameterName;
  const TypeNode *argumentTypeNode = nullptr;
  std::optional<uint64_t> uint64Value;
};

struct ComptimeArgument {
  ComptimeArg::Kind kind = ComptimeArg::Kind::Type;
  const Type *type = nullptr;
  std::unique_ptr<TypeNode> typeNode;
  uint64_t uint64Value = 0;
};

struct InferredComptimeArgument {
  ComptimeParam::Kind kind = ComptimeParam::Kind::Type;
  const Type *type = nullptr;
  std::unique_ptr<TypeNode> typeNode;
  std::optional<uint64_t> uint64Value;

  auto isResolved() const -> bool {
    if (kind == ComptimeParam::Kind::Type)
      return type != nullptr;
    return uint64Value.has_value();
  }
};

auto substituteExpr(const Expr *node,
                    const std::vector<TypeSubstitution> &substitutions)
    -> std::unique_ptr<Expr>;

auto substituteBlockExpr(const BlockExpr *node,
                         const std::vector<TypeSubstitution> &substitutions)
    -> std::unique_ptr<BlockExpr>;

auto toComptimeTypeValues(const std::vector<ComptimeArgument> &arguments)
    -> std::vector<ComptimeTypeValue> {
  std::vector<ComptimeTypeValue> values;
  for (auto &argument : arguments) {
    if (argument.kind == ComptimeArg::Kind::Type)
      values.push_back(ComptimeTypeValue(argument.type));
    else
      values.push_back(ComptimeTypeValue(argument.uint64Value));
  }
  return values;
}

auto replacePlaceholder(std::string message, std::string_view placeholder,
                        std::string_view value) -> std::string {
  auto position = message.find(placeholder);
  if (position == std::string::npos)
    return message;
  message.replace(position, placeholder.size(), value);
  return message;
}

auto formatNameDiagnostic(const char *diagnostic, std::string_view name)
    -> std::string {
  return replacePlaceholder(diagnostic, "%s", name);
}

auto formatNameSizeDiagnostic(const char *diagnostic, std::string_view name,
                              size_t size) -> std::string {
  auto message = formatNameDiagnostic(diagnostic, name);
  return replacePlaceholder(message, "%d", std::to_string(size));
}

auto declareName(NameSet &names, std::string_view name) -> bool {
  return names.insert(std::string(name)).second;
}

auto packageNameOf(std::string_view name) -> std::string {
  auto dot = name.rfind('.');
  if (dot == std::string_view::npos)
    return {};
  return std::string(name.substr(0, dot));
}

auto createMemberAccessChain(llvm::SMLoc location, std::string_view name)
    -> std::unique_ptr<Expr> {
  auto dot = name.find('.');
  if (dot == std::string_view::npos)
    return std::make_unique<VariableExpr>(location, name);

  std::unique_ptr<Expr> expr =
      std::make_unique<VariableExpr>(location, name.substr(0, dot));
  while (dot != std::string_view::npos) {
    auto nextDot = name.find('.', dot + 1);
    auto fieldName = name.substr(dot + 1, nextDot - dot - 1);
    expr = std::make_unique<MemberExpr>(location, std::move(expr), fieldName);
    dot = nextDot;
  }
  return expr;
}

auto methodFunctionName(std::string_view ownerName,
                        std::string_view methodName) -> std::string {
  std::string name(ownerName);
  name += ".";
  name += methodName;
  return name;
}

auto cloneTypeNode(const TypeNode *node) -> std::unique_ptr<TypeNode> {
  if (auto *unitType = dyn_cast<UnitTypeNode>(node))
    return std::make_unique<UnitTypeNode>(unitType->location());

  if (auto *namedType = dyn_cast<NamedTypeNode>(node))
    return std::make_unique<NamedTypeNode>(namedType->location(),
                                           namedType->name());

  if (auto *tensorType = dyn_cast<TensorTypeNode>(node)) {
    return std::make_unique<TensorTypeNode>(
        cloneTypeNode(tensorType->elementTypeNode()), tensorType->shape(),
        tensorType->location());
  }

  if (auto *listType = dyn_cast<ListTypeNode>(node)) {
    return std::make_unique<ListTypeNode>(
        cloneTypeNode(listType->elementTypeNode()), listType->location());
  }

  if (auto *ptrType = dyn_cast<PtrTypeNode>(node)) {
    return std::make_unique<PtrTypeNode>(
        cloneTypeNode(ptrType->pointeeTypeNode()), ptrType->location());
  }

  if (auto *genericType = dyn_cast<GenericTypeNode>(node)) {
    std::vector<ComptimeArg> arguments;
    for (auto &argument : genericType->arguments()) {
      if (argument.kind() == ComptimeArg::Kind::UInt64) {
        arguments.push_back(
            ComptimeArg(argument.location(), argument.uint64Value()));
        continue;
      }
      arguments.push_back(ComptimeArg(cloneTypeNode(argument.typeNode())));
    }
    return std::make_unique<GenericTypeNode>(
        genericType->location(), genericType->name(), std::move(arguments));
  }

  auto *structType = cast<StructTypeNode>(node);
  VectorUniquePtr<VariableStat> fields;
  for (auto &field : structType->fields()) {
    auto variable = std::make_unique<VariableExpr>(
        field->variable()->location(), field->variable()->name());
    fields.push_back(std::make_unique<VariableStat>(
        field->location(), std::move(variable),
        cloneTypeNode(field->typeNode()), nullptr));
  }
  VectorUniquePtr<FunctionDecl> methods;
  for (auto &method : structType->methods()) {
    auto functionName = std::make_unique<FunctionName>(
        method->proto()->id()->location(), method->proto()->id()->name());
    VectorUniquePtr<VariableStat> parameters;
    for (auto &parameter : method->proto()->parameters()) {
      auto variable = std::make_unique<VariableExpr>(
          parameter->variable()->location(), parameter->variable()->name());
      parameters.push_back(std::make_unique<VariableStat>(
          parameter->location(), std::move(variable),
          cloneTypeNode(parameter->typeNode()), nullptr));
    }
    auto prototype = std::make_unique<Prototype>(
        method->proto()->location(), std::move(functionName),
        std::move(parameters), cloneTypeNode(method->proto()->returnTypeNode()),
        std::vector<ComptimeParam>(method->proto()->comptimeParameters().begin(),
                                   method->proto()->comptimeParameters().end()));
    methods.push_back(std::make_unique<FunctionDecl>(
        method->location(), std::move(prototype),
        substituteBlockExpr(method->body().get(), {})));
  }
  return std::make_unique<StructTypeNode>(
      structType->location(), std::move(fields), std::move(methods));
}

auto typeToTypeNode(const Type *type, llvm::SMLoc location)
    -> std::unique_ptr<TypeNode> {
  if (auto *builtinType = getBuiltinType(type))
    return std::make_unique<NamedTypeNode>(location, builtinType->name());

  if (auto *structType = getStructType(type)) {
    if (auto *origin = structType->origin()) {
      // Keep generic alias identity when a resolved type is substituted into
      // another alias; the internal mangled struct name is not a source type.
      std::vector<ComptimeArg> arguments;
      for (auto &argument : origin->arguments()) {
        if (argument.kind() == ComptimeTypeValue::Kind::UInt64) {
          arguments.push_back(
              ComptimeArg(location, argument.uint64Value()));
          continue;
        }
        arguments.push_back(ComptimeArg(typeToTypeNode(
            argument.type(), location)));
      }
      return std::make_unique<GenericTypeNode>(
          location, origin->aliasName(), std::move(arguments));
    }

    return std::make_unique<NamedTypeNode>(location, structType->name());
  }

  if (auto *tensorType = getTensorType(type)) {
    return std::make_unique<TensorTypeNode>(
        typeToTypeNode(tensorType->elementType(), location),
        tensorType->shape(), location);
  }

  if (auto *listType = getListType(type)) {
    return std::make_unique<ListTypeNode>(
        typeToTypeNode(listType->elementType(), location), location);
  }

  auto *ptrType = cast<PtrType>(type);
  return std::make_unique<PtrTypeNode>(
      typeToTypeNode(ptrType->pointeeType(), location), location);
}

auto substituteTypeNode(const TypeNode *node,
                        const TypeSubstitution &substitution)
    -> std::unique_ptr<TypeNode> {
  if (substitution.parameterName.empty())
    return cloneTypeNode(node);

  if (auto *namedType = dyn_cast<NamedTypeNode>(node)) {
    if (namedType->name() == substitution.parameterName &&
        substitution.argumentTypeNode)
      return cloneTypeNode(substitution.argumentTypeNode);
    return cloneTypeNode(namedType);
  }

  if (auto *tensorType = dyn_cast<TensorTypeNode>(node)) {
    return std::make_unique<TensorTypeNode>(
        substituteTypeNode(tensorType->elementTypeNode(), substitution),
        tensorType->shape(), tensorType->location());
  }

  if (auto *listType = dyn_cast<ListTypeNode>(node)) {
    return std::make_unique<ListTypeNode>(
        substituteTypeNode(listType->elementTypeNode(), substitution),
        listType->location());
  }

  if (auto *ptrType = dyn_cast<PtrTypeNode>(node)) {
    return std::make_unique<PtrTypeNode>(
        substituteTypeNode(ptrType->pointeeTypeNode(), substitution),
        ptrType->location());
  }

  if (auto *genericType = dyn_cast<GenericTypeNode>(node)) {
    std::vector<ComptimeArg> arguments;
    for (auto &argument : genericType->arguments()) {
      if (argument.kind() == ComptimeArg::Kind::UInt64) {
        arguments.push_back(
            ComptimeArg(argument.location(), argument.uint64Value()));
        continue;
      }

      if (auto *namedType = dyn_cast<NamedTypeNode>(argument.typeNode())) {
        if (namedType->name() == substitution.parameterName &&
            substitution.uint64Value) {
          arguments.push_back(
              ComptimeArg(namedType->location(), *substitution.uint64Value));
          continue;
        }
      }

      arguments.push_back(
          ComptimeArg(substituteTypeNode(argument.typeNode(), substitution)));
    }
    return std::make_unique<GenericTypeNode>(
        genericType->location(), genericType->name(), std::move(arguments));
  }

  auto *structType = cast<StructTypeNode>(node);
  VectorUniquePtr<VariableStat> fields;
  for (auto &field : structType->fields()) {
    auto variable = std::make_unique<VariableExpr>(
        field->variable()->location(), field->variable()->name());
    fields.push_back(std::make_unique<VariableStat>(
        field->location(), std::move(variable),
        substituteTypeNode(field->typeNode(), substitution),
        nullptr));
  }
  VectorUniquePtr<FunctionDecl> methods;
  for (auto &method : structType->methods()) {
    auto functionName = std::make_unique<FunctionName>(
        method->proto()->id()->location(), method->proto()->id()->name());
    VectorUniquePtr<VariableStat> parameters;
    for (auto &parameter : method->proto()->parameters()) {
      auto variable = std::make_unique<VariableExpr>(
          parameter->variable()->location(), parameter->variable()->name());
      parameters.push_back(std::make_unique<VariableStat>(
          parameter->location(), std::move(variable),
          substituteTypeNode(parameter->typeNode(), substitution), nullptr));
    }
    auto prototype = std::make_unique<Prototype>(
        method->proto()->location(), std::move(functionName),
        std::move(parameters),
        substituteTypeNode(method->proto()->returnTypeNode(), substitution),
        std::vector<ComptimeParam>(method->proto()->comptimeParameters().begin(),
                                   method->proto()->comptimeParameters().end()));
    methods.push_back(std::make_unique<FunctionDecl>(
        method->location(), std::move(prototype),
        substituteBlockExpr(
            method->body().get(),
            std::vector<TypeSubstitution>{substitution})));
  }
  return std::make_unique<StructTypeNode>(
      structType->location(), std::move(fields), std::move(methods));
}

auto substituteTypeNode(const TypeNode *node,
                        const std::vector<TypeSubstitution> &substitutions)
    -> std::unique_ptr<TypeNode> {
  auto result = cloneTypeNode(node);
  for (auto &substitution : substitutions)
    result = substituteTypeNode(result.get(), substitution);
  return result;
}

auto containsComptimeParameter(const TypeNode *node,
                               const std::vector<ComptimeParam> &parameters)
    -> bool {
  if (auto *namedType = dyn_cast<NamedTypeNode>(node)) {
    for (auto &parameter : parameters)
      if (namedType->name() == parameter.name)
        return true;
    return false;
  }

  if (auto *tensorType = dyn_cast<TensorTypeNode>(node))
    return containsComptimeParameter(tensorType->elementTypeNode(),
                                     parameters);

  if (auto *listType = dyn_cast<ListTypeNode>(node))
    return containsComptimeParameter(listType->elementTypeNode(), parameters);

  if (auto *ptrType = dyn_cast<PtrTypeNode>(node))
    return containsComptimeParameter(ptrType->pointeeTypeNode(), parameters);

  if (auto *genericType = dyn_cast<GenericTypeNode>(node)) {
    for (auto &argument : genericType->arguments()) {
      if (argument.kind() == ComptimeArg::Kind::Type &&
          containsComptimeParameter(argument.typeNode(), parameters))
        return true;
    }
    return false;
  }

  if (auto *structType = dyn_cast<StructTypeNode>(node)) {
    for (auto &field : structType->fields())
      if (containsComptimeParameter(field->typeNode(), parameters))
        return true;
  }

  return false;
}

auto substituteBlockExpr(const BlockExpr *node,
                         const std::vector<TypeSubstitution> &substitutions)
    -> std::unique_ptr<BlockExpr> {
  VectorUniquePtr<Stat> statements;
  for (auto &statement : node->statements()) {
    if (auto *variable = dyn_cast<VariableStat>(statement.get())) {
      auto clonedVariable = std::make_unique<VariableExpr>(
          variable->variable()->location(), variable->variable()->name());
      auto clonedInit = variable->init()
                            ? substituteExpr(variable->init().get(),
                                             substitutions)
                            : nullptr;
      statements.push_back(std::make_unique<VariableStat>(
          variable->location(), std::move(clonedVariable),
          substituteTypeNode(variable->typeNode(), substitutions),
          std::move(clonedInit), variable->isConst()));
      continue;
    }

    auto *exprStat = cast<ExprStat>(statement.get());
    statements.push_back(std::make_unique<ExprStat>(
        exprStat->location(),
        substituteExpr(exprStat->expression().get(), substitutions)));
  }

  return std::make_unique<BlockExpr>(
      node->location(), std::move(statements),
      substituteExpr(node->expression().get(), substitutions));
}

auto substituteExpr(const Expr *node,
                    const std::vector<TypeSubstitution> &substitutions)
    -> std::unique_ptr<Expr> {
  switch (node->getKind()) {
  case Expr::Expr_Unit:
    return std::make_unique<UnitExpr>(node->location());
  case Expr::Expr_DecimalLiteral: {
    auto *expr = cast<DecimalLiteralExpr>(node);
    return std::make_unique<DecimalLiteralExpr>(expr->location(),
                                                expr->value());
  }
  case Expr::Expr_FloatLiteral: {
    auto *expr = cast<FloatLiteralExpr>(node);
    return std::make_unique<FloatLiteralExpr>(expr->location(),
                                              expr->value());
  }
  case Expr::Expr_BoolLiteral: {
    auto *expr = cast<BoolLiteralExpr>(node);
    return std::make_unique<BoolLiteralExpr>(expr->location(), expr->value());
  }
  case Expr::Expr_StringLiteral: {
    auto *expr = cast<StringLiteralExpr>(node);
    return std::make_unique<StringLiteralExpr>(
        expr->location(), std::string(expr->value()));
  }
  case Expr::Expr_CharLiteral: {
    auto *expr = cast<CharLiteralExpr>(node);
    return std::make_unique<CharLiteralExpr>(expr->location(), expr->value());
  }
  case Expr::Expr_ArrayLiteral: {
    auto *expr = cast<ArrayLiteralExpr>(node);
    std::vector<std::unique_ptr<Expr>> elements;
    for (auto &element : expr->getElements())
      elements.push_back(substituteExpr(element.get(), substitutions));
    return std::make_unique<ArrayLiteralExpr>(expr->location(),
                                              std::move(elements));
  }
  case Expr::Expr_Index: {
    auto *expr = cast<IndexExpr>(node);
    std::vector<std::unique_ptr<Expr>> indices;
    for (auto &index : expr->indices())
      indices.push_back(substituteExpr(index.get(), substitutions));
    return std::make_unique<IndexExpr>(
        expr->location(), substituteExpr(expr->base().get(), substitutions),
        std::move(indices));
  }
  case Expr::Expr_Member: {
    auto *expr = cast<MemberExpr>(node);
    return std::make_unique<MemberExpr>(
        expr->location(), substituteExpr(expr->base().get(), substitutions),
        expr->fieldName());
  }
  case Expr::Expr_Variable: {
    auto *expr = cast<VariableExpr>(node);
    return std::make_unique<VariableExpr>(expr->location(), expr->name());
  }
  case Expr::Expr_Assign: {
    auto *expr = cast<AssignExpr>(node);
    return std::make_unique<AssignExpr>(
        expr->location(), substituteExpr(expr->lhs().get(), substitutions),
        substituteExpr(expr->rhs().get(), substitutions));
  }
  case Expr::Expr_Binary: {
    auto *expr = cast<BinaryExpr>(node);
    return std::make_unique<BinaryExpr>(
        expr->location(), expr->opEnum(),
        substituteExpr(expr->lhs().get(), substitutions),
        substituteExpr(expr->rhs().get(), substitutions));
  }
  case Expr::Expr_Block:
    return substituteBlockExpr(cast<BlockExpr>(node), substitutions);
  case Expr::Expr_If: {
    auto *expr = cast<IfExpr>(node);
    auto elseBlock = expr->hasElseBlock()
                         ? substituteBlockExpr(expr->elseBlock().get(),
                                               substitutions)
                         : nullptr;
    return std::make_unique<IfExpr>(
        expr->location(),
        substituteExpr(expr->conditionExpr().get(), substitutions),
        substituteBlockExpr(expr->thenBlock().get(), substitutions),
        std::move(elseBlock));
  }
  case Expr::Expr_While: {
    auto *expr = cast<WhileExpr>(node);
    return std::make_unique<WhileExpr>(
        expr->location(),
        substituteExpr(expr->conditionExpr().get(), substitutions),
        substituteBlockExpr(expr->bodyBlock().get(), substitutions));
  }
  case Expr::Expr_Break:
    return std::make_unique<BreakExpr>(node->location());
  case Expr::Expr_Continue:
    return std::make_unique<ContinueExpr>(node->location());
  case Expr::Expr_For: {
    auto *expr = cast<ForExpr>(node);
    return std::make_unique<ForExpr>(
        expr->location(), expr->variableName(),
        substituteExpr(expr->startExpr().get(), substitutions),
        substituteExpr(expr->endExpr().get(), substitutions),
        substituteBlockExpr(expr->bodyBlock().get(), substitutions));
  }
  case Expr::Expr_TypeLayout: {
    auto *expr = cast<TypeLayoutExpr>(node);
    return std::make_unique<TypeLayoutExpr>(
        expr->location(), expr->query(),
        substituteTypeNode(expr->typeNode(), substitutions));
  }
  case Expr::Expr_HeapAlloc: {
    auto *expr = cast<HeapAllocExpr>(node);
    auto count = expr->count()
                     ? substituteExpr(expr->count().get(), substitutions)
                     : nullptr;
    return std::make_unique<HeapAllocExpr>(
        expr->location(), substituteTypeNode(expr->typeNode(), substitutions),
        std::move(count));
  }
  case Expr::Expr_Deref: {
    auto *expr = cast<DerefExpr>(node);
    return std::make_unique<DerefExpr>(
        expr->location(), substituteExpr(expr->pointer().get(), substitutions));
  }
  case Expr::Expr_TensorPack: {
    auto *expr = cast<TensorPackExpr>(node);
    return std::make_unique<TensorPackExpr>(
        expr->location(), substituteExpr(expr->tensor().get(), substitutions));
  }
  case Expr::Expr_TensorView: {
    auto *expr = cast<TensorViewExpr>(node);
    return std::make_unique<TensorViewExpr>(
        expr->location(),
        substituteExpr(expr->tensorRecord().get(), substitutions));
  }
  case Expr::Expr_Call: {
    auto *expr = cast<CallExpr>(node);
    VectorUniquePtr<Expr> expressions;
    for (auto &argument : expr->expressions())
      expressions.push_back(substituteExpr(argument.get(), substitutions));
    if (expr->hasReceiver()) {
      return std::make_unique<CallExpr>(
          expr->location(), substituteExpr(expr->receiver().get(),
                                           substitutions),
          expr->name(), std::move(expressions));
    }
    return std::make_unique<CallExpr>(
        expr->location(), expr->name(), std::move(expressions));
  }
  case Expr::Expr_TensorZeros:
    return std::make_unique<TensorZerosExpr>(node->location());
  case Expr::Expr_StructLiteral: {
    auto *expr = cast<StructLiteralExpr>(node);
    VectorUniquePtr<Expr> expressions;
    for (auto &argument : expr->expressions())
      expressions.push_back(substituteExpr(argument.get(), substitutions));
    return std::make_unique<StructLiteralExpr>(
        expr->location(), substituteTypeNode(expr->typeNode(), substitutions),
        std::move(expressions));
  }
  }

  llvm_unreachable("Unexpected expression");
}

auto instantiateFunctionDecl(const FunctionDecl *node,
                             std::string_view concreteName,
                             const std::vector<TypeSubstitution> &substitutions)
    -> std::unique_ptr<FunctionDecl> {
  VectorUniquePtr<VariableStat> parameters;
  for (auto &parameter : node->proto()->parameters()) {
    auto variable = std::make_unique<VariableExpr>(
        parameter->variable()->location(), parameter->variable()->name());
    parameters.push_back(std::make_unique<VariableStat>(
        parameter->location(), std::move(variable),
        substituteTypeNode(parameter->typeNode(), substitutions), nullptr));
  }

  auto functionName =
      std::make_unique<FunctionName>(node->proto()->id()->location(),
                                     concreteName);
  auto prototype = std::make_unique<Prototype>(
      node->proto()->location(), std::move(functionName),
      std::move(parameters),
      substituteTypeNode(node->proto()->returnTypeNode(), substitutions));
  return std::make_unique<FunctionDecl>(
      node->location(), std::move(prototype),
      substituteBlockExpr(node->body().get(), substitutions));
}

class SemaImpl {
public:
  SemaImpl(const llvm::SourceMgr &sourceManager)
      : _sourceManager{sourceManager} {
    addBuiltins();
  }

  SemaImpl(const llvm::SourceMgr &sourceManager,
           const std::map<std::string, std::string> &importAliases)
      : _sourceManager{sourceManager}, _importAliases{importAliases} {
    addBuiltins();
  }

  auto sema(Module &node) -> MulberryResult {
    for (auto &decl : node)
      if (sema(decl.get()))
        return failure();

    for (size_t i = 0; i < _instantiatedFunctions.size(); ++i) {
      if (sema(_instantiatedFunctions[i].get()))
        return failure();
    }

    auto declarations = node.takeDeclarations();
    for (auto &function : _instantiatedFunctions)
      declarations.push_back(std::move(function));
    node.setDeclarations(std::move(declarations));

    std::string mainName = node.packageName().empty()
                               ? "main"
                               : std::string(node.packageName()) + ".main";
    auto *mainSignature = lookupFunction(mainName);
    if (!mainSignature ||
        !mainSignature->parameterTypes.empty() ||
        !isUInt64Type(mainSignature->returnType))
      return emitError(llvm::SMLoc{}, diag::undefined_main);
    return success();
  }

private:
  const llvm::SourceMgr &_sourceManager;
  TypeContext _typeContext;
  Symbols _symbols;
  std::map<std::string, const StructType *> _genericStructTypes;
  std::map<std::string, const FunctionSymbol *> _instantiatedFunctionSymbols;
  std::map<std::string, std::string> _functionPackages;
  std::map<std::string, std::string> _genericFunctionPackages;
  std::map<std::string, std::string> _instantiatedFunctionPackages;
  VectorUniquePtr<FunctionDecl> _instantiatedFunctions;
  const std::map<std::string, std::string> &_importAliases =
      emptyImportAliases();
  std::string _currentPackageName;
  int _whileDepth = 0;

  enum class UnitPolicy {
    Allow,
    Reject,
  };

  // Semantic Analysis

  // Declarations
  auto sema(Decl *node) -> MulberryResult;
  auto sema(Prototype *node) -> MulberryResult;
  auto semaFunctionParameters(Prototype *node,
                              std::vector<const Type *> &parameterTypes)
      -> MulberryResult;
  auto bindFunctionParameters(Prototype *node,
                              const FunctionSymbol *signature)
      -> MulberryResult;
  auto semaFunctionSignature(Prototype *node) -> MulberryResult;
  auto sema(FunctionDecl *node) -> MulberryResult;
  auto sema(StructDecl *node) -> MulberryResult;
  auto sema(ComptimeTypeAliasDecl *node) -> MulberryResult;

  // Expressions
  auto sema(Expr *node) -> MulberryResult;
  auto sema(Expr *node, const Type *type) -> MulberryResult;
  auto sema(UnitExpr *node) -> MulberryResult;
  auto sema(BlockExpr *node) -> MulberryResult;
  auto sema(BlockExpr *node, const Type *returnType) -> MulberryResult;
  auto sema(CallExpr *node) -> MulberryResult;
  auto sema(StructLiteralExpr *node) -> MulberryResult;
  auto sema(VariableExpr *node) -> MulberryResult;
  auto sema(MemberExpr *node) -> MulberryResult;
  auto sema(AssignExpr *node) -> MulberryResult;
  auto sema(DecimalLiteralExpr *node) -> MulberryResult;
  auto sema(FloatLiteralExpr *node) -> MulberryResult;
  auto sema(BoolLiteralExpr *node) -> MulberryResult;
  auto sema(StringLiteralExpr *node) -> MulberryResult;
  auto sema(CharLiteralExpr *node) -> MulberryResult;
  auto sema(TypeLayoutExpr *node) -> MulberryResult;
  auto sema(HeapAllocExpr *node) -> MulberryResult;
  auto sema(DerefExpr *node) -> MulberryResult;
  auto sema(TensorZerosExpr *node, const TensorType *type) -> MulberryResult;
  auto sema(TensorPackExpr *node) -> MulberryResult;
  auto sema(TensorViewExpr *node,
            const Type *expectedType = nullptr) -> MulberryResult;
  auto sema(BinaryExpr *node) -> MulberryResult;
  auto semaBinaryOperandsSameType(BinaryExpr *node) -> MulberryResult;
  auto checkAssignable(const Expr *expr) -> MulberryResult;
  auto checkConstTensorUseAsMutable(const Expr *expr) -> MulberryResult;
  auto checkConstTensorBinding(const VariableStat *node,
                               const Type *type) -> MulberryResult;
  auto sema(ArrayLiteralExpr *expr) -> MulberryResult;
  auto semaStdlibListLiteral(ArrayLiteralExpr *expr, const Type *type,
                             const Type *elementType) -> MulberryResult;
  auto sema(ArrayLiteralExpr *expr, const TensorType *type) -> MulberryResult;
  auto sema(IndexExpr *expr) -> MulberryResult;
  auto sema(IfExpr *node) -> MulberryResult;
  auto sema(WhileExpr *node) -> MulberryResult;
  auto sema(BreakExpr *node) -> MulberryResult;
  auto sema(ContinueExpr *node) -> MulberryResult;
  auto sema(ForExpr *node) -> MulberryResult;
  auto semaTensorLiteralElement(Expr *expr, const Type *type)
      -> MulberryResult;
  auto semaGenericCall(CallExpr *node, const GenericFunctionSymbol *symbol,
                       const Type *expectedType = nullptr) -> MulberryResult;
  auto semaMethodCall(CallExpr *node,
                      const Type *expectedType = nullptr) -> MulberryResult;
  auto semaDottedMethodCall(CallExpr *node,
                            const Type *expectedType = nullptr)
      -> MulberryResult;
  auto declareStructMethods(std::string_view ownerName,
                            const VectorUniquePtr<FunctionDecl> &methods,
                            const std::vector<ComptimeParam> &typeParameters,
                            std::string_view packageName) -> MulberryResult;

  // Statements
  auto sema(Stat *node) -> MulberryResult;
  auto sema(VariableStat *node) -> MulberryResult;
  auto sema(ExprStat *node) -> MulberryResult;

  // Errors
  auto emitError(const Node *node, const llvm::Twine &msg) -> MulberryResult {
    _sourceManager.PrintMessage(node->location(),
                                llvm::SourceMgr::DiagKind::DK_Error, msg);
    return failure();
  }

  auto emitError(llvm::SMLoc loc, const llvm::Twine &msg) -> MulberryResult {
    _sourceManager.PrintMessage(loc, llvm::SourceMgr::DiagKind::DK_Error, msg);
    return failure();
  }

  auto lookupVariable(std::string_view name) -> const VariableSymbol * {
    return _symbols.lookupVariable(name);
  }

  auto addBuiltins() -> void {
    declareBuiltinType(BuiltinTypeKind::Unit);
    declareBuiltinType(BuiltinTypeKind::Bool);
    declareBuiltinType(BuiltinTypeKind::UInt8);
    declareBuiltinType(BuiltinTypeKind::UInt64);
    declareBuiltinType(BuiltinTypeKind::Float32);
  }

  auto lookupFunction(std::string_view name) -> const FunctionSymbol * {
    if (auto *signature = _symbols.lookupFunction(name))
      return signature;

    auto importedName = canonicalizeImportedName(name);
    if (auto *signature = _symbols.lookupFunction(importedName))
      return signature;

    return _symbols.lookupFunction(qualifyCurrentPackageName(name));
  }

  auto resolveFunctionName(std::string_view name) -> std::string {
    if (_symbols.lookupFunction(name))
      return std::string(name);

    auto importedName = canonicalizeImportedName(name);
    if (_symbols.lookupFunction(importedName))
      return importedName;

    auto qualifiedName = qualifyCurrentPackageName(name);
    if (_symbols.lookupFunction(qualifiedName))
      return qualifiedName;

    return {};
  }

  auto lookupGenericFunction(std::string_view name)
      -> const GenericFunctionSymbol * {
    if (auto *genericFunction = _symbols.lookupGenericFunction(name))
      return genericFunction;

    auto importedName = canonicalizeImportedName(name);
    if (auto *genericFunction = _symbols.lookupGenericFunction(importedName))
      return genericFunction;

    return _symbols.lookupGenericFunction(qualifyCurrentPackageName(name));
  }

  static auto emptyImportAliases()
      -> const std::map<std::string, std::string> & {
    static const std::map<std::string, std::string> aliases;
    return aliases;
  }

  auto canonicalizeImportedName(std::string_view name) const -> std::string {
    auto importedName = _importAliases.find(std::string(name));
    if (importedName != _importAliases.end())
      return importedName->second;

    auto dot = name.find('.');
    if (dot == std::string_view::npos)
      return std::string(name);

    auto alias = _importAliases.find(std::string(name.substr(0, dot)));
    if (alias == _importAliases.end())
      return std::string(name);

    std::string fullName = alias->second;
    fullName += ".";
    fullName += name.substr(dot + 1);
    return fullName;
  }

  auto qualifyCurrentPackageName(std::string_view name) const -> std::string {
    if (_currentPackageName.empty() ||
        name.find('.') != std::string_view::npos)
      return std::string(name);

    std::string fullName = _currentPackageName;
    fullName += ".";
    fullName += name;
    return fullName;
  }

  auto lookupType(std::string_view name) -> const Type * {
    if (auto *type = _symbols.lookupType(name))
      return type;

    auto importedName = canonicalizeImportedName(name);
    if (auto *type = _symbols.lookupType(importedName))
      return type;

    return _symbols.lookupType(qualifyCurrentPackageName(name));
  }

  auto lookupStructType(std::string_view name) -> const StructType * {
    return mulberry::getStructType(lookupType(name));
  }

  auto comptimeTypeAliasName(std::string_view name) -> std::string {
    if (_symbols.lookupComptimeTypeAlias(name))
      return std::string(name);

    auto importedName = canonicalizeImportedName(name);
    if (_symbols.lookupComptimeTypeAlias(importedName))
      return importedName;

    auto packageName = qualifyCurrentPackageName(name);
    if (_symbols.lookupComptimeTypeAlias(packageName))
      return packageName;

    return {};
  }

  auto lookupComptimeTypeAlias(std::string_view name)
      -> const ComptimeTypeAliasSymbol * {
    auto aliasName = comptimeTypeAliasName(name);
    if (aliasName.empty())
      return nullptr;
    return _symbols.lookupComptimeTypeAlias(aliasName);
  }

  auto resolveType(const NamedTypeNode *typeNode) -> const Type * {
    auto *type = lookupType(typeNode->name());
    if (!type) {
      emitError(typeNode, diag::undefined_type);
      return nullptr;
    }

    return type;
  }

  auto resolveType(const UnitTypeNode *typeNode) -> const Type * {
    return _typeContext.getBuiltinType(BuiltinTypeKind::Unit);
  }

  auto declareVariable(std::string_view name, const Type *type,
                       bool isConst = false)
      -> MulberryResult {
    return _symbols.declareVariable(name, type, isConst);
  }

  class VariableScope {
  public:
    explicit VariableScope(Symbols &symbols) : _symbols(symbols) {
      _symbols.enterVariableScope();
    }

    ~VariableScope() { _symbols.leaveVariableScope(); }

  private:
    Symbols &_symbols;
  };

  class PackageScope {
  public:
    PackageScope(std::string &currentPackageName, std::string_view packageName)
        : _currentPackageName(currentPackageName),
          _oldPackageName(currentPackageName) {
      _currentPackageName = packageName;
    }

    ~PackageScope() { _currentPackageName = _oldPackageName; }

  private:
    std::string &_currentPackageName;
    std::string _oldPackageName;
  };

  class WhileScope {
  public:
    explicit WhileScope(int &whileDepth) : _whileDepth(whileDepth) {
      ++_whileDepth;
    }

    ~WhileScope() { --_whileDepth; }

  private:
    int &_whileDepth;
  };

  auto declareFunction(std::string_view name,
                       std::vector<const Type *> parameterTypes,
                       const Type *returnType,
                       std::string_view packageName = {})
      -> MulberryResult {
    if (packageName.empty())
      packageName = _currentPackageName;
    _functionPackages[std::string(name)] = std::string(packageName);
    return _symbols.declareFunction(name, std::move(parameterTypes),
                                    returnType);
  }

  auto declareGenericFunction(std::string_view name,
                              const FunctionDecl *decl,
                              std::string_view packageName = {})
      -> MulberryResult {
    if (packageName.empty())
      packageName = _currentPackageName;
    _genericFunctionPackages[std::string(name)] = std::string(packageName);
    return _symbols.declareGenericFunction(name, decl);
  }

  auto declareType(std::string_view name, const Type *type) -> MulberryResult {
    return _symbols.declareType(name, type);
  }

  auto declareBuiltinType(BuiltinTypeKind kind) -> const BuiltinType * {
    auto *type = _typeContext.getBuiltinType(kind);
    declareType(type->name(), type);
    return type;
  }

  auto declareStructType(const StructType *type) -> MulberryResult {
    return declareType(type->name(), type);
  }

  static auto mangleTypeName(std::string name) -> std::string {
    for (auto &character : name) {
      if ((character >= 'a' && character <= 'z') ||
          (character >= 'A' && character <= 'Z') ||
          (character >= '0' && character <= '9'))
        continue;
      character = '_';
    }
    return name;
  }

  auto genericStructName(std::string_view aliasName,
                         const std::vector<ComptimeArgument> &arguments) const
      -> std::string {
    std::string name = mangleTypeName(std::string(aliasName));
    for (auto &argument : arguments) {
      name += "__";
      if (argument.kind == ComptimeArg::Kind::Type)
        name += mangleTypeName(formatType(argument.type));
      else
        name += std::to_string(argument.uint64Value);
    }
    return name;
  }

  auto genericFunctionName(std::string_view name,
                           const Type *argumentType) const -> std::string {
    std::string result = mangleTypeName(std::string(name));
    result += "__";
    result += mangleTypeName(formatType(argumentType));
    return result;
  }

  auto genericFunctionName(
      std::string_view name,
      const std::vector<InferredComptimeArgument> &arguments) const
      -> std::string {
    std::string result = mangleTypeName(std::string(name));
    for (auto &argument : arguments) {
      result += "__";
      if (argument.kind == ComptimeParam::Kind::Type)
        result += mangleTypeName(formatType(argument.type));
      else
        result += std::to_string(*argument.uint64Value);
    }
    return result;
  }

  auto makeInferredComptimeArguments(
      const std::vector<ComptimeParam> &parameters) const
      -> std::vector<InferredComptimeArgument> {
    std::vector<InferredComptimeArgument> arguments;
    for (auto &parameter : parameters) {
      InferredComptimeArgument argument;
      argument.kind = parameter.kind;
      arguments.push_back(std::move(argument));
    }
    return arguments;
  }

  auto comptimeSubstitutions(
      const std::vector<ComptimeParam> &parameters,
      const std::vector<InferredComptimeArgument> &arguments) const
      -> std::vector<TypeSubstitution> {
    std::vector<TypeSubstitution> substitutions;
    for (size_t i = 0; i < parameters.size(); ++i) {
      auto &parameter = parameters[i];
      auto &argument = arguments[i];
      if (parameter.kind == ComptimeParam::Kind::Type) {
        substitutions.push_back(TypeSubstitution{
            parameter.name, argument.typeNode.get(), std::nullopt});
        continue;
      }
      substitutions.push_back(TypeSubstitution{
          parameter.name, nullptr, *argument.uint64Value});
    }
    return substitutions;
  }

  auto comptimeParameterIndex(const std::vector<ComptimeParam> &parameters,
                              std::string_view name) const
      -> std::optional<size_t> {
    for (size_t i = 0; i < parameters.size(); ++i)
      if (parameters[i].name == name)
        return i;
    return std::nullopt;
  }

  auto bindComptimeTypeArgument(
      const Type *type, InferredComptimeArgument &argument,
      llvm::SMLoc location) -> bool {
    if (argument.kind != ComptimeParam::Kind::Type)
      return false;
    if (!argument.type) {
      argument.type = type;
      argument.typeNode = typeToTypeNode(type, location);
      return true;
    }
    return sameType(argument.type, type);
  }

  auto bindComptimeUInt64Argument(uint64_t value,
                                  InferredComptimeArgument &argument) -> bool {
    if (argument.kind != ComptimeParam::Kind::UInt64)
      return false;
    if (!argument.uint64Value) {
      argument.uint64Value = value;
      return true;
    }
    return *argument.uint64Value == value;
  }

  auto matchComptimeArgument(
      const ComptimeArg &pattern, const ComptimeTypeValue &actual,
      const std::vector<ComptimeParam> &parameters,
      std::vector<InferredComptimeArgument> &arguments) -> bool {
    if (pattern.kind() == ComptimeArg::Kind::UInt64)
      return actual.kind() == ComptimeTypeValue::Kind::UInt64 &&
             pattern.uint64Value() == actual.uint64Value();

    if (auto *namedType = dyn_cast<NamedTypeNode>(pattern.typeNode())) {
      if (auto index = comptimeParameterIndex(parameters, namedType->name())) {
        if (actual.kind() == ComptimeTypeValue::Kind::Type)
          return bindComptimeTypeArgument(
              actual.type(), arguments[*index], namedType->location());
        return bindComptimeUInt64Argument(actual.uint64Value(),
                                          arguments[*index]);
      }
    }

    if (actual.kind() != ComptimeTypeValue::Kind::Type)
      return false;
    return matchGenericType(pattern.typeNode(), actual.type(), parameters,
                            arguments);
  }

  auto matchGenericType(const TypeNode *pattern, const Type *actualType,
                        const std::vector<ComptimeParam> &parameters,
                        std::vector<InferredComptimeArgument> &arguments)
      -> bool {
    if (llvm::isa<UnitTypeNode>(pattern))
      return isUnitType(actualType);

    if (auto *namedType = dyn_cast<NamedTypeNode>(pattern)) {
      if (auto index = comptimeParameterIndex(parameters, namedType->name()))
        return bindComptimeTypeArgument(actualType, arguments[*index],
                                        namedType->location());

      auto *patternType = resolveType(namedType);
      return sameType(patternType, actualType);
    }

    if (auto *tensorPattern = dyn_cast<TensorTypeNode>(pattern)) {
      auto *tensorType = getTensorType(actualType);
      return tensorType &&
             tensorPattern->shape() == tensorType->shape() &&
             matchGenericType(tensorPattern->elementTypeNode(),
                              tensorType->elementType(), parameters,
                              arguments);
    }

    if (auto *listPattern = dyn_cast<ListTypeNode>(pattern)) {
      auto *listType = getListType(actualType);
      return listType &&
             matchGenericType(listPattern->elementTypeNode(),
                              listType->elementType(), parameters,
                              arguments);
    }

    if (auto *ptrPattern = dyn_cast<PtrTypeNode>(pattern)) {
      auto *ptrType = getPtrType(actualType);
      return ptrType &&
             matchGenericType(ptrPattern->pointeeTypeNode(),
                              ptrType->pointeeType(), parameters, arguments);
    }

    if (auto *genericPattern = dyn_cast<GenericTypeNode>(pattern)) {
      auto aliasName = comptimeTypeAliasName(genericPattern->name());
      auto *structType = getStructType(actualType);
      auto *origin = structType ? structType->origin() : nullptr;
      auto &patternArguments = genericPattern->arguments();
      if (origin && origin->aliasName() == aliasName) {
        auto &actualArguments = origin->arguments();
        if (patternArguments.size() != actualArguments.size())
          return false;
        for (size_t i = 0; i < patternArguments.size(); ++i)
          if (!matchComptimeArgument(patternArguments[i], actualArguments[i],
                                     parameters, arguments))
            return false;
        return true;
      }

      auto *alias = lookupComptimeTypeAlias(genericPattern->name());
      if (!alias || patternArguments.size() != alias->parameters.size())
        return false;

      std::vector<TypeSubstitution> substitutions;
      for (size_t i = 0; i < patternArguments.size(); ++i) {
        auto &patternArgument = patternArguments[i];
        auto &aliasParameter = alias->parameters[i];
        if (patternArgument.kind() == ComptimeArg::Kind::UInt64) {
          if (aliasParameter.kind != ComptimeParam::Kind::UInt64)
            return false;
          substitutions.push_back(TypeSubstitution{
              aliasParameter.name, nullptr, patternArgument.uint64Value()});
          continue;
        }

        if (aliasParameter.kind != ComptimeParam::Kind::Type)
          return false;

        auto *argumentTypeNode = patternArgument.typeNode();
        if (containsComptimeParameter(argumentTypeNode, parameters)) {
          substitutions.push_back(TypeSubstitution{
              aliasParameter.name, argumentTypeNode, std::nullopt});
          continue;
        }

        auto *argumentType = resolveType(argumentTypeNode);
        if (!argumentType)
          return false;
        auto resolvedArgumentTypeNode =
            typeToTypeNode(argumentType, argumentTypeNode->location());
        substitutions.push_back(TypeSubstitution{
            aliasParameter.name, resolvedArgumentTypeNode.get(),
            std::nullopt});
      }

      // Expand the alias in its defining package before matching its body.
      // This lets generic methods infer T from aliases such as `List<T>`.
      auto aliasBody = substituteTypeNode(alias->bodyTypeNode, substitutions);
      PackageScope packageScope(_currentPackageName, alias->packageName);
      return matchGenericType(aliasBody.get(), actualType, parameters,
                              arguments);
    }

    if (auto *structPattern = dyn_cast<StructTypeNode>(pattern)) {
      auto *structType = getStructType(actualType);
      if (!structType)
        return false;
      auto &patternFields = structPattern->fields();
      auto &actualFields = structType->fields();
      if (patternFields.size() != actualFields.size())
        return false;

      for (size_t i = 0; i < patternFields.size(); ++i) {
        if (patternFields[i]->variable()->name() != actualFields[i].name())
          return false;
        if (!matchGenericType(patternFields[i]->typeNode(),
                              actualFields[i].type(), parameters, arguments))
          return false;
      }
      return true;
    }

    auto *patternType = resolveType(pattern);
    return sameType(patternType, actualType);
  }

  auto matchMethodReceiverType(
      const TypeNode *pattern, const Type *actualType,
      const std::vector<ComptimeParam> &parameters,
      std::vector<InferredComptimeArgument> &arguments) -> bool {
    if (auto *ptrPattern = dyn_cast<PtrTypeNode>(pattern)) {
      if (matchGenericType(pattern, actualType, parameters, arguments))
        return true;
      return matchGenericType(ptrPattern->pointeeTypeNode(), actualType,
                              parameters, arguments);
    }
    return matchGenericType(pattern, actualType, parameters, arguments);
  }

  auto sameCallArgumentType(const Type *parameterType, const Type *actualType,
                            bool allowAddressOf) -> bool {
    if (sameType(parameterType, actualType))
      return true;
    if (!allowAddressOf)
      return false;

    auto *ptrType = getPtrType(parameterType);
    return ptrType && sameType(ptrType->pointeeType(), actualType);
  }

  auto sameReturnType(const Type *returnType, const Type *actualType) -> bool {
    if (sameType(returnType, actualType))
      return true;

    // Pointer reinterpretation stays explicit in source code through helpers
    // such as std.ptr.asUInt8<T>(). The helper's body returns Ptr<T>; MLIRGen
    // materializes the declared Ptr<UInt8> return with mulberry.ptr.cast.
    return getPtrType(returnType) && getPtrType(actualType);
  }

  auto instantiateGenericFunction(const Node *diagnosticNode,
                                  std::string_view name,
                                  const Type *argumentType,
                                  std::string &concreteName)
      -> MulberryResult {
    auto *symbol = lookupGenericFunction(name);
    if (!symbol) {
      auto diagnostic = formatNameDiagnostic(diag::undefined_func, name);
      return emitError(diagnosticNode, diagnostic);
    }

    auto *genericFunction = symbol->decl;
    auto *genericProto = genericFunction->proto().get();
    auto &genericParameters = genericProto->comptimeParameters();
    if (genericParameters.size() != 1 ||
        genericParameters.front().kind != ComptimeParam::Kind::Type)
      return emitError(diagnosticNode, diag::mismatch_type);

    concreteName = genericFunctionName(name, argumentType);

    auto cached = _instantiatedFunctionSymbols.find(concreteName);
    if (cached != _instantiatedFunctionSymbols.end())
      return success();

    auto argumentTypeNode =
        typeToTypeNode(argumentType, genericProto->location());
    auto concreteFunction = instantiateFunctionDecl(
        genericFunction, concreteName,
        std::vector<TypeSubstitution>{
            TypeSubstitution{genericParameters.front().name,
                             argumentTypeNode.get(), std::nullopt}});
    _instantiatedFunctionPackages[concreteName] =
        genericFunctionPackageName(name);

    VariableScope signatureScope(_symbols);
    PackageScope packageScope(_currentPackageName,
                              genericFunctionPackageName(name));
    if (semaFunctionSignature(concreteFunction->proto().get()))
      return failure();
    auto *signature = lookupFunction(concreteName);
    if (!signature)
      return failure();

    _instantiatedFunctionSymbols.insert({concreteName, signature});
    _instantiatedFunctions.push_back(std::move(concreteFunction));
    return success();
  }

  auto functionPackageName(std::string_view name) const -> std::string {
    auto package = _instantiatedFunctionPackages.find(std::string(name));
    if (package != _instantiatedFunctionPackages.end())
      return package->second;
    package = _functionPackages.find(std::string(name));
    if (package != _functionPackages.end())
      return package->second;
    package = _genericFunctionPackages.find(std::string(name));
    if (package != _genericFunctionPackages.end())
      return package->second;
    return packageNameOf(name);
  }

  auto genericFunctionPackageName(std::string_view name) const -> std::string {
    auto package = _genericFunctionPackages.find(std::string(name));
    if (package != _genericFunctionPackages.end())
      return package->second;
    return packageNameOf(name);
  }

  auto rejectUnitType(const TypeNode *typeNode, const Type *type)
      -> MulberryResult {
    if (!isUnitType(type))
      return success();

    return emitError(typeNode, diag::unexpected_unit_type);
  }

  auto rejectUnitElementType(const TypeNode *typeNode, const Type *type)
      -> MulberryResult {
    if (!hasUnitElementType(type))
      return success();

    return emitError(typeNode, diag::unexpected_unit_type);
  }

  auto resolveType(const TensorTypeNode *typeNode) -> const Type * {
    auto *elementType = resolveType(typeNode->elementTypeNode());
    if (!elementType)
      return nullptr;

    return _typeContext.createTensorType(elementType, typeNode->shape());
  }

  auto resolveType(const ListTypeNode *typeNode) -> const Type * {
    auto *elementType = resolveType(typeNode->elementTypeNode());
    if (!elementType)
      return nullptr;

    return _typeContext.createListType(elementType);
  }

  auto resolveType(const PtrTypeNode *typeNode) -> const Type * {
    auto *pointeeType = resolveType(typeNode->pointeeTypeNode());
    if (!pointeeType)
      return nullptr;

    return _typeContext.createPtrType(pointeeType);
  }

  auto resolveStructFields(const StructTypeNode *typeNode)
      -> std::optional<std::vector<StructField>> {
    std::vector<StructField> fields;
    NameSet fieldNames;
    unsigned fieldIndex = 0;
    for (auto &fieldDecl : typeNode->fields()) {
      auto variable = fieldDecl->variable().get();
      auto *fieldType = checkType(fieldDecl->typeNode(), UnitPolicy::Reject);
      if (!fieldType)
        return std::nullopt;
      fieldDecl->setType(fieldType);
      auto fieldName = variable->name();
      if (!declareName(fieldNames, fieldName)) {
        emitError(variable, diag::redefinition_var);
        return std::nullopt;
      }
      fields.push_back(StructField{fieldName, fieldType, fieldIndex++});
    }
    return fields;
  }

  auto resolveType(const StructTypeNode *typeNode, std::string_view name)
      -> const StructType * {
    auto fields = resolveStructFields(typeNode);
    if (!fields)
      return nullptr;
    return _typeContext.createStructType(name, std::move(*fields));
  }

  auto resolveType(const StructTypeNode *typeNode, std::string_view name,
                   ComptimeAliasOrigin origin) -> const StructType * {
    auto fields = resolveStructFields(typeNode);
    if (!fields)
      return nullptr;
    return _typeContext.createStructType(name, std::move(*fields),
                                         std::move(origin));
  }

  auto resolveType(const GenericTypeNode *typeNode) -> const Type * {
    auto aliasName = comptimeTypeAliasName(typeNode->name());
    auto *alias = aliasName.empty()
                      ? nullptr
                      : _symbols.lookupComptimeTypeAlias(aliasName);
    if (!alias) {
      emitError(typeNode, diag::undefined_type);
      return nullptr;
    }
    if (typeNode->arguments().size() != alias->parameters.size()) {
      emitError(typeNode, diag::mismatch_type);
      return nullptr;
    }

    std::vector<ComptimeArgument> arguments;
    std::vector<TypeSubstitution> substitutions;
    for (size_t i = 0; i < typeNode->arguments().size(); ++i) {
      auto &argument = typeNode->arguments()[i];
      auto &parameter = alias->parameters[i];
      if (argument.kind() == ComptimeArg::Kind::Type &&
          parameter.kind == ComptimeParam::Kind::Type) {
        auto *argumentType = resolveType(argument.typeNode());
        if (!argumentType)
          return nullptr;

        auto argumentTypeNode =
            typeToTypeNode(argumentType, argument.typeNode()->location());
        substitutions.push_back(
            TypeSubstitution{parameter.name, argumentTypeNode.get(),
                             std::nullopt});
        ComptimeArgument resolvedArgument;
        resolvedArgument.kind = argument.kind();
        resolvedArgument.type = argumentType;
        resolvedArgument.typeNode = std::move(argumentTypeNode);
        arguments.push_back(std::move(resolvedArgument));
        continue;
      }

      if (argument.kind() == ComptimeArg::Kind::UInt64 &&
          parameter.kind == ComptimeParam::Kind::UInt64) {
        substitutions.push_back(
            TypeSubstitution{parameter.name, nullptr,
                             argument.uint64Value()});
        ComptimeArgument resolvedArgument;
        resolvedArgument.kind = argument.kind();
        resolvedArgument.uint64Value = argument.uint64Value();
        arguments.push_back(std::move(resolvedArgument));
        continue;
      }

      emitError(typeNode, diag::mismatch_type);
      return nullptr;
    }

    // Generic arguments belong to the use site, while the alias body belongs to
    // the definition package. Resolve arguments first, then switch package
    // scope for the substituted alias body.
    auto instantiatedTypeNode =
        substituteTypeNode(alias->bodyTypeNode, substitutions);
    PackageScope packageScope(_currentPackageName, alias->packageName);
    if (auto *structTypeNode =
            dyn_cast<StructTypeNode>(instantiatedTypeNode.get())) {
      auto structName = genericStructName(aliasName, arguments);
      auto cached = _genericStructTypes.find(structName);
      if (cached != _genericStructTypes.end())
        return cached->second;

      auto origin = ComptimeAliasOrigin(
          aliasName, toComptimeTypeValues(arguments));
      auto *structType = resolveType(structTypeNode, structName,
                                     std::move(origin));
      if (!structType)
        return nullptr;
      _genericStructTypes[structName] = structType;
      return structType;
    }

    return resolveType(instantiatedTypeNode.get());
  }

  auto resolveType(const TypeNode *typeNode) -> const Type * {
    if (auto *unitType = dyn_cast<UnitTypeNode>(typeNode))
      return resolveType(unitType);

    if (auto *tensorType = dyn_cast<TensorTypeNode>(typeNode))
      return resolveType(tensorType);

    if (auto *listType = dyn_cast<ListTypeNode>(typeNode))
      return resolveType(listType);

    if (auto *ptrType = dyn_cast<PtrTypeNode>(typeNode))
      return resolveType(ptrType);

    if (auto *genericType = dyn_cast<GenericTypeNode>(typeNode))
      return resolveType(genericType);

    if (auto *structType = dyn_cast<StructTypeNode>(typeNode))
      return resolveType(structType, "$anonymous");

    return resolveType(cast<NamedTypeNode>(typeNode));
  }

  auto checkType(const TypeNode *typeNode, UnitPolicy unitPolicy)
      -> const Type * {
    auto *type = resolveType(typeNode);
    if (!type)
      return nullptr;
    if (unitPolicy == UnitPolicy::Reject && rejectUnitType(typeNode, type))
      return nullptr;
    if (rejectUnitElementType(typeNode, type))
      return nullptr;
    return type;
  }

  auto isTensorParameter(const FunctionSymbol *signature, size_t index) -> bool {
    auto *parameterType = signature->parameterTypes[index];
    if (parameterType)
      return mulberry::isTensorType(parameterType);
    return false;
  }

  auto setBuiltinType(Expr *expr, BuiltinTypeKind kind) -> void {
    expr->setType(_typeContext.getBuiltinType(kind));
  }

  static auto stdlibListElementType(const Type *type) -> const Type * {
    auto *structType = mulberry::getStructType(type);
    if (!structType)
      return nullptr;

    auto *origin = structType->origin();
    if (!origin || origin->aliasName() != "std.collections.List")
      return nullptr;

    auto &fields = structType->fields();
    if (fields.size() != 3)
      return nullptr;
    if (fields[0].name() != "length" || !isUInt64Type(fields[0].type()))
      return nullptr;
    if (fields[1].name() != "capacity" || !isUInt64Type(fields[1].type()))
      return nullptr;
    if (fields[2].name() != "data")
      return nullptr;

    auto *dataPtrType = mulberry::getPtrType(fields[2].type());
    if (!dataPtrType)
      return nullptr;
    return dataPtrType->pointeeType();
  }

  auto tensorRecordType(const TensorType *type, llvm::SMLoc location)
      -> const Type * {
    std::vector<ComptimeArg> arguments;
    arguments.push_back(ComptimeArg(typeToTypeNode(type->elementType(),
                                                   location)));
    auto typeNode = std::make_unique<GenericTypeNode>(
        location, "std.tensor.Tensor", std::move(arguments));
    return resolveType(typeNode.get());
  }

  auto tensorElementType(const Type *type) -> const Type * {
    auto *structType = mulberry::getStructType(type);
    if (!structType)
      return nullptr;

    auto *origin = structType->origin();
    if (!origin || origin->aliasName() != "std.tensor.Tensor")
      return nullptr;

    auto &arguments = origin->arguments();
    if (arguments.size() != 1)
      return nullptr;
    if (arguments[0].kind() != ComptimeTypeValue::Kind::Type)
      return nullptr;
    return arguments[0].type();
  }

  auto tensorViewType(const Type *type, const Type *expectedType)
      -> const TensorType * {
    auto *elementType = tensorElementType(type);
    auto *expectedTensorType = mulberry::getTensorType(expectedType);
    if (!elementType || !expectedTensorType)
      return nullptr;
    if (!sameType(expectedTensorType->elementType(), elementType))
      return nullptr;
    return expectedTensorType;
  }

};

} // end namespace

auto SemaImpl::sema(Decl *node) -> MulberryResult {
  switch (node->getKind()) {
  case Decl::Decl_Import:
    return success();
  case Decl::Decl_Function:
    return sema(cast<FunctionDecl>(node));
  case Decl::Decl_Struct:
    return sema(cast<StructDecl>(node));
  case Decl::Decl_ComptimeTypeAlias:
    return sema(cast<ComptimeTypeAliasDecl>(node));
  }
}

auto SemaImpl::sema(Prototype *node) -> MulberryResult {
  if (node->isGeneric())
    return success();

  return semaFunctionSignature(node);
}

auto SemaImpl::semaFunctionParameters(
    Prototype *node, std::vector<const Type *> &parameterTypes)
    -> MulberryResult {
  for (auto &par : node->parameters()) {
    auto *parameterType = checkType(par->typeNode(), UnitPolicy::Reject);
    if (!parameterType)
      return failure();
    par->setType(parameterType);
    if (declareVariable(par->variable()->name(), parameterType))
      return emitError(par->variable().get(), diag::redefinition_var);
    parameterTypes.push_back(parameterType);
  }
  return success();
}

auto SemaImpl::bindFunctionParameters(Prototype *node,
                                      const FunctionSymbol *signature)
    -> MulberryResult {
  auto &parameters = node->parameters();
  for (size_t i = 0; i < parameters.size(); ++i) {
    auto &parameter = parameters[i];
    auto *parameterType = signature->parameterTypes[i];
    parameter->setType(parameterType);
    if (declareVariable(parameter->variable()->name(), parameterType))
      return emitError(parameter->variable().get(), diag::redefinition_var);
  }
  node->setType(signature->returnType);
  return success();
}

auto SemaImpl::semaFunctionSignature(Prototype *node) -> MulberryResult {
  std::vector<const Type *> parameterTypes;
  if (semaFunctionParameters(node, parameterTypes))
    return failure();

  auto *returnType = resolveType(node->returnTypeNode());
  if (!returnType)
    return failure();
  node->setType(returnType);

  auto name = node->id()->name();
  if (declareFunction(name, std::move(parameterTypes), returnType)) {
    auto diagnostic = formatNameDiagnostic(diag::redefinition_func, name);
    return emitError(node->id().get(), diagnostic);
  }
  return success();
}

auto SemaImpl::sema(FunctionDecl *node) -> MulberryResult {
  auto functionPackage = node->isExtern()
                             ? _currentPackageName
                             : functionPackageName(node->proto()->id()->name());
  PackageScope packageScope(_currentPackageName, functionPackage);
  if (node->isExtern()) {
    if (node->proto()->isGeneric())
      return emitError(node->proto()->id().get(), diag::mismatch_type);
    _symbols.resetVariables();
    auto result = sema(node->proto().get());
    _symbols.resetVariables();
    return result;
  }

  if (node->proto()->isGeneric()) {
    auto name = node->proto()->id()->name();
    if (lookupFunction(name) || lookupGenericFunction(name)) {
      auto diagnostic = formatNameDiagnostic(diag::redefinition_func, name);
      return emitError(node->proto()->id().get(), diagnostic);
    }
    if (declareGenericFunction(name, node)) {
      auto diagnostic = formatNameDiagnostic(diag::redefinition_func, name);
      return emitError(node->proto()->id().get(), diagnostic);
    }
    return success();
  }

  _symbols.resetVariables();
  auto *signature = lookupFunction(node->proto()->id()->name());
  if (signature) {
    if (bindFunctionParameters(node->proto().get(), signature))
      return failure();
  } else {
    if (sema(node->proto().get()))
      return failure();
    signature = lookupFunction(node->proto()->id()->name());
    if (!signature)
      return failure();
  }
  if (sema(node->body().get(), signature->returnType))
    return failure();

  auto expression = node->body()->expression().get();
  if (!sameReturnType(signature->returnType, expression->type()))
    return emitError(expression, diag::wrong_return_type);

  return success();
}

auto SemaImpl::declareStructMethods(
    std::string_view ownerName, const VectorUniquePtr<FunctionDecl> &methods,
    const std::vector<ComptimeParam> &typeParameters,
    std::string_view packageName)
    -> MulberryResult {
  NameSet methodNames;
  for (auto &method : methods) {
    auto *prototype = method->proto().get();
    auto methodName = prototype->id()->name();
    if (!declareName(methodNames, methodName)) {
      auto diagnostic = formatNameDiagnostic(diag::redefinition_func,
                                             methodName);
      return emitError(prototype->id().get(), diagnostic);
    }

    auto fullName = methodFunctionName(ownerName, methodName);
    prototype->id()->setName(fullName);
    if (!prototype->isGeneric() && !typeParameters.empty())
      prototype->setComptimeParameters(
          std::vector<ComptimeParam>(typeParameters.begin(),
                                     typeParameters.end()));

    if (prototype->isGeneric()) {
      if (declareGenericFunction(fullName, method.get(), packageName)) {
        auto diagnostic = formatNameDiagnostic(diag::redefinition_func,
                                               fullName);
        return emitError(prototype->id().get(), diagnostic);
      }
      continue;
    }

    _functionPackages[fullName] = std::string(packageName);
    if (sema(method.get()))
      return failure();
  }
  return success();
}

auto SemaImpl::sema(StructDecl *node) -> MulberryResult {
  PackageScope packageScope(_currentPackageName,
                            packageNameOf(node->id()->name()));
  std::vector<StructField> fields;
  NameSet fieldNames;
  unsigned fieldIndex = 0;
  for (auto &varDecl : *node) {
    auto var = varDecl->variable().get();
    auto *fieldType = checkType(varDecl->typeNode(), UnitPolicy::Reject);
    if (!fieldType)
      return failure();
    varDecl->setType(fieldType);
    auto fieldName = var->name();
    if (!declareName(fieldNames, fieldName))
      return emitError(var, diag::redefinition_var);
    fields.push_back(StructField{fieldName, fieldType, fieldIndex++});
  }
  auto id = node->id().get();
  if (lookupType(id->name()))
    return emitError(id, diag::redefinition_type);

  auto *structType =
      _typeContext.createStructType(id->name(), std::move(fields));
  id->setType(structType);
  if (declareStructType(structType))
    return emitError(id, diag::redefinition_type);
  if (declareStructMethods(id->name(), node->methods(), {},
                           packageNameOf(id->name())))
    return failure();
  return success();
}

auto SemaImpl::sema(ComptimeTypeAliasDecl *node) -> MulberryResult {
  auto packageName = packageNameOf(node->name());
  PackageScope packageScope(_currentPackageName, packageName);
  if (_symbols.lookupType(node->name()) ||
      _symbols.lookupComptimeTypeAlias(node->name()))
    return emitError(node, diag::redefinition_type);

  if (!node->isGeneric()) {
    auto *bodyType = checkType(node->bodyTypeNode(), UnitPolicy::Reject);
    if (!bodyType)
      return failure();
    if (_symbols.declareType(node->name(), bodyType))
      return emitError(node, diag::redefinition_type);
    if (auto *structTypeNode = dyn_cast<StructTypeNode>(node->bodyTypeNode()))
      return declareStructMethods(node->name(), structTypeNode->methods(), {},
                                  packageName);
    return success();
  }

  if (_symbols.declareComptimeTypeAlias(
          node->name(), packageName,
          std::vector<ComptimeParam>(node->parameters().begin(),
                                     node->parameters().end()),
          node->bodyTypeNode()))
    return emitError(node, diag::redefinition_type);
  if (auto *structTypeNode = dyn_cast<StructTypeNode>(node->bodyTypeNode())) {
    if (declareStructMethods(node->name(), structTypeNode->methods(),
                             node->parameters(), packageName))
      return failure();
  }
  return success();
}

auto SemaImpl::sema(Expr *node) -> MulberryResult {
  switch (node->getKind()) {
  case Expr::Expr_Unit:
    return sema(cast<UnitExpr>(node));
  case Expr::Expr_DecimalLiteral:
    return sema(cast<DecimalLiteralExpr>(node));
  case Expr::Expr_FloatLiteral:
    return sema(cast<FloatLiteralExpr>(node));
  case Expr::Expr_BoolLiteral:
    return sema(cast<BoolLiteralExpr>(node));
  case Expr::Expr_StringLiteral:
    return sema(cast<StringLiteralExpr>(node));
  case Expr::Expr_CharLiteral:
    return sema(cast<CharLiteralExpr>(node));
  case Expr::Expr_TypeLayout:
    return sema(cast<TypeLayoutExpr>(node));
  case Expr::Expr_HeapAlloc:
    return sema(cast<HeapAllocExpr>(node));
  case Expr::Expr_Deref:
    return sema(cast<DerefExpr>(node));
  case Expr::Expr_TensorZeros:
    return emitError(node, diag::mismatch_type);
  case Expr::Expr_TensorPack:
    return sema(cast<TensorPackExpr>(node));
  case Expr::Expr_TensorView:
    return sema(cast<TensorViewExpr>(node));
  case Expr::Expr_Call:
    return sema(cast<CallExpr>(node));
  case Expr::Expr_StructLiteral:
    return sema(cast<StructLiteralExpr>(node));
  case Expr::Expr_Variable:
    return sema(cast<VariableExpr>(node));
  case Expr::Expr_Member:
    return sema(cast<MemberExpr>(node));
  case Expr::Expr_Assign:
    return sema(cast<AssignExpr>(node));
  case Expr::Expr_ArrayLiteral:
    return sema(cast<ArrayLiteralExpr>(node));
  case Expr::Expr_Index:
    return sema(cast<IndexExpr>(node));
  case Expr::Expr_Binary:
    return sema(cast<BinaryExpr>(node));
  case Expr::Expr_If:
    return sema(cast<IfExpr>(node));
  case Expr::Expr_While:
    return sema(cast<WhileExpr>(node));
  case Expr::Expr_Break:
    return sema(cast<BreakExpr>(node));
  case Expr::Expr_Continue:
    return sema(cast<ContinueExpr>(node));
  case Expr::Expr_For:
    return sema(cast<ForExpr>(node));
  default:
    llvm_unreachable("Unexpected expression");
  }
}

auto SemaImpl::sema(Expr *node, const Type *type) -> MulberryResult {
  auto *arrayLiteral = dyn_cast<ArrayLiteralExpr>(node);
  if (arrayLiteral) {
    // Source `[...]` is neutral syntax. Expected type decides whether it is a
    // Tensor literal or a stdlib List alias; other expressions stay bottom-up.
    if (auto *tensorType = mulberry::getTensorType(type))
      return sema(arrayLiteral, tensorType);
    if (auto *elementType = stdlibListElementType(type))
      return semaStdlibListLiteral(arrayLiteral, type, elementType);
  }

  auto *tensorZeros = dyn_cast<TensorZerosExpr>(node);
  if (tensorZeros) {
    auto *tensorType = mulberry::getTensorType(type);
    if (!tensorType)
      return emitError(tensorZeros, diag::mismatch_type);
    return sema(tensorZeros, tensorType);
  }

  if (auto *tensorView = dyn_cast<TensorViewExpr>(node))
    return sema(tensorView, type);

  if (auto *ifExpr = dyn_cast<IfExpr>(node)) {
    if (!ifExpr->hasElseBlock() || isUnitType(type))
      return sema(ifExpr);

    auto conditionExpr = ifExpr->conditionExpr().get();
    if (sema(conditionExpr))
      return failure();
    if (!isBoolType(conditionExpr->type()))
      return emitError(conditionExpr, diag::expected_bool);

    if (sema(ifExpr->thenBlock().get(), type) ||
        sema(ifExpr->elseBlock().get(), type))
      return failure();

    auto *thenType = ifExpr->thenBlock()->expression()->type();
    auto *elseType = ifExpr->elseBlock()->expression()->type();
    if (!sameType(thenType, elseType))
      return emitError(ifExpr->elseBlock()->expression().get(),
                       diag::mismatch_type_then_else);
    if (!sameType(type, elseType))
      return emitError(ifExpr, diag::mismatch_type);
    ifExpr->setType(type);
    return success();
  }

  auto *call = dyn_cast<CallExpr>(node);
  if (call && call->hasReceiver())
    return semaMethodCall(call, type);

  if (call) {
    auto name = canonicalizeImportedName(call->name());
    if (auto *genericFunction = lookupGenericFunction(name)) {
      call->setName(name);
      return semaGenericCall(call, genericFunction, type);
    }
  }

  return sema(node);
}

auto SemaImpl::semaGenericCall(CallExpr *node,
                               const GenericFunctionSymbol *symbol,
                               const Type *expectedType)
    -> MulberryResult {
  auto *genericFunction = symbol->decl;
  auto *genericProto = genericFunction->proto().get();
  auto name = genericProto->id()->name();
  PackageScope packageScope(_currentPackageName,
                            genericFunctionPackageName(name));
  auto &expressions = node->expressions();
  auto &parameters = genericProto->parameters();
  if (expressions.size() != parameters.size()) {
    auto diagnostic =
        formatNameSizeDiagnostic(diag::func_param, name, parameters.size());
    return emitError(node, diagnostic);
  }

  for (auto &argument : expressions) {
    if (sema(argument.get()))
      return failure();
  }

  auto &comptimeParameters = genericProto->comptimeParameters();
  auto inferredArguments =
      makeInferredComptimeArguments(comptimeParameters);
  for (size_t i = 0; i < expressions.size(); ++i) {
    auto *parameterTypeNode = parameters[i]->typeNode();
    auto matched =
        node->isLoweredMethodCall() && i == 0
            ? matchMethodReceiverType(parameterTypeNode, expressions[i]->type(),
                                      comptimeParameters, inferredArguments)
            : matchGenericType(parameterTypeNode, expressions[i]->type(),
                               comptimeParameters, inferredArguments);
    if (!matched)
      return emitError(expressions[i].get(), diag::mismatch_type);
  }

  if (expectedType &&
      !matchGenericType(genericProto->returnTypeNode(), expectedType,
                        comptimeParameters, inferredArguments))
    return emitError(node, diag::mismatch_type);

  for (auto &argument : inferredArguments)
    if (!argument.isResolved())
      return emitError(node, diag::mismatch_type);

  auto concreteName = genericFunctionName(name, inferredArguments);
  auto cached = _instantiatedFunctionSymbols.find(concreteName);
  if (cached == _instantiatedFunctionSymbols.end()) {
    auto substitutions = comptimeSubstitutions(
        comptimeParameters, inferredArguments);
    auto concreteFunction = instantiateFunctionDecl(
        genericFunction, concreteName, substitutions);
    _instantiatedFunctionPackages[concreteName] =
        genericFunctionPackageName(name);

    VariableScope signatureScope(_symbols);
    if (semaFunctionSignature(concreteFunction->proto().get()))
      return failure();
    auto *signature = lookupFunction(concreteName);
    if (!signature)
      return failure();

    cached = _instantiatedFunctionSymbols
                 .insert({concreteName, signature})
                 .first;
    _instantiatedFunctions.push_back(std::move(concreteFunction));
  }

  auto *signature = cached->second;
  for (size_t i = 0; i < expressions.size(); ++i) {
    auto &arg = expressions[i];
    auto *parameterType = signature->parameterTypes[i];
    if (!sameCallArgumentType(parameterType, arg->type(),
                              node->isLoweredMethodCall() && i == 0))
      return emitError(arg.get(), diag::mismatch_type);
    if (isTensorParameter(signature, i) &&
        checkConstTensorUseAsMutable(arg.get()))
      return failure();
  }

  node->setName(concreteName);
  if (signature->returnType)
    node->setType(signature->returnType);
  return success();
}

auto SemaImpl::sema(UnitExpr *node) -> MulberryResult {
  setBuiltinType(node, BuiltinTypeKind::Unit);
  return success();
}

auto SemaImpl::sema(BlockExpr *node) -> MulberryResult {
  VariableScope blockScope(_symbols);
  for (auto &expr : *node)
    if (sema(expr.get()))
      return failure();

  if (sema(node->expression().get()))
    return failure();
  node->setType(node->expression()->type());
  return success();
}

auto SemaImpl::sema(BlockExpr *node, const Type *returnType)
    -> MulberryResult {
  VariableScope blockScope(_symbols);
  for (auto &expr : *node)
    if (sema(expr.get()))
      return failure();

  if (sema(node->expression().get(), returnType))
    return failure();
  node->setType(node->expression()->type());
  return success();
}

auto SemaImpl::sema(CallExpr *node) -> MulberryResult {
  if (node->hasReceiver())
    return semaMethodCall(node);

  node->setName(canonicalizeImportedName(node->name()));
  auto name = node->name();

  auto *signature = lookupFunction(name);
  if (!signature) {
    if (auto *genericFunction = lookupGenericFunction(name))
      return semaGenericCall(node, genericFunction);

    if (name.find('.') != std::string_view::npos)
      return semaDottedMethodCall(node);

    auto diagnostic = formatNameDiagnostic(diag::undefined_func, name);
    return emitError(node, diagnostic);
  }

  auto &expressions = node->expressions();
  if (expressions.size() != signature->parameterTypes.size()) {
    auto diagnostic = formatNameSizeDiagnostic(
        diag::func_param, name, signature->parameterTypes.size());
    return emitError(node, diagnostic);
  }

  for (size_t i = 0; i < expressions.size(); ++i) {
    auto &arg = expressions[i];
    auto *parameterType = signature->parameterTypes[i];
    if (node->isLoweredMethodCall() && i == 0) {
      if (sema(arg.get()))
        return failure();
    } else if (sema(arg.get(), parameterType)) {
      return failure();
    }
    if (!sameCallArgumentType(parameterType, arg->type(),
                              node->isLoweredMethodCall() && i == 0))
      return emitError(arg.get(), diag::mismatch_type);
    if (isTensorParameter(signature, i) &&
        checkConstTensorUseAsMutable(arg.get()))
      return failure();
  }

  auto resolvedName = resolveFunctionName(name);
  if (!resolvedName.empty())
    node->setName(resolvedName);

  if (signature->returnType)
    node->setType(signature->returnType);
  return success();
}

auto SemaImpl::semaMethodCall(CallExpr *node, const Type *expectedType)
    -> MulberryResult {
  if (!node->hasReceiver())
    return semaDottedMethodCall(node, expectedType);

  if (sema(node->receiver().get()))
    return failure();

  auto *receiverType = node->receiver()->type();
  auto *ptrType = mulberry::getPtrType(receiverType);
  auto *structType = ptrType ? mulberry::getStructType(ptrType->pointeeType())
                             : mulberry::getStructType(receiverType);
  if (!structType)
    return emitError(node->receiver().get(), diag::mismatch_type);

  std::vector<std::string> owners;
  if (auto *origin = structType->origin())
    owners.push_back(std::string(origin->aliasName()));
  owners.push_back(std::string(structType->name()));

  auto methodName = std::string(node->name());
  for (auto &owner : owners) {
    auto fullName = methodFunctionName(owner, methodName);
    if (auto *genericFunction = lookupGenericFunction(fullName)) {
      node->lowerMethodCall(fullName);
      return semaGenericCall(node, genericFunction, expectedType);
    }

    if (lookupFunction(fullName)) {
      node->lowerMethodCall(fullName);
      return sema(node);
    }
  }

  auto diagnostic = formatNameDiagnostic(diag::undefined_func, methodName);
  return emitError(node, diagnostic);
}

auto SemaImpl::semaDottedMethodCall(CallExpr *node, const Type *expectedType)
    -> MulberryResult {
  auto name = std::string(node->name());
  auto dot = name.rfind('.');
  if (dot == std::string::npos) {
    auto diagnostic = formatNameDiagnostic(diag::undefined_func, name);
    return emitError(node, diagnostic);
  }

  auto receiverName = name.substr(0, dot);
  auto methodName = name.substr(dot + 1);
  node->setReceiver(createMemberAccessChain(node->location(), receiverName),
                    methodName);
  return semaMethodCall(node, expectedType);
}

auto SemaImpl::sema(StructLiteralExpr *node) -> MulberryResult {
  auto *type = resolveType(node->typeNode());
  auto *structType = mulberry::getStructType(type);
  if (!structType)
    return emitError(node, diag::undefined_type);
  node->setStructType(structType);

  auto &expressions = node->expressions();
  auto &fields = structType->fields();
  if (expressions.size() != fields.size())
    return emitError(node, diag::wrong_num_arg);

  for (size_t i = 0; i < expressions.size(); ++i) {
    auto &expr = expressions[i];
    auto &field = fields[i];
    if (sema(expr.get(), field.type()))
      return failure();
    if (!sameType(field.type(), expr->type()))
      return emitError(expr.get(), diag::mismatch_type);
  }

  node->setType(structType);
  return success();
}

auto SemaImpl::sema(VariableExpr *node) -> MulberryResult {
  auto *symbol = lookupVariable(node->name());
  if (!symbol)
    return emitError(node, diag::undefined_var);
  node->setType(symbol->type);
  return success();
}

auto SemaImpl::sema(MemberExpr *node) -> MulberryResult {
  if (sema(node->base().get()))
    return failure();

  auto *baseType = node->base()->type();
  auto *ptrType = mulberry::getPtrType(baseType);
  auto *structType = ptrType ? mulberry::getStructType(ptrType->pointeeType())
                             : mulberry::getStructType(baseType);
  if (!structType)
    return emitError(node->base().get(), diag::mismatch_type);

  auto *field = structType->field(node->fieldName());
  if (!field)
    return emitError(node, diag::undefined_field);
  if (!field->type())
    return emitError(node, diag::mismatch_type);

  auto *baseIndex = llvm::dyn_cast<IndexExpr>(node->base().get());
  auto isStdlibListElement =
      baseIndex &&
      baseIndex->indexKind() == IndexExpr::IndexKind::StdlibList;

  node->setType(field->type());
  node->setFieldIndex(field->index());
  // `xs[i]` is assignable as a whole through List.set.  Its field access
  // is only a read from the value returned by List.get until the
  // language has element references.
  node->setLvalue(!isStdlibListElement &&
                  (ptrType || node->base()->isLvalue()));
  return success();
}

auto SemaImpl::sema(DecimalLiteralExpr *node) -> MulberryResult {
  setBuiltinType(node, BuiltinTypeKind::UInt64);
  return success();
}

auto SemaImpl::sema(FloatLiteralExpr *node) -> MulberryResult {
  setBuiltinType(node, BuiltinTypeKind::Float32);
  return success();
}

auto SemaImpl::sema(BoolLiteralExpr *node) -> MulberryResult {
  setBuiltinType(node, BuiltinTypeKind::Bool);
  return success();
}

auto SemaImpl::sema(StringLiteralExpr *node) -> MulberryResult {
  auto *type = lookupType("String");
  if (!type)
    return emitError(node, diag::undefined_type);
  node->setType(type);
  return success();
}

auto SemaImpl::sema(CharLiteralExpr *node) -> MulberryResult {
  setBuiltinType(node, BuiltinTypeKind::UInt8);
  return success();
}

auto SemaImpl::sema(TypeLayoutExpr *node) -> MulberryResult {
  auto *queriedType = resolveType(node->typeNode());
  if (!queriedType)
    return failure();

  auto value = node->query() == TypeLayoutExpr::Query::SizeOf
                   ? sizeOfType(queriedType)
                   : alignOfType(queriedType);
  if (!value)
    return emitError(node, diag::unsupported_type_layout);

  node->setQueriedType(queriedType);
  node->setValue(*value);
  setBuiltinType(node, BuiltinTypeKind::UInt64);
  return success();
}

auto SemaImpl::sema(HeapAllocExpr *node) -> MulberryResult {
  auto *allocatedType = checkType(node->typeNode(), UnitPolicy::Reject);
  if (!allocatedType)
    return failure();

  if (node->count()) {
    if (sema(node->count().get()))
      return failure();
    if (!isUInt64Type(node->count()->type()))
      return emitError(node->count().get(), diag::mismatch_type);
  }

  node->setAllocatedType(allocatedType);
  node->setType(_typeContext.createPtrType(allocatedType));
  return success();
}

auto SemaImpl::sema(DerefExpr *node) -> MulberryResult {
  if (sema(node->pointer().get()))
    return failure();

  auto *ptrType = mulberry::getPtrType(node->pointer()->type());
  if (!ptrType)
    return emitError(node->pointer().get(), diag::mismatch_type);

  node->setType(ptrType->pointeeType());
  return success();
}

auto SemaImpl::sema(AssignExpr *node) -> MulberryResult {
  if (sema(node->lhs().get()) ||
      sema(node->rhs().get(), node->lhs()->type()))
    return failure();
  if (!sameType(node->lhs()->type(), node->rhs()->type()))
    return emitError(node->lhs().get(), diag::mismatch_type);
  if (!node->lhs()->isLvalue())
    return emitError(node->lhs().get(), diag::expected_lvalue);
  if (checkAssignable(node->lhs().get()))
    return failure();
  if (mulberry::isTensorType(node->lhs()->type()) &&
      checkConstTensorUseAsMutable(node->rhs().get()))
    return failure();
  setBuiltinType(node, BuiltinTypeKind::Unit);
  return success();
}

auto SemaImpl::sema(BinaryExpr *node) -> MulberryResult {
  using Operator = BinaryExpr::Operator;
  if (semaBinaryOperandsSameType(node))
    return llvm::failure();

  auto *lhsType = node->lhs()->type();
  switch (node->opEnum()) {
  case Operator::Add:
  case Operator::Mul:
  case Operator::Diff:
  case Operator::Div: {
    if (!isNumericType(lhsType))
      return emitError(node->lhs().get(), diag::mismatch_type);
    if (lhsType)
      node->setType(lhsType);
    return success();
  }
  case Operator::Rem: {
    if (!isUInt64Type(lhsType))
      return emitError(node->lhs().get(), diag::mismatch_type);
    setBuiltinType(node, BuiltinTypeKind::UInt64);
    return success();
  }
  case Operator::And:
  case Operator::Or: {
    if (!isBoolType(lhsType))
      return emitError(node->lhs().get(), diag::mismatch_type);
    setBuiltinType(node, BuiltinTypeKind::Bool);
    return success();
  }
  case Operator::EQ:
  case Operator::NEQ: {
    if (!isEquatableType(lhsType))
      return emitError(node->lhs().get(), diag::mismatch_type);
    setBuiltinType(node, BuiltinTypeKind::Bool);
    return success();
  }
  case Operator::LT:
  case Operator::LE:
  case Operator::GT:
  case Operator::GE: {
    if (!isNumericType(lhsType))
      return emitError(node->lhs().get(), diag::mismatch_type);
    setBuiltinType(node, BuiltinTypeKind::Bool);
    return success();
  }
  }

  llvm_unreachable("Unexpected BinaryExpr operator");
}

auto SemaImpl::semaBinaryOperandsSameType(BinaryExpr *node) -> MulberryResult {
  if (sema(node->lhs().get()) || sema(node->rhs().get()))
    return failure();
  if (!sameType(node->lhs()->type(), node->rhs()->type()))
    return emitError(node->lhs().get(), diag::mismatch_type);
  return success();
}

auto SemaImpl::checkAssignable(const Expr *expr) -> MulberryResult {
  if (auto *var = llvm::dyn_cast<VariableExpr>(expr)) {
    auto *symbol = lookupVariable(var->name());
    if (!symbol)
      return emitError(var->location(), diag::undefined_var);
    if (symbol->isConst)
      return emitError(var->location(), diag::assign_const);
    return success();
  }

  if (auto *index = llvm::dyn_cast<IndexExpr>(expr)) {
    return checkAssignable(index->base().get());
  }

  if (auto *memberAccess = llvm::dyn_cast<MemberExpr>(expr)) {
    if (!memberAccess->isLvalue())
      return emitError(memberAccess, diag::expected_lvalue);
    return checkAssignable(memberAccess->base().get());
  }

  return success();
}

auto SemaImpl::checkConstTensorUseAsMutable(const Expr *expr) -> MulberryResult {
  auto *var = llvm::dyn_cast<VariableExpr>(expr);
  if (!var)
    return success();

  auto *symbol = lookupVariable(var->name());
  if (!symbol)
    return emitError(var->location(), diag::undefined_var);
  if (symbol->isConst)
    return emitError(var->location(), diag::const_to_mutable);
  return success();
}

auto SemaImpl::checkConstTensorBinding(const VariableStat *node,
                                       const Type *type)
    -> MulberryResult {
  if (node->isConst() || !mulberry::isTensorType(type))
    return success();
  return checkConstTensorUseAsMutable(node->init().get());
}

auto SemaImpl::sema(ArrayLiteralExpr *expr) -> MulberryResult {
  auto &elements = expr->getElements();
  if (elements.empty())
    return emitError(expr, diag::expected_expr);

  if (sema(elements.front().get()))
    return failure();

  auto *firstElementType = elements.front()->type();
  auto *elementType = firstElementType;
  std::vector<int64_t> currentShape{static_cast<int64_t>(elements.size())};

  if (auto *nestedTensorType = mulberry::getTensorType(elementType)) {
    elementType = nestedTensorType->elementType();
    currentShape.insert(currentShape.end(), nestedTensorType->shape().begin(),
                        nestedTensorType->shape().end());
  }

  if (!elementType)
    return emitError(elements.front().get(), diag::mismatch_type);

  for (size_t i = 1; i < elements.size(); ++i) {
    auto &element = elements[i];
    if (sema(element.get()))
      return failure();
    if (!sameType(firstElementType, element->type()))
      return emitError(element.get(), diag::mismatch_type);
  }

  expr->setInferredShape(currentShape);
  auto *tensorType =
      _typeContext.createTensorType(elementType, std::move(currentShape));
  expr->setTensorLiteral();
  expr->setType(tensorType);
  return success();
}

auto SemaImpl::semaStdlibListLiteral(ArrayLiteralExpr *expr,
                                     const Type *type,
                                     const Type *elementType)
    -> MulberryResult {
  auto &elements = expr->getElements();
  for (auto &element : elements) {
    if (sema(element.get(), elementType))
      return failure();
    if (!sameType(elementType, element->type()))
      return emitError(element.get(), diag::mismatch_type);
  }

  // The literal lowers to normal stdlib calls. Pre-instantiating the generic
  // helpers here keeps MLIRGen simple and avoids a separate list-literal IR op.
  std::string withCapacityFunctionName;
  std::string pushFunctionName;
  if (instantiateGenericFunction(expr, "std.collections.withCapacity",
                                 elementType, withCapacityFunctionName))
    return failure();
  if (!elements.empty() &&
      instantiateGenericFunction(expr, "std.collections.List.push",
                                 elementType, pushFunctionName))
    return failure();

  expr->setStdlibListLiteral(elementType, withCapacityFunctionName,
                             pushFunctionName);
  expr->setType(type);
  return success();
}

auto SemaImpl::sema(ArrayLiteralExpr *expr, const TensorType *type)
    -> MulberryResult {
  auto &elements = expr->getElements();
  if (elements.empty())
    return emitError(expr, diag::expected_expr);

  auto &shape = type->shape();
  if (shape.empty())
    return emitError(expr, diag::mismatch_type);

  auto dim = static_cast<int64_t>(elements.size());
  if (shape.front() >= 0 && shape.front() != dim)
    return emitError(expr, diag::mismatch_type);

  std::vector<int64_t> inferredShape{dim};
  if (shape.size() == 1) {
    for (auto &element : elements)
      if (semaTensorLiteralElement(element.get(), type->elementType()))
        return failure();

    expr->setInferredShape(std::move(inferredShape));
    expr->setTensorLiteral();
    expr->setType(type);
    return success();
  }

  auto nestedShape = std::vector<int64_t>(shape.begin() + 1, shape.end());
  auto *nestedType =
      _typeContext.createTensorType(type->elementType(), nestedShape);
  std::vector<int64_t> firstNestedShape;
  for (auto &element : elements) {
    auto *nestedLiteral = dyn_cast<ArrayLiteralExpr>(element.get());
    if (!nestedLiteral)
      return emitError(element.get(), diag::mismatch_type);
    if (sema(nestedLiteral, nestedType))
      return failure();
    if (firstNestedShape.empty()) {
      firstNestedShape = nestedLiteral->getInferredShape();
    } else if (firstNestedShape != nestedLiteral->getInferredShape()) {
      return emitError(nestedLiteral, diag::mismatch_type);
    }
  }

  inferredShape.insert(inferredShape.end(), firstNestedShape.begin(),
                       firstNestedShape.end());
  expr->setInferredShape(std::move(inferredShape));
  expr->setTensorLiteral();
  expr->setType(type);
  return success();
}

auto SemaImpl::semaTensorLiteralElement(Expr *expr, const Type *type)
    -> MulberryResult {
  if (auto *decimal = dyn_cast<DecimalLiteralExpr>(expr)) {
    if (isUInt8Type(type)) {
      if (decimal->value() > 255)
        return emitError(expr, diag::mismatch_type);
      expr->setType(type);
      return success();
    }

    if (isUInt64Type(type)) {
      expr->setType(type);
      return success();
    }
  }

  if (sema(expr, type))
    return failure();
  if (!sameType(type, expr->type()))
    return emitError(expr, diag::mismatch_type);
  return success();
}

auto SemaImpl::sema(TensorZerosExpr *node, const TensorType *type)
    -> MulberryResult {
  // zeros() is target-typed so it can allocate a Tensor without a huge literal.
  // Dynamic-shape zero fill needs loop-based initialization and is a separate
  // operation from the static raw-file buffers needed by the current pipeline.
  for (auto dim : type->shape()) {
    if (dim < 0)
      return emitError(node, diag::mismatch_type);
  }

  node->setType(type);
  return success();
}

auto SemaImpl::sema(IndexExpr *expr) -> MulberryResult {
  if (sema(expr->base().get()))
    return failure();

  if (auto *elementType = stdlibListElementType(expr->base()->type())) {
    if (expr->indices().size() != 1)
      return emitError(expr, diag::mismatch_type);
    auto &index = expr->indices().front();
    if (sema(index.get()))
      return failure();
    if (!isUInt64Type(index->type()))
      return emitError(index.get(), diag::mismatch_type);

    std::string getFunctionName;
    std::string setFunctionName;
    if (instantiateGenericFunction(expr, "std.collections.List.get",
                                   elementType, getFunctionName) ||
        instantiateGenericFunction(expr, "std.collections.List.set",
                                   elementType, setFunctionName))
      return failure();

    expr->setType(elementType);
    expr->setLvalue(true);
    expr->setStdlibListIndex(getFunctionName, setFunctionName);
    return success();
  }

  auto *ptrType = mulberry::getPtrType(expr->base()->type());
  if (ptrType) {
    if (expr->indices().size() != 1)
      return emitError(expr, diag::mismatch_type);
    auto &index = expr->indices().front();
    if (sema(index.get()))
      return failure();
    if (!isUInt64Type(index->type()))
      return emitError(index.get(), diag::mismatch_type);

    expr->setType(ptrType->pointeeType());
    expr->setLvalue(true);
    expr->setPtrIndex();
    return success();
  }

  auto *tensorType = mulberry::getTensorType(expr->base()->type());
  if (!tensorType)
    return emitError(expr, diag::mismatch_type);

  auto tensorRank = tensorType->shape().size();
  if (expr->indices().size() != tensorRank)
    return emitError(expr, diag::mismatch_type);

  for (auto &index : expr->indices()) {
    if (sema(index.get()))
      return failure();
    if (!isUInt64Type(index->type()))
      return emitError(index.get(), diag::mismatch_type);
  }

  expr->setType(tensorType->elementType());
  expr->setLvalue(true);
  expr->setTensorIndex();
  return success();
}

auto SemaImpl::sema(TensorPackExpr *node) -> MulberryResult {
  if (sema(node->tensor().get()))
    return failure();

  auto *tensorType = mulberry::getTensorType(node->tensor()->type());
  if (!tensorType)
    return emitError(node->tensor().get(), diag::mismatch_type);

  auto *recordType = tensorRecordType(tensorType, node->location());
  if (!recordType)
    return failure();

  node->setType(recordType);
  return success();
}

auto SemaImpl::sema(TensorViewExpr *node, const Type *expectedType)
    -> MulberryResult {
  if (sema(node->tensorRecord().get()))
    return failure();

  auto *tensorType = tensorViewType(node->tensorRecord()->type(), expectedType);
  if (!tensorType)
    return emitError(node->tensorRecord().get(), diag::mismatch_type);

  node->setType(tensorType);
  return success();
}

auto SemaImpl::sema(IfExpr *node) -> MulberryResult {
  auto conditionExpr = node->conditionExpr().get();
  if (sema(conditionExpr))
    return failure();
  if (!isBoolType(conditionExpr->type()))
    return emitError(conditionExpr, diag::expected_bool);

  auto thenBlock = node->thenBlock().get();
  if (sema(thenBlock))
    return failure();

  if (!node->hasElseBlock()) {
    setBuiltinType(node, BuiltinTypeKind::Unit);
    return success();
  }

  auto elseBlock = node->elseBlock().get();
  if (sema(elseBlock))
    return failure();

  auto elseExpr = elseBlock->expression().get();
  if (!sameType(thenBlock->expression()->type(), elseExpr->type()))
    return emitError(elseExpr, diag::mismatch_type_then_else);

  if (auto *elseType = elseExpr->type())
    node->setType(elseType);
  return success();
}

auto SemaImpl::sema(WhileExpr *node) -> MulberryResult {
  auto conditionExpr = node->conditionExpr().get();
  if (sema(conditionExpr))
    return failure();
  if (!isBoolType(conditionExpr->type()))
    return emitError(conditionExpr, diag::expected_bool);

  auto bodyBlock = node->bodyBlock().get();
  WhileScope whileScope(_whileDepth);
  if (sema(bodyBlock))
    return failure();

  if (!isUnitType(bodyBlock->expression()->type()))
    return emitError(bodyBlock->expression().get(), diag::mismatch_type);

  setBuiltinType(node, BuiltinTypeKind::Unit);
  return success();
}

auto SemaImpl::sema(BreakExpr *node) -> MulberryResult {
  if (_whileDepth == 0)
    return emitError(node, diag::loop_control_outside_loop);
  setBuiltinType(node, BuiltinTypeKind::Unit);
  return success();
}

auto SemaImpl::sema(ContinueExpr *node) -> MulberryResult {
  if (_whileDepth == 0)
    return emitError(node, diag::loop_control_outside_loop);
  setBuiltinType(node, BuiltinTypeKind::Unit);
  return success();
}

auto SemaImpl::sema(ForExpr *node) -> MulberryResult {
  if (sema(node->startExpr().get()) || sema(node->endExpr().get()))
    return failure();

  if (!isUInt64Type(node->startExpr()->type()))
    return emitError(node->startExpr().get(), diag::mismatch_type);
  if (!isUInt64Type(node->endExpr()->type()))
    return emitError(node->endExpr().get(), diag::mismatch_type);

  VariableScope loopScope(_symbols);
  auto *uint64Type = _typeContext.getBuiltinType(BuiltinTypeKind::UInt64);
  if (declareVariable(node->variableName(), uint64Type, /*isConst=*/true))
    return emitError(node, diag::redefinition_var);

  auto bodyBlock = node->bodyBlock().get();
  if (sema(bodyBlock))
    return failure();

  if (!isUnitType(bodyBlock->expression()->type()))
    return emitError(bodyBlock->expression().get(), diag::mismatch_type);

  setBuiltinType(node, BuiltinTypeKind::Unit);
  return success();
}

auto SemaImpl::sema(Stat *node) -> MulberryResult {
  switch (node->getKind()) {
  case Stat::Stat_VariableDecl:
    return sema(cast<VariableStat>(node));
  case Stat::Stat_Expression:
    return sema(cast<ExprStat>(node));
  }
}

auto SemaImpl::sema(VariableStat *node) -> MulberryResult {
  auto var = node->variable().get();
  auto *varType = checkType(node->typeNode(), UnitPolicy::Allow);
  if (!varType)
    return failure();
  node->setType(varType);
  if (declareVariable(var->name(), varType, node->isConst()))
    return emitError(var, diag::redefinition_var);

  auto initExpr = node->init().get();
  if (sema(initExpr, varType))
    return failure();
  if (!sameType(varType, initExpr->type()))
    return emitError(initExpr, diag::mismatch_type);
  if (checkConstTensorBinding(node, varType))
    return failure();
  return success();
}

auto SemaImpl::sema(ExprStat *node) -> MulberryResult {
  return sema(node->expression().get());
}

namespace mulberry {

auto sema(const llvm::SourceMgr &sourceManager, Module &moduleAST)
    -> MulberryResult {
  return SemaImpl(sourceManager).sema(moduleAST);
}

auto sema(const llvm::SourceMgr &sourceManager, Module &moduleAST,
          const std::map<std::string, std::string> &importAliases)
    -> MulberryResult {
  return SemaImpl(sourceManager, importAliases).sema(moduleAST);
}

} // end namespace mulberry
