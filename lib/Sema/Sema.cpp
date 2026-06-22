//===--- Sema.cpp - Cherry Semantic Analysis ------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/Sema/Sema.h"
#include "Symbols.h"
#include "cherry/AST/AST.h"
#include "cherry/Basic/Builtins.h"
#include "cherry/Sema/DiagnosticsSema.h"
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using namespace cherry;
using llvm::cast;
using llvm::dyn_cast;

using NameSet = std::set<std::string, std::less<>>;

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
    return std::make_unique<GenericTypeNode>(
        genericType->location(), genericType->name(),
        cloneTypeNode(genericType->argumentTypeNode()));
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
  return std::make_unique<StructTypeNode>(
      structType->location(), std::move(fields));
}

auto typeToTypeNode(const Type *type, llvm::SMLoc location)
    -> std::unique_ptr<TypeNode> {
  if (auto *builtinType = getBuiltinType(type))
    return std::make_unique<NamedTypeNode>(location, builtinType->name());

  if (auto *structType = getStructType(type))
    return std::make_unique<NamedTypeNode>(location, structType->name());

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

auto substituteTypeNode(const TypeNode *node, std::string_view parameterName,
                        const TypeNode *argumentTypeNode)
    -> std::unique_ptr<TypeNode> {
  if (auto *namedType = dyn_cast<NamedTypeNode>(node)) {
    if (namedType->name() == parameterName)
      return cloneTypeNode(argumentTypeNode);
    return cloneTypeNode(namedType);
  }

  if (auto *tensorType = dyn_cast<TensorTypeNode>(node)) {
    return std::make_unique<TensorTypeNode>(
        substituteTypeNode(tensorType->elementTypeNode(), parameterName,
                           argumentTypeNode),
        tensorType->shape(), tensorType->location());
  }

  if (auto *listType = dyn_cast<ListTypeNode>(node)) {
    return std::make_unique<ListTypeNode>(
        substituteTypeNode(listType->elementTypeNode(), parameterName,
                           argumentTypeNode),
        listType->location());
  }

  if (auto *ptrType = dyn_cast<PtrTypeNode>(node)) {
    return std::make_unique<PtrTypeNode>(
        substituteTypeNode(ptrType->pointeeTypeNode(), parameterName,
                           argumentTypeNode),
        ptrType->location());
  }

  if (auto *genericType = dyn_cast<GenericTypeNode>(node)) {
    return std::make_unique<GenericTypeNode>(
        genericType->location(), genericType->name(),
        substituteTypeNode(genericType->argumentTypeNode(), parameterName,
                           argumentTypeNode));
  }

  auto *structType = cast<StructTypeNode>(node);
  VectorUniquePtr<VariableStat> fields;
  for (auto &field : structType->fields()) {
    auto variable = std::make_unique<VariableExpr>(
        field->variable()->location(), field->variable()->name());
    fields.push_back(std::make_unique<VariableStat>(
        field->location(), std::move(variable),
        substituteTypeNode(field->typeNode(), parameterName, argumentTypeNode),
        nullptr));
  }
  return std::make_unique<StructTypeNode>(
      structType->location(), std::move(fields));
}

auto hasTypeParameter(const TypeNode *node, std::string_view parameterName)
    -> bool {
  if (auto *namedType = dyn_cast<NamedTypeNode>(node))
    return namedType->name() == parameterName;

  if (auto *tensorType = dyn_cast<TensorTypeNode>(node))
    return hasTypeParameter(tensorType->elementTypeNode(), parameterName);

  if (auto *listType = dyn_cast<ListTypeNode>(node))
    return hasTypeParameter(listType->elementTypeNode(), parameterName);

  if (auto *ptrType = dyn_cast<PtrTypeNode>(node))
    return hasTypeParameter(ptrType->pointeeTypeNode(), parameterName);

  if (auto *genericType = dyn_cast<GenericTypeNode>(node))
    return hasTypeParameter(genericType->argumentTypeNode(), parameterName);

  if (auto *structType = dyn_cast<StructTypeNode>(node)) {
    for (auto &field : structType->fields())
      if (hasTypeParameter(field->typeNode(), parameterName))
        return true;
  }

  return false;
}

auto substituteExpr(const Expr *node, std::string_view parameterName,
                    const TypeNode *argumentTypeNode)
    -> std::unique_ptr<Expr>;

auto substituteBlockExpr(const BlockExpr *node, std::string_view parameterName,
                         const TypeNode *argumentTypeNode)
    -> std::unique_ptr<BlockExpr> {
  VectorUniquePtr<Stat> statements;
  for (auto &statement : node->statements()) {
    if (auto *variable = dyn_cast<VariableStat>(statement.get())) {
      auto clonedVariable = std::make_unique<VariableExpr>(
          variable->variable()->location(), variable->variable()->name());
      auto clonedInit = variable->init()
                            ? substituteExpr(variable->init().get(),
                                             parameterName, argumentTypeNode)
                            : nullptr;
      statements.push_back(std::make_unique<VariableStat>(
          variable->location(), std::move(clonedVariable),
          substituteTypeNode(variable->typeNode(), parameterName,
                             argumentTypeNode),
          std::move(clonedInit), variable->isConst()));
      continue;
    }

    auto *exprStat = cast<ExprStat>(statement.get());
    statements.push_back(std::make_unique<ExprStat>(
        exprStat->location(),
        substituteExpr(exprStat->expression().get(), parameterName,
                       argumentTypeNode)));
  }

  return std::make_unique<BlockExpr>(
      node->location(), std::move(statements),
      substituteExpr(node->expression().get(), parameterName,
                     argumentTypeNode));
}

auto substituteExpr(const Expr *node, std::string_view parameterName,
                    const TypeNode *argumentTypeNode)
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
  case Expr::Expr_ArrayLiteral: {
    auto *expr = cast<ArrayLiteralExpr>(node);
    std::vector<std::unique_ptr<Expr>> elements;
    for (auto &element : expr->getElements())
      elements.push_back(
          substituteExpr(element.get(), parameterName, argumentTypeNode));
    return std::make_unique<ArrayLiteralExpr>(expr->location(),
                                              std::move(elements));
  }
  case Expr::Expr_Index: {
    auto *expr = cast<IndexExpr>(node);
    std::vector<std::unique_ptr<Expr>> indices;
    for (auto &index : expr->indices())
      indices.push_back(
          substituteExpr(index.get(), parameterName, argumentTypeNode));
    return std::make_unique<IndexExpr>(
        expr->location(),
        substituteExpr(expr->base().get(), parameterName, argumentTypeNode),
        std::move(indices));
  }
  case Expr::Expr_Member: {
    auto *expr = cast<MemberExpr>(node);
    return std::make_unique<MemberExpr>(
        expr->location(),
        substituteExpr(expr->base().get(), parameterName, argumentTypeNode),
        expr->fieldName());
  }
  case Expr::Expr_Variable: {
    auto *expr = cast<VariableExpr>(node);
    return std::make_unique<VariableExpr>(expr->location(), expr->name());
  }
  case Expr::Expr_Assign: {
    auto *expr = cast<AssignExpr>(node);
    return std::make_unique<AssignExpr>(
        expr->location(),
        substituteExpr(expr->lhs().get(), parameterName, argumentTypeNode),
        substituteExpr(expr->rhs().get(), parameterName, argumentTypeNode));
  }
  case Expr::Expr_Binary: {
    auto *expr = cast<BinaryExpr>(node);
    return std::make_unique<BinaryExpr>(
        expr->location(), expr->opEnum(),
        substituteExpr(expr->lhs().get(), parameterName, argumentTypeNode),
        substituteExpr(expr->rhs().get(), parameterName, argumentTypeNode));
  }
  case Expr::Expr_Block:
    return substituteBlockExpr(cast<BlockExpr>(node), parameterName,
                               argumentTypeNode);
  case Expr::Expr_If: {
    auto *expr = cast<IfExpr>(node);
    return std::make_unique<IfExpr>(
        expr->location(),
        substituteExpr(expr->conditionExpr().get(), parameterName,
                       argumentTypeNode),
        substituteBlockExpr(expr->thenBlock().get(), parameterName,
                            argumentTypeNode),
        substituteBlockExpr(expr->elseBlock().get(), parameterName,
                            argumentTypeNode));
  }
  case Expr::Expr_While: {
    auto *expr = cast<WhileExpr>(node);
    return std::make_unique<WhileExpr>(
        expr->location(),
        substituteExpr(expr->conditionExpr().get(), parameterName,
                       argumentTypeNode),
        substituteBlockExpr(expr->bodyBlock().get(), parameterName,
                            argumentTypeNode));
  }
  case Expr::Expr_For: {
    auto *expr = cast<ForExpr>(node);
    return std::make_unique<ForExpr>(
        expr->location(), expr->variableName(),
        substituteExpr(expr->startExpr().get(), parameterName,
                       argumentTypeNode),
        substituteExpr(expr->endExpr().get(), parameterName,
                       argumentTypeNode),
        substituteBlockExpr(expr->bodyBlock().get(), parameterName,
                            argumentTypeNode));
  }
  case Expr::Expr_TypeLayout: {
    auto *expr = cast<TypeLayoutExpr>(node);
    return std::make_unique<TypeLayoutExpr>(
        expr->location(), expr->query(),
        substituteTypeNode(expr->typeNode(), parameterName,
                           argumentTypeNode));
  }
  case Expr::Expr_HeapAlloc: {
    auto *expr = cast<HeapAllocExpr>(node);
    auto count = expr->count()
                     ? substituteExpr(expr->count().get(), parameterName,
                                      argumentTypeNode)
                     : nullptr;
    return std::make_unique<HeapAllocExpr>(
        expr->location(),
        substituteTypeNode(expr->typeNode(), parameterName, argumentTypeNode),
        std::move(count));
  }
  case Expr::Expr_Deref: {
    auto *expr = cast<DerefExpr>(node);
    return std::make_unique<DerefExpr>(
        expr->location(),
        substituteExpr(expr->pointer().get(), parameterName,
                       argumentTypeNode));
  }
  case Expr::Expr_Call: {
    auto *expr = cast<CallExpr>(node);
    VectorUniquePtr<Expr> expressions;
    for (auto &argument : expr->expressions())
      expressions.push_back(
          substituteExpr(argument.get(), parameterName, argumentTypeNode));
    return std::make_unique<CallExpr>(
        expr->location(), expr->name(), std::move(expressions));
  }
  case Expr::Expr_StructLiteral: {
    auto *expr = cast<StructLiteralExpr>(node);
    VectorUniquePtr<Expr> expressions;
    for (auto &argument : expr->expressions())
      expressions.push_back(
          substituteExpr(argument.get(), parameterName, argumentTypeNode));
    return std::make_unique<StructLiteralExpr>(
        expr->location(),
        substituteTypeNode(expr->typeNode(), parameterName, argumentTypeNode),
        std::move(expressions));
  }
  }

  llvm_unreachable("Unexpected expression");
}

auto instantiateFunctionDecl(const FunctionDecl *node,
                             std::string_view concreteName,
                             std::string_view parameterName,
                             const Type *argumentType)
    -> std::unique_ptr<FunctionDecl> {
  auto location = node->proto()->location();
  auto argumentTypeNode = typeToTypeNode(argumentType, location);
  VectorUniquePtr<VariableStat> parameters;
  for (auto &parameter : node->proto()->parameters()) {
    auto variable = std::make_unique<VariableExpr>(
        parameter->variable()->location(), parameter->variable()->name());
    parameters.push_back(std::make_unique<VariableStat>(
        parameter->location(), std::move(variable),
        substituteTypeNode(parameter->typeNode(), parameterName,
                           argumentTypeNode.get()),
        nullptr));
  }

  auto functionName =
      std::make_unique<FunctionName>(node->proto()->id()->location(),
                                     concreteName);
  auto prototype = std::make_unique<Prototype>(
      node->proto()->location(), std::move(functionName),
      std::move(parameters),
      substituteTypeNode(node->proto()->returnTypeNode(), parameterName,
                         argumentTypeNode.get()));
  return std::make_unique<FunctionDecl>(
      node->location(), std::move(prototype),
      substituteBlockExpr(node->body().get(), parameterName,
                          argumentTypeNode.get()));
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

  auto sema(Module &node) -> CherryResult {
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
  std::map<std::string, std::string> _instantiatedFunctionPackages;
  VectorUniquePtr<FunctionDecl> _instantiatedFunctions;
  const std::map<std::string, std::string> &_importAliases =
      emptyImportAliases();
  std::string _currentPackageName;

  enum class UnitPolicy {
    Allow,
    Reject,
  };

  // Semantic Analysis

  // Declarations
  auto sema(Decl *node) -> CherryResult;
  auto sema(Prototype *node) -> CherryResult;
  auto semaFunctionParameters(Prototype *node,
                              std::vector<const Type *> &parameterTypes)
      -> CherryResult;
  auto bindFunctionParameters(Prototype *node,
                              const FunctionSymbol *signature)
      -> CherryResult;
  auto semaFunctionSignature(Prototype *node) -> CherryResult;
  auto sema(FunctionDecl *node) -> CherryResult;
  auto sema(StructDecl *node) -> CherryResult;
  auto sema(ComptimeTypeAliasDecl *node) -> CherryResult;

  // Expressions
  auto sema(Expr *node) -> CherryResult;
  auto sema(Expr *node, const Type *type) -> CherryResult;
  auto sema(UnitExpr *node) -> CherryResult;
  auto sema(BlockExpr *node) -> CherryResult;
  auto sema(BlockExpr *node, const Type *returnType) -> CherryResult;
  auto sema(CallExpr *node) -> CherryResult;
  auto sema(StructLiteralExpr *node) -> CherryResult;
  auto sema(VariableExpr *node) -> CherryResult;
  auto sema(MemberExpr *node) -> CherryResult;
  auto sema(AssignExpr *node) -> CherryResult;
  auto sema(DecimalLiteralExpr *node) -> CherryResult;
  auto sema(FloatLiteralExpr *node) -> CherryResult;
  auto sema(BoolLiteralExpr *node) -> CherryResult;
  auto sema(StringLiteralExpr *node) -> CherryResult;
  auto sema(TypeLayoutExpr *node) -> CherryResult;
  auto sema(HeapAllocExpr *node) -> CherryResult;
  auto sema(DerefExpr *node) -> CherryResult;
  auto sema(BinaryExpr *node) -> CherryResult;
  auto semaBinaryOperandsSameType(BinaryExpr *node) -> CherryResult;
  auto checkAssignable(const Expr *expr) -> CherryResult;
  auto checkConstTensorUseAsMutable(const Expr *expr) -> CherryResult;
  auto checkConstTensorBinding(const VariableStat *node,
                               const Type *type) -> CherryResult;
  auto sema(ArrayLiteralExpr *expr) -> CherryResult;
  auto semaStdlibListLiteral(ArrayLiteralExpr *expr, const Type *type,
                             const Type *elementType) -> CherryResult;
  auto sema(ArrayLiteralExpr *expr, const TensorType *type) -> CherryResult;
  auto semaZeros(CallExpr *node, const TensorType *type) -> CherryResult;
  auto sema(IndexExpr *expr) -> CherryResult;
  auto semaMatmul(CallExpr *node) -> CherryResult;
  auto semaTensorBinary(CallExpr *node) -> CherryResult;
  auto semaMatscale(CallExpr *node) -> CherryResult;
  auto semaTranspose(CallExpr *node) -> CherryResult;
  auto semaElementwiseNN(CallExpr *node) -> CherryResult;
  auto semaArgmax(CallExpr *node) -> CherryResult;
  auto semaPrint(CallExpr *node) -> CherryResult;
  auto semaSize(CallExpr *node) -> CherryResult;
  auto semaFileOpen(CallExpr *node) -> CherryResult;
  auto semaFileRead(CallExpr *node) -> CherryResult;
  auto semaReadTensor(CallExpr *node, const TensorType *type) -> CherryResult;
  auto semaFileWrite(CallExpr *node) -> CherryResult;
  auto semaFileClose(CallExpr *node) -> CherryResult;
  auto sema(IfExpr *node) -> CherryResult;
  auto sema(WhileExpr *node) -> CherryResult;
  auto sema(ForExpr *node) -> CherryResult;
  auto semaTensorLiteralElement(Expr *expr, const Type *type)
      -> CherryResult;
  auto semaGenericCall(CallExpr *node, const GenericFunctionSymbol *symbol,
                       const Type *expectedType = nullptr) -> CherryResult;

  // Statements
  auto sema(Stat *node) -> CherryResult;
  auto sema(VariableStat *node) -> CherryResult;
  auto sema(ExprStat *node) -> CherryResult;

  // Errors
  auto emitError(const Node *node, const llvm::Twine &msg) -> CherryResult {
    _sourceManager.PrintMessage(node->location(),
                                llvm::SourceMgr::DiagKind::DK_Error, msg);
    return failure();
  }

  auto emitError(llvm::SMLoc loc, const llvm::Twine &msg) -> CherryResult {
    _sourceManager.PrintMessage(loc, llvm::SourceMgr::DiagKind::DK_Error, msg);
    return failure();
  }

  auto lookupVariable(std::string_view name) -> const VariableSymbol * {
    return _symbols.lookupVariable(name);
  }

  auto addBuiltins() -> void {
    declareBuiltinType(BuiltinTypeKind::Unit);
    auto *boolType = declareBuiltinType(BuiltinTypeKind::Bool);
    declareBuiltinType(BuiltinTypeKind::UInt8);
    auto *uint64Type = declareBuiltinType(BuiltinTypeKind::UInt64);
    declareBuiltinType(BuiltinTypeKind::Float32);
    declareBuiltinType(BuiltinTypeKind::String);
    declareBuiltinType(BuiltinTypeKind::File);

    declareFunction(builtins::builtinPrint,
                    std::vector<const Type *>{uint64Type}, uint64Type);
    declareFunction(builtins::print, std::vector<const Type *>{uint64Type},
                    uint64Type);
    declareFunction(builtins::boolToUInt64,
                    std::vector<const Type *>{boolType}, uint64Type);
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
    return cherry::getStructType(lookupType(name));
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
      -> CherryResult {
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

  auto declareFunction(std::string_view name,
                       std::vector<const Type *> parameterTypes,
                       const Type *returnType)
      -> CherryResult {
    return _symbols.declareFunction(name, std::move(parameterTypes),
                                    returnType);
  }

  auto declareGenericFunction(std::string_view name,
                              const FunctionDecl *decl) -> CherryResult {
    return _symbols.declareGenericFunction(name, decl);
  }

  auto declareType(std::string_view name, const Type *type) -> CherryResult {
    return _symbols.declareType(name, type);
  }

  auto declareBuiltinType(BuiltinTypeKind kind) -> const BuiltinType * {
    auto *type = _typeContext.getBuiltinType(kind);
    declareType(type->name(), type);
    return type;
  }

  auto declareStructType(const StructType *type) -> CherryResult {
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
                         const Type *argumentType) const -> std::string {
    std::string name = mangleTypeName(std::string(aliasName));
    name += "__";
    name += mangleTypeName(formatType(argumentType));
    return name;
  }

  auto genericFunctionName(std::string_view name,
                           const Type *argumentType) const -> std::string {
    std::string result = mangleTypeName(std::string(name));
    result += "__";
    result += mangleTypeName(formatType(argumentType));
    return result;
  }

  auto instantiateGenericFunction(const Node *diagnosticNode,
                                  std::string_view name,
                                  const Type *argumentType,
                                  std::string &concreteName)
      -> CherryResult {
    auto *symbol = lookupGenericFunction(name);
    if (!symbol) {
      auto diagnostic = formatNameDiagnostic(diag::undefined_func, name);
      return emitError(diagnosticNode, diagnostic);
    }

    auto *genericFunction = symbol->decl;
    auto *genericProto = genericFunction->proto().get();
    auto parameterName = genericProto->typeParameterName();
    concreteName = genericFunctionName(name, argumentType);

    auto cached = _instantiatedFunctionSymbols.find(concreteName);
    if (cached != _instantiatedFunctionSymbols.end())
      return success();

    auto concreteFunction = instantiateFunctionDecl(
        genericFunction, concreteName, parameterName, argumentType);
    _instantiatedFunctionPackages[concreteName] = packageNameOf(name);

    VariableScope signatureScope(_symbols);
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
    return packageNameOf(name);
  }

  auto matchGenericType(const TypeNode *pattern, const Type *actualType,
                        std::string_view parameterName,
                        const Type *&argumentType) -> bool {
    if (llvm::isa<UnitTypeNode>(pattern))
      return isUnitType(actualType);

    if (auto *namedType = dyn_cast<NamedTypeNode>(pattern)) {
      if (namedType->name() == parameterName) {
        if (!argumentType) {
          argumentType = actualType;
          return true;
        }
        return sameType(argumentType, actualType);
      }

      auto *patternType = resolveType(namedType);
      return sameType(patternType, actualType);
    }

    if (auto *tensorPattern = dyn_cast<TensorTypeNode>(pattern)) {
      auto *tensorType = getTensorType(actualType);
      return tensorType &&
             tensorPattern->shape() == tensorType->shape() &&
             matchGenericType(tensorPattern->elementTypeNode(),
                              tensorType->elementType(), parameterName,
                              argumentType);
    }

    if (auto *listPattern = dyn_cast<ListTypeNode>(pattern)) {
      auto *listType = getListType(actualType);
      return listType &&
             matchGenericType(listPattern->elementTypeNode(),
                              listType->elementType(), parameterName,
                              argumentType);
    }

    if (auto *ptrPattern = dyn_cast<PtrTypeNode>(pattern)) {
      auto *ptrType = getPtrType(actualType);
      return ptrType &&
             matchGenericType(ptrPattern->pointeeTypeNode(),
                              ptrType->pointeeType(), parameterName,
                              argumentType);
    }

    if (auto *genericPattern = dyn_cast<GenericTypeNode>(pattern)) {
      auto *alias = lookupComptimeTypeAlias(genericPattern->name());
      if (!alias)
        return false;

      // A generic alias first expands in its definition package, then that
      // body is matched structurally against the actual concrete type. This is
      // the piece that lets a call infer T from `Ptr<List<T>>`.
      auto argumentTypeNode =
          hasTypeParameter(genericPattern->argumentTypeNode(), parameterName)
              ? cloneTypeNode(genericPattern->argumentTypeNode())
              : nullptr;
      if (!argumentTypeNode) {
        auto *argumentType = resolveType(genericPattern->argumentTypeNode());
        if (!argumentType)
          return false;
        argumentTypeNode = typeToTypeNode(
            argumentType, genericPattern->argumentTypeNode()->location());
      }
      auto aliasBody = substituteTypeNode(
          alias->bodyTypeNode, alias->parameterName, argumentTypeNode.get());
      PackageScope packageScope(_currentPackageName, alias->packageName);
      return matchGenericType(aliasBody.get(), actualType, parameterName,
                              argumentType);
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
                              actualFields[i].type(), parameterName,
                              argumentType))
          return false;
      }
      return true;
    }

    auto *patternType = resolveType(pattern);
    return sameType(patternType, actualType);
  }

  auto rejectUnitType(const TypeNode *typeNode, const Type *type)
      -> CherryResult {
    if (!isUnitType(type))
      return success();

    return emitError(typeNode, diag::unexpected_unit_type);
  }

  auto rejectUnitElementType(const TypeNode *typeNode, const Type *type)
      -> CherryResult {
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

  auto resolveType(const StructTypeNode *typeNode, std::string_view name)
      -> const StructType * {
    std::vector<StructField> fields;
    NameSet fieldNames;
    unsigned fieldIndex = 0;
    for (auto &fieldDecl : typeNode->fields()) {
      auto variable = fieldDecl->variable().get();
      auto *fieldType = checkType(fieldDecl->typeNode(), UnitPolicy::Reject);
      if (!fieldType)
        return nullptr;
      fieldDecl->setType(fieldType);
      auto fieldName = variable->name();
      if (!declareName(fieldNames, fieldName)) {
        emitError(variable, diag::redefinition_var);
        return nullptr;
      }
      fields.push_back(StructField{fieldName, fieldType, fieldIndex++});
    }

    return _typeContext.createStructType(name, std::move(fields));
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

    auto *argumentType = resolveType(typeNode->argumentTypeNode());
    if (!argumentType)
      return nullptr;

    // The type argument belongs to the use site, while the alias body belongs
    // to the definition package. Resolve the argument first, then substitute a
    // concrete TypeNode before switching package scope for the alias body.
    auto argumentTypeNode =
        typeToTypeNode(argumentType, typeNode->argumentTypeNode()->location());
    auto instantiatedTypeNode =
        substituteTypeNode(alias->bodyTypeNode, alias->parameterName,
                           argumentTypeNode.get());
    PackageScope packageScope(_currentPackageName, alias->packageName);
    if (auto *structTypeNode =
            dyn_cast<StructTypeNode>(instantiatedTypeNode.get())) {
      auto structName = genericStructName(aliasName, argumentType);
      auto cached = _genericStructTypes.find(structName);
      if (cached != _genericStructTypes.end())
        return cached->second;

      auto *structType = resolveType(structTypeNode, structName);
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
      return cherry::isTensorType(parameterType);
    return false;
  }

  auto setBuiltinType(Expr *expr, BuiltinTypeKind kind) -> void {
    expr->setType(_typeContext.getBuiltinType(kind));
  }

  static auto isFloat32TensorType(const TensorType *type) -> bool {
    return type && isFloat32Type(type->elementType());
  }

  static auto stdlibListElementType(const Type *type) -> const Type * {
    auto *ptrType = cherry::getPtrType(type);
    if (!ptrType)
      return nullptr;

    auto *structType = cherry::getStructType(ptrType->pointeeType());
    if (!structType)
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

    auto *dataPtrType = cherry::getPtrType(fields[2].type());
    if (!dataPtrType)
      return nullptr;
    return dataPtrType->pointeeType();
  }

};

} // end namespace

auto SemaImpl::sema(Decl *node) -> CherryResult {
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

auto SemaImpl::sema(Prototype *node) -> CherryResult {
  if (node->isGeneric())
    return success();

  return semaFunctionSignature(node);
}

auto SemaImpl::semaFunctionParameters(
    Prototype *node, std::vector<const Type *> &parameterTypes)
    -> CherryResult {
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
    -> CherryResult {
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

auto SemaImpl::semaFunctionSignature(Prototype *node) -> CherryResult {
  std::vector<const Type *> parameterTypes;
  if (semaFunctionParameters(node, parameterTypes))
    return failure();

  auto *returnType = resolveType(node->returnTypeNode());
  if (!returnType)
    return failure();
  node->setType(returnType);

  auto name = node->id()->name();
  if (name == builtins::size) {
    auto diagnostic = formatNameDiagnostic(diag::redefinition_func, name);
    return emitError(node->id().get(), diagnostic);
  }
  if (declareFunction(name, std::move(parameterTypes), returnType)) {
    auto diagnostic = formatNameDiagnostic(diag::redefinition_func, name);
    return emitError(node->id().get(), diagnostic);
  }
  return success();
}

auto SemaImpl::sema(FunctionDecl *node) -> CherryResult {
  PackageScope packageScope(_currentPackageName,
                            functionPackageName(node->proto()->id()->name()));
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
  if (!sameType(signature->returnType, expression->type()))
    return emitError(expression, diag::wrong_return_type);

  return success();
}

auto SemaImpl::sema(StructDecl *node) -> CherryResult {
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
  return success();
}

auto SemaImpl::sema(ComptimeTypeAliasDecl *node) -> CherryResult {
  auto packageName = packageNameOf(node->name());
  PackageScope packageScope(_currentPackageName, packageName);
  if (_symbols.lookupType(node->name()) ||
      _symbols.lookupComptimeTypeAlias(node->name()))
    return emitError(node, diag::redefinition_type);

  if (_symbols.declareComptimeTypeAlias(node->name(), packageName,
                                        node->parameterName(),
                                        node->bodyTypeNode()))
    return emitError(node, diag::redefinition_type);
  return success();
}

auto SemaImpl::sema(Expr *node) -> CherryResult {
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
  case Expr::Expr_TypeLayout:
    return sema(cast<TypeLayoutExpr>(node));
  case Expr::Expr_HeapAlloc:
    return sema(cast<HeapAllocExpr>(node));
  case Expr::Expr_Deref:
    return sema(cast<DerefExpr>(node));
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
  case Expr::Expr_For:
    return sema(cast<ForExpr>(node));
  default:
    llvm_unreachable("Unexpected expression");
  }
}

auto SemaImpl::sema(Expr *node, const Type *type) -> CherryResult {
  auto *arrayLiteral = dyn_cast<ArrayLiteralExpr>(node);
  if (arrayLiteral) {
    // Source `[...]` is neutral syntax. Expected type decides whether it is a
    // Tensor literal or a stdlib List alias; other expressions stay bottom-up.
    if (auto *tensorType = cherry::getTensorType(type))
      return sema(arrayLiteral, tensorType);
    if (auto *elementType = stdlibListElementType(type))
      return semaStdlibListLiteral(arrayLiteral, type, elementType);
  }

  auto *call = dyn_cast<CallExpr>(node);
  if (call && call->name() == builtins::zeros) {
    auto *tensorType = cherry::getTensorType(type);
    if (!tensorType)
      return emitError(call, diag::mismatch_type);
    return semaZeros(call, tensorType);
  }
  if (call && call->name() == builtins::readTensor) {
    auto *tensorType = cherry::getTensorType(type);
    if (!tensorType)
      return emitError(call, diag::mismatch_type);
    return semaReadTensor(call, tensorType);
  }

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
    -> CherryResult {
  auto *genericFunction = symbol->decl;
  auto *genericProto = genericFunction->proto().get();
  auto name = genericProto->id()->name();
  PackageScope packageScope(_currentPackageName, packageNameOf(name));
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

  const Type *argumentType = nullptr;
  auto parameterName = genericProto->typeParameterName();
  for (size_t i = 0; i < expressions.size(); ++i) {
    if (!matchGenericType(parameters[i]->typeNode(), expressions[i]->type(),
                          parameterName, argumentType))
      return emitError(expressions[i].get(), diag::mismatch_type);
  }

  if (expectedType &&
      !matchGenericType(genericProto->returnTypeNode(), expectedType,
                        parameterName, argumentType))
    return emitError(node, diag::mismatch_type);

  if (!argumentType)
    return emitError(node, diag::mismatch_type);

  auto concreteName = genericFunctionName(name, argumentType);
  auto cached = _instantiatedFunctionSymbols.find(concreteName);
  if (cached == _instantiatedFunctionSymbols.end()) {
    auto concreteFunction = instantiateFunctionDecl(
        genericFunction, concreteName, parameterName, argumentType);
    _instantiatedFunctionPackages[concreteName] = packageNameOf(name);

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
    if (!sameType(parameterType, arg->type()))
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

auto SemaImpl::sema(UnitExpr *node) -> CherryResult {
  setBuiltinType(node, BuiltinTypeKind::Unit);
  return success();
}

auto SemaImpl::sema(BlockExpr *node) -> CherryResult {
  VariableScope blockScope(_symbols);
  for (auto &expr : *node)
    if (sema(expr.get()))
      return failure();

  return sema(node->expression().get());
}

auto SemaImpl::sema(BlockExpr *node, const Type *returnType)
    -> CherryResult {
  VariableScope blockScope(_symbols);
  for (auto &expr : *node)
    if (sema(expr.get()))
      return failure();

  return sema(node->expression().get(), returnType);
}

auto SemaImpl::sema(CallExpr *node) -> CherryResult {
  node->setName(canonicalizeImportedName(node->name()));
  auto name = node->name();

  if (name == builtins::builtinPrint || name == builtins::print) {
    return semaPrint(node);
  }
  if (name == nn::matmul) {
    return semaMatmul(node);
  }
  if (name == nn::matadd || name == nn::matsub || name == nn::hadamard) {
    return semaTensorBinary(node);
  }
  if (name == nn::matscale) {
    return semaMatscale(node);
  }
  if (name == nn::transpose) {
    return semaTranspose(node);
  }
  if (name == nn::exp || name == nn::sigmoid ||
      name == nn::sigmoidPrime) {
    return semaElementwiseNN(node);
  }
  if (name == nn::argmax) {
    return semaArgmax(node);
  }
  if (name == builtins::size) {
    return semaSize(node);
  }
  if (name == builtins::zeros) {
    return emitError(node, diag::mismatch_type);
  }
  if (name == builtins::builtinOpen || name == builtins::open) {
    return semaFileOpen(node);
  }
  if (name == builtins::read) {
    return semaFileRead(node);
  }
  if (name == builtins::readTensor) {
    return emitError(node, diag::mismatch_type);
  }
  if (name == builtins::write) {
    return semaFileWrite(node);
  }
  if (name == builtins::builtinClose || name == builtins::close) {
    return semaFileClose(node);
  }

  auto *signature = lookupFunction(name);
  if (!signature) {
    if (auto *genericFunction = lookupGenericFunction(name))
      return semaGenericCall(node, genericFunction);

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
    if (sema(arg.get(), parameterType))
      return failure();
    if (!sameType(parameterType, arg->type()))
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

auto SemaImpl::sema(StructLiteralExpr *node) -> CherryResult {
  auto *type = resolveType(node->typeNode());
  auto *structType = cherry::getStructType(type);
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

auto SemaImpl::sema(VariableExpr *node) -> CherryResult {
  auto *symbol = lookupVariable(node->name());
  if (!symbol)
    return emitError(node, diag::undefined_var);
  node->setType(symbol->type);
  return success();
}

auto SemaImpl::sema(MemberExpr *node) -> CherryResult {
  if (sema(node->base().get()))
    return failure();

  auto *baseType = node->base()->type();
  auto *ptrType = cherry::getPtrType(baseType);
  auto *structType = ptrType ? cherry::getStructType(ptrType->pointeeType())
                             : cherry::getStructType(baseType);
  if (!structType)
    return emitError(node->base().get(), diag::mismatch_type);

  auto *field = structType->field(node->fieldName());
  if (!field)
    return emitError(node, diag::undefined_field);
  if (!field->type())
    return emitError(node, diag::mismatch_type);

  node->setType(field->type());
  node->setFieldIndex(field->index());
  node->setLvalue(ptrType || node->base()->isLvalue());
  return success();
}

auto SemaImpl::sema(DecimalLiteralExpr *node) -> CherryResult {
  setBuiltinType(node, BuiltinTypeKind::UInt64);
  return success();
}

auto SemaImpl::sema(FloatLiteralExpr *node) -> CherryResult {
  setBuiltinType(node, BuiltinTypeKind::Float32);
  return success();
}

auto SemaImpl::sema(BoolLiteralExpr *node) -> CherryResult {
  setBuiltinType(node, BuiltinTypeKind::Bool);
  return success();
}

auto SemaImpl::sema(StringLiteralExpr *node) -> CherryResult {
  setBuiltinType(node, BuiltinTypeKind::String);
  return success();
}

auto SemaImpl::sema(TypeLayoutExpr *node) -> CherryResult {
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

auto SemaImpl::sema(HeapAllocExpr *node) -> CherryResult {
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

auto SemaImpl::sema(DerefExpr *node) -> CherryResult {
  if (sema(node->pointer().get()))
    return failure();

  auto *ptrType = cherry::getPtrType(node->pointer()->type());
  if (!ptrType)
    return emitError(node->pointer().get(), diag::mismatch_type);

  node->setType(ptrType->pointeeType());
  return success();
}

auto SemaImpl::sema(AssignExpr *node) -> CherryResult {
  if (sema(node->lhs().get()) ||
      sema(node->rhs().get(), node->lhs()->type()))
    return failure();
  if (!sameType(node->lhs()->type(), node->rhs()->type()))
    return emitError(node->lhs().get(), diag::mismatch_type);
  if (!node->lhs()->isLvalue())
    return emitError(node->lhs().get(), diag::expected_lvalue);
  if (checkAssignable(node->lhs().get()))
    return failure();
  if (cherry::isTensorType(node->lhs()->type()) &&
      checkConstTensorUseAsMutable(node->rhs().get()))
    return failure();
  setBuiltinType(node, BuiltinTypeKind::Unit);
  return success();
}

auto SemaImpl::sema(BinaryExpr *node) -> CherryResult {
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

auto SemaImpl::semaBinaryOperandsSameType(BinaryExpr *node) -> CherryResult {
  if (sema(node->lhs().get()) || sema(node->rhs().get()))
    return failure();
  if (!sameType(node->lhs()->type(), node->rhs()->type()))
    return emitError(node->lhs().get(), diag::mismatch_type);
  return success();
}

auto SemaImpl::checkAssignable(const Expr *expr) -> CherryResult {
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

auto SemaImpl::checkConstTensorUseAsMutable(const Expr *expr) -> CherryResult {
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
    -> CherryResult {
  if (node->isConst() || !cherry::isTensorType(type))
    return success();
  return checkConstTensorUseAsMutable(node->init().get());
}

auto SemaImpl::sema(ArrayLiteralExpr *expr) -> CherryResult {
  auto &elements = expr->getElements();
  if (elements.empty())
    return emitError(expr, diag::expected_expr);

  if (sema(elements.front().get()))
    return failure();

  auto *firstElementType = elements.front()->type();
  auto *elementType = firstElementType;
  std::vector<int64_t> currentShape{static_cast<int64_t>(elements.size())};

  if (auto *nestedTensorType = cherry::getTensorType(elementType)) {
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
    -> CherryResult {
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
                                 elementType, withCapacityFunctionName) ||
      instantiateGenericFunction(expr, "std.collections.push",
                                 elementType, pushFunctionName))
    return failure();

  expr->setStdlibListLiteral(elementType, withCapacityFunctionName,
                             pushFunctionName);
  expr->setType(type);
  return success();
}

auto SemaImpl::sema(ArrayLiteralExpr *expr, const TensorType *type)
    -> CherryResult {
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
    -> CherryResult {
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

auto SemaImpl::semaZeros(CallExpr *node, const TensorType *type)
    -> CherryResult {
  if (!node->expressions().empty())
    return emitError(node, diag::wrong_num_arg);

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

auto SemaImpl::sema(IndexExpr *expr) -> CherryResult {
  if (sema(expr->base().get()))
    return failure();

  auto *ptrType = cherry::getPtrType(expr->base()->type());
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
    return success();
  }

  auto *tensorType = cherry::getTensorType(expr->base()->type());
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
  return success();
}

auto SemaImpl::semaMatmul(CallExpr *node) -> CherryResult {
  auto &expressions = node->expressions();
  if (expressions.size() != 2)
    return emitError(node, diag::wrong_num_arg);

  for (auto &expr : expressions)
    if (sema(expr.get()))
      return failure();

  auto *lhsType = cherry::getTensorType(expressions[0]->type());
  auto *rhsType = cherry::getTensorType(expressions[1]->type());
  if (!lhsType || !rhsType)
    return emitError(node, diag::mismatch_type);
  if (!isFloat32TensorType(lhsType) || !isFloat32TensorType(rhsType))
    return emitError(node, diag::mismatch_type);
  auto &lhsShape = lhsType->shape();
  auto &rhsShape = rhsType->shape();
  if (lhsShape.size() != 2 || rhsShape.size() != 2)
    return emitError(node, diag::mismatch_type);

  auto lhsCols = lhsShape[1];
  auto rhsRows = rhsShape[0];
  if (lhsCols >= 0 && rhsRows >= 0 && lhsCols != rhsRows)
    return emitError(node, diag::mismatch_type);

  auto *resultType =
      _typeContext.createTensorType(
          _typeContext.getBuiltinType(BuiltinTypeKind::Float32),
          std::vector<int64_t>{lhsShape[0], rhsShape[1]});
  node->setType(resultType);

  return success();
}

auto SemaImpl::semaTensorBinary(CallExpr *node) -> CherryResult {
  auto &expressions = node->expressions();
  if (expressions.size() != 2)
    return emitError(node, diag::wrong_num_arg);

  for (auto &expr : expressions)
    if (sema(expr.get()))
      return failure();

  auto *lhsType = cherry::getTensorType(expressions[0]->type());
  auto *rhsType = cherry::getTensorType(expressions[1]->type());
  if (!lhsType || !rhsType)
    return emitError(node, diag::mismatch_type);
  if (!isFloat32TensorType(lhsType) || !isFloat32TensorType(rhsType))
    return emitError(node, diag::mismatch_type);
  auto &lhsShape = lhsType->shape();
  auto &rhsShape = rhsType->shape();
  if (lhsShape.size() != 2 || rhsShape.size() != 2)
    return emitError(node, diag::mismatch_type);

  std::vector<int64_t> resultShape;
  for (size_t i = 0; i < lhsShape.size(); ++i) {
    auto lhsDim = lhsShape[i];
    auto rhsDim = rhsShape[i];
    if (lhsDim >= 0 && rhsDim >= 0 && lhsDim != rhsDim)
      return emitError(node, diag::mismatch_type);
    resultShape.push_back(lhsDim >= 0 ? lhsDim : rhsDim);
  }

  auto *resultType = _typeContext.createTensorType(
      _typeContext.getBuiltinType(BuiltinTypeKind::Float32),
      std::move(resultShape));
  node->setType(resultType);

  return success();
}

auto SemaImpl::semaMatscale(CallExpr *node) -> CherryResult {
  auto &expressions = node->expressions();
  if (expressions.size() != 2)
    return emitError(node, diag::wrong_num_arg);

  for (auto &expr : expressions)
    if (sema(expr.get()))
      return failure();

  auto *inputType = cherry::getTensorType(expressions[0]->type());
  if (!inputType)
    return emitError(node, diag::mismatch_type);
  if (!isFloat32TensorType(inputType) || !isFloat32Type(expressions[1]->type()))
    return emitError(node, diag::mismatch_type);
  if (inputType->shape().size() != 2)
    return emitError(node, diag::mismatch_type);

  auto *resultType = _typeContext.createTensorType(
      _typeContext.getBuiltinType(BuiltinTypeKind::Float32),
      inputType->shape());
  node->setType(resultType);

  return success();
}

auto SemaImpl::semaTranspose(CallExpr *node) -> CherryResult {
  auto &expressions = node->expressions();
  if (expressions.size() != 1)
    return emitError(node, diag::wrong_num_arg);

  if (sema(expressions[0].get()))
    return failure();

  auto *inputType = cherry::getTensorType(expressions[0]->type());
  if (!inputType)
    return emitError(node, diag::mismatch_type);
  if (!isFloat32TensorType(inputType))
    return emitError(node, diag::mismatch_type);
  auto &inputShape = inputType->shape();
  if (inputShape.size() != 2)
    return emitError(node, diag::mismatch_type);

  auto *resultType =
      _typeContext.createTensorType(
          _typeContext.getBuiltinType(BuiltinTypeKind::Float32),
          std::vector<int64_t>{inputShape[1], inputShape[0]});
  node->setType(resultType);

  return success();
}

auto SemaImpl::semaElementwiseNN(CallExpr *node) -> CherryResult {
  auto &expressions = node->expressions();
  if (expressions.size() != 1)
    return emitError(node, diag::wrong_num_arg);

  if (sema(expressions[0].get()))
    return failure();

  auto *inputType = cherry::getTensorType(expressions[0]->type());
  if (!inputType)
    return emitError(node, diag::mismatch_type);
  if (!isFloat32TensorType(inputType))
    return emitError(node, diag::mismatch_type);
  if (inputType->shape().size() != 2)
    return emitError(node, diag::mismatch_type);

  auto *resultType = _typeContext.createTensorType(
      _typeContext.getBuiltinType(BuiltinTypeKind::Float32),
      inputType->shape());
  node->setType(resultType);

  return success();
}

auto SemaImpl::semaArgmax(CallExpr *node) -> CherryResult {
  auto &expressions = node->expressions();
  if (expressions.size() != 1)
    return emitError(node, diag::wrong_num_arg);

  if (sema(expressions[0].get()))
    return failure();

  auto *inputType = cherry::getTensorType(expressions[0]->type());
  if (!inputType)
    return emitError(node, diag::mismatch_type);
  if (!isFloat32TensorType(inputType))
    return emitError(node, diag::mismatch_type);
  auto &inputShape = inputType->shape();
  if (inputShape.size() != 1 && inputShape.size() != 2)
    return emitError(node, diag::mismatch_type);
  // Argmax scans the runtime tensor extent. Rank matters here, but individual
  // dimensions may be dynamic.

  setBuiltinType(node, BuiltinTypeKind::UInt64);

  return success();
}

auto SemaImpl::semaPrint(CallExpr *node) -> CherryResult {
  auto &expressions = node->expressions();
  if (expressions.size() != 1)
    return emitError(node, diag::wrong_num_arg);

  auto *expr = expressions.front().get();
  if (sema(expr))
    return failure();
  if (!isUInt64Type(expr->type()) && !isUInt8Type(expr->type()))
    return emitError(expr, diag::mismatch_type);

  setBuiltinType(node, BuiltinTypeKind::UInt64);
  return success();
}

auto SemaImpl::semaSize(CallExpr *node) -> CherryResult {
  auto &expressions = node->expressions();
  if (expressions.size() != 1)
    return emitError(node, diag::wrong_num_arg);

  if (sema(expressions[0].get()))
    return failure();

  auto *tensorType = cherry::getTensorType(expressions[0]->type());
  if (!tensorType)
    return emitError(expressions[0].get(), diag::mismatch_type);

  auto &shape = tensorType->shape();
  if (shape.empty())
    return emitError(node, diag::mismatch_type);

  setBuiltinType(node, BuiltinTypeKind::UInt64);
  return success();
}

auto SemaImpl::semaFileOpen(CallExpr *node) -> CherryResult {
  auto &expressions = node->expressions();
  if (expressions.size() != 2)
    return emitError(node, diag::wrong_num_arg);

  for (auto &expr : expressions) {
    if (sema(expr.get()))
      return failure();
    if (!isStringType(expr->type()))
      return emitError(expr.get(), diag::mismatch_type);
  }

  setBuiltinType(node, BuiltinTypeKind::File);
  return success();
}

static auto isRawFileTensorType(const Type *type) -> bool {
  auto *tensorType = cherry::getTensorType(type);
  return tensorType && !tensorType->shape().empty() &&
         isNumericType(tensorType->elementType());
}

auto SemaImpl::semaFileRead(CallExpr *node) -> CherryResult {
  auto &expressions = node->expressions();
  if (expressions.size() != 2)
    return emitError(node, diag::wrong_num_arg);

  if (sema(expressions[0].get()))
    return failure();
  if (!isFileType(expressions[0]->type()))
    return emitError(expressions[0].get(), diag::mismatch_type);

  if (sema(expressions[1].get()))
    return failure();
  if (!isRawFileTensorType(expressions[1]->type()))
    return emitError(expressions[1].get(), diag::mismatch_type);
  if (checkConstTensorUseAsMutable(expressions[1].get()))
    return failure();

  setBuiltinType(node, BuiltinTypeKind::UInt64);
  return success();
}

auto SemaImpl::semaReadTensor(CallExpr *node, const TensorType *type)
    -> CherryResult {
  auto &expressions = node->expressions();
  if (expressions.size() != 2)
    return emitError(node, diag::wrong_num_arg);

  if (!isFloat32Type(type->elementType()))
    return emitError(node, diag::mismatch_type);
  if (type->shape().empty())
    return emitError(node, diag::mismatch_type);

  if (sema(expressions[0].get()))
    return failure();
  if (!isFileType(expressions[0]->type()))
    return emitError(expressions[0].get(), diag::mismatch_type);

  if (sema(expressions[1].get()))
    return failure();
  if (!isStringType(expressions[1]->type()))
    return emitError(expressions[1].get(), diag::mismatch_type);

  node->setType(type);
  return success();
}

auto SemaImpl::semaFileWrite(CallExpr *node) -> CherryResult {
  auto &expressions = node->expressions();
  if (expressions.size() != 2)
    return emitError(node, diag::wrong_num_arg);

  if (sema(expressions[0].get()))
    return failure();
  if (!isFileType(expressions[0]->type()))
    return emitError(expressions[0].get(), diag::mismatch_type);

  if (sema(expressions[1].get()))
    return failure();
  if (!isRawFileTensorType(expressions[1]->type()))
    return emitError(expressions[1].get(), diag::mismatch_type);

  setBuiltinType(node, BuiltinTypeKind::UInt64);
  return success();
}

auto SemaImpl::semaFileClose(CallExpr *node) -> CherryResult {
  auto &expressions = node->expressions();
  if (expressions.size() != 1)
    return emitError(node, diag::wrong_num_arg);

  if (sema(expressions[0].get()))
    return failure();
  if (!isFileType(expressions[0]->type()))
    return emitError(expressions[0].get(), diag::mismatch_type);

  setBuiltinType(node, BuiltinTypeKind::UInt64);
  return success();
}

auto SemaImpl::sema(IfExpr *node) -> CherryResult {
  auto conditionExpr = node->conditionExpr().get();
  if (sema(conditionExpr))
    return failure();
  if (!isBoolType(conditionExpr->type()))
    return emitError(conditionExpr, diag::expected_bool);

  auto thenBlock = node->thenBlock().get();
  auto elseBlock = node->elseBlock().get();
  if (sema(thenBlock) || sema(elseBlock))
    return failure();

  auto elseExpr = elseBlock->expression().get();
  if (!sameType(thenBlock->expression()->type(), elseExpr->type()))
    return emitError(elseExpr, diag::mismatch_type_then_else);

  if (auto *elseType = elseExpr->type())
    node->setType(elseType);
  return success();
}

auto SemaImpl::sema(WhileExpr *node) -> CherryResult {
  auto conditionExpr = node->conditionExpr().get();
  if (sema(conditionExpr))
    return failure();
  if (!isBoolType(conditionExpr->type()))
    return emitError(conditionExpr, diag::expected_bool);

  auto bodyBlock = node->bodyBlock().get();
  if (sema(bodyBlock))
    return failure();

  if (!isUnitType(bodyBlock->expression()->type()))
    return emitError(bodyBlock->expression().get(), diag::mismatch_type);

  setBuiltinType(node, BuiltinTypeKind::Unit);
  return success();
}

auto SemaImpl::sema(ForExpr *node) -> CherryResult {
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

auto SemaImpl::sema(Stat *node) -> CherryResult {
  switch (node->getKind()) {
  case Stat::Stat_VariableDecl:
    return sema(cast<VariableStat>(node));
  case Stat::Stat_Expression:
    return sema(cast<ExprStat>(node));
  }
}

auto SemaImpl::sema(VariableStat *node) -> CherryResult {
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

auto SemaImpl::sema(ExprStat *node) -> CherryResult {
  return sema(node->expression().get());
}

namespace cherry {

auto sema(const llvm::SourceMgr &sourceManager, Module &moduleAST)
    -> CherryResult {
  return SemaImpl(sourceManager).sema(moduleAST);
}

auto sema(const llvm::SourceMgr &sourceManager, Module &moduleAST,
          const std::map<std::string, std::string> &importAliases)
    -> CherryResult {
  return SemaImpl(sourceManager, importAliases).sema(moduleAST);
}

} // end namespace cherry
