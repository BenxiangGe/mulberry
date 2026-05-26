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
#include <set>
#include <string>
#include <string_view>

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

class SemaImpl {
public:
  SemaImpl(const llvm::SourceMgr &sourceManager)
      : _sourceManager{sourceManager} {
    addBuiltins();
  }

  auto sema(Module &node) -> CherryResult {
    for (auto &decl : node)
      if (sema(decl.get()))
        return failure();

    auto *mainSignature = lookupFunction("main");
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

  enum class UnitPolicy {
    Allow,
    Reject,
  };

  // Semantic Analysis

  // Declarations
  auto sema(Decl *node) -> CherryResult;
  auto sema(Prototype *node) -> CherryResult;
  auto sema(FunctionDecl *node) -> CherryResult;
  auto sema(StructDecl *node) -> CherryResult;

  // Expressions
  auto sema(Expr *node) -> CherryResult;
  auto sema(UnitExpr *node) -> CherryResult;
  auto sema(BlockExpr *node) -> CherryResult;
  auto sema(CallExpr *node) -> CherryResult;
  auto sema(StructLiteralExpr *node) -> CherryResult;
  auto sema(VariableExpr *node) -> CherryResult;
  auto sema(MemberExpr *node) -> CherryResult;
  auto sema(AssignExpr *node) -> CherryResult;
  auto sema(DecimalLiteralExpr *node) -> CherryResult;
  auto sema(FloatLiteralExpr *node) -> CherryResult;
  auto sema(BoolLiteralExpr *node) -> CherryResult;
  auto sema(BinaryExpr *node) -> CherryResult;
  auto semaBinaryOperandsSameType(BinaryExpr *node) -> CherryResult;
  auto checkAssignable(const Expr *expr) -> CherryResult;
  auto checkConstListUseAsMutable(const Expr *expr) -> CherryResult;
  auto checkConstListBinding(const VariableStat *node,
                             const Type *type) -> CherryResult;
  auto sema(ListLiteralExpr *expr) -> CherryResult;
  auto sema(ListAccessExpr *expr) -> CherryResult;
  auto semaMatmul(CallExpr *node) -> CherryResult;
  auto semaMatadd(CallExpr *node) -> CherryResult;
  auto semaTranspose(CallExpr *node) -> CherryResult;
  auto semaElementwiseNN(CallExpr *node) -> CherryResult;
  auto semaArgmax(CallExpr *node) -> CherryResult;
  auto semaSize(CallExpr *node) -> CherryResult;
  auto sema(IfExpr *node) -> CherryResult;
  auto sema(WhileExpr *node) -> CherryResult;
  auto sema(ForExpr *node) -> CherryResult;

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
    auto *uint64Type = declareBuiltinType(BuiltinTypeKind::UInt64);
    declareBuiltinType(BuiltinTypeKind::Float32);

    declareFunction(builtins::print, std::vector<const Type *>{uint64Type},
                    uint64Type);
    declareFunction(builtins::boolToUInt64,
                    std::vector<const Type *>{boolType}, uint64Type);
  }

  auto lookupFunction(std::string_view name) -> const FunctionSymbol * {
    return _symbols.lookupFunction(name);
  }

  auto lookupType(std::string_view name) -> const Type * {
    return _symbols.lookupType(name);
  }

  auto lookupStructType(std::string_view name) -> const StructType * {
    return cherry::getStructType(lookupType(name));
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

  auto declareFunction(std::string_view name,
                       std::vector<const Type *> parameterTypes,
                       const Type *returnType)
      -> CherryResult {
    return _symbols.declareFunction(name, std::move(parameterTypes),
                                    returnType);
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

  auto resolveType(const ListTypeNode *typeNode) -> const Type * {
    auto *elementType = resolveType(typeNode->elementTypeNode());
    if (!elementType)
      return nullptr;

    return _typeContext.createListType(elementType, typeNode->shape());
  }

  auto resolveType(const TypeNode *typeNode) -> const Type * {
    if (auto *unitType = dyn_cast<UnitTypeNode>(typeNode))
      return resolveType(unitType);

    if (auto *listType = dyn_cast<ListTypeNode>(typeNode))
      return resolveType(listType);

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

  auto isListParameter(const FunctionSymbol *signature, size_t index) -> bool {
    auto *parameterType = signature->parameterTypes[index];
    if (parameterType)
      return cherry::isListType(parameterType);
    return false;
  }

  auto setBuiltinType(Expr *expr, BuiltinTypeKind kind) -> void {
    expr->setType(_typeContext.getBuiltinType(kind));
  }

  static auto isFloat32ListType(const ListType *type) -> bool {
    return type && isFloat32Type(type->elementType());
  }

};

} // end namespace

auto SemaImpl::sema(Decl *node) -> CherryResult {
  switch (node->getKind()) {
  case Decl::Decl_Function:
    return sema(cast<FunctionDecl>(node));
  case Decl::Decl_Struct:
    return sema(cast<StructDecl>(node));
  }
}

auto SemaImpl::sema(Prototype *node) -> CherryResult {
  std::vector<const Type *> parameterTypes;
  for (auto &par : node->parameters()) {
    auto *parameterType = checkType(par->typeNode(), UnitPolicy::Reject);
    if (!parameterType)
      return failure();
    par->setType(parameterType);
    if (declareVariable(par->variable()->name(), parameterType))
      return emitError(par->variable().get(), diag::redefinition_var);
    parameterTypes.push_back(parameterType);
  }

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
  _symbols.resetVariables();
  if (sema(node->proto().get()) || sema(node->body().get()))
    return failure();

  auto expression = node->body()->expression().get();
  auto *signature = lookupFunction(node->proto()->id()->name());
  if (!signature)
    return failure();
  if (!sameType(signature->returnType, expression->type()))
    return emitError(expression, diag::wrong_return_type);

  return success();
}

auto SemaImpl::sema(StructDecl *node) -> CherryResult {
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
  case Expr::Expr_ListLiteral:
    return sema(cast<ListLiteralExpr>(node));
  case Expr::Expr_ListAccess:
    return sema(cast<ListAccessExpr>(node));
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

auto SemaImpl::sema(CallExpr *node) -> CherryResult {
  auto name = node->name();

  if (name == nn::matmul) {
    return semaMatmul(node);
  }
  if (name == nn::matadd) {
    return semaMatadd(node);
  }
  if (name == nn::transpose) {
    return semaTranspose(node);
  }
  if (name == nn::exp || name == nn::sigmoid) {
    return semaElementwiseNN(node);
  }
  if (name == nn::argmax) {
    return semaArgmax(node);
  }
  if (name == builtins::size) {
    return semaSize(node);
  }

  auto *signature = lookupFunction(name);
  if (!signature) {
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
    if (sema(arg.get()))
      return failure();
    if (!sameType(signature->parameterTypes[i], arg->type()))
      return emitError(arg.get(), diag::mismatch_type);
    if (isListParameter(signature, i) &&
        checkConstListUseAsMutable(arg.get()))
      return failure();
  }

  if (signature->returnType)
    node->setType(signature->returnType);
  return success();
}

auto SemaImpl::sema(StructLiteralExpr *node) -> CherryResult {
  auto *structType = lookupStructType(node->name());
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
    if (sema(expr.get()))
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

  if (!node->base()->isLvalue())
    return emitError(node->base().get(), diag::expected_lvalue);

  auto *structType = cherry::getStructType(node->base()->type());
  if (!structType)
    return emitError(node->base().get(), diag::mismatch_type);

  auto *field = structType->field(node->fieldName());
  if (!field)
    return emitError(node, diag::undefined_field);
  if (!field->type())
    return emitError(node, diag::mismatch_type);

  node->setType(field->type());
  node->setFieldIndex(field->index());
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

auto SemaImpl::sema(AssignExpr *node) -> CherryResult {
  if (sema(node->lhs().get()) || sema(node->rhs().get()))
    return failure();
  if (!sameType(node->lhs()->type(), node->rhs()->type()))
    return emitError(node->lhs().get(), diag::mismatch_type);
  if (!node->lhs()->isLvalue())
    return emitError(node->lhs().get(), diag::expected_lvalue);
  if (checkAssignable(node->lhs().get()))
    return failure();
  if (cherry::isListType(node->lhs()->type()) &&
      checkConstListUseAsMutable(node->rhs().get()))
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

  if (auto *listAccess = llvm::dyn_cast<ListAccessExpr>(expr)) {
    auto *symbol = lookupVariable(listAccess->getVarName());
    if (!symbol)
      return emitError(listAccess->location(), diag::undefined_var);
    if (symbol->isConst)
      return emitError(listAccess->location(), diag::assign_const);
    return success();
  }

  if (auto *memberAccess = llvm::dyn_cast<MemberExpr>(expr))
    return checkAssignable(memberAccess->base().get());

  return success();
}

auto SemaImpl::checkConstListUseAsMutable(const Expr *expr) -> CherryResult {
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

auto SemaImpl::checkConstListBinding(const VariableStat *node,
                                     const Type *type)
    -> CherryResult {
  if (node->isConst() || !cherry::isListType(type))
    return success();
  return checkConstListUseAsMutable(node->init().get());
}

auto SemaImpl::sema(ListLiteralExpr *expr) -> CherryResult {
  auto &elements = expr->getElements();
  if (elements.empty())
    return emitError(expr, diag::expected_expr);

  if (sema(elements.front().get()))
    return failure();

  auto *firstElementType = elements.front()->type();
  auto *elementType = firstElementType;
  std::vector<int64_t> currentShape{static_cast<int64_t>(elements.size())};

  if (auto *nestedListType = cherry::getListType(elementType)) {
    elementType = nestedListType->elementType();
    currentShape.insert(currentShape.end(), nestedListType->shape().begin(),
                        nestedListType->shape().end());
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
  auto *listType =
      _typeContext.createListType(elementType, std::move(currentShape));
  expr->setType(listType);
  return success();
}

auto SemaImpl::sema(ListAccessExpr *expr) -> CherryResult {
  auto *symbol = lookupVariable(expr->getVarName());
  if (!symbol)
    return emitError(expr, diag::undefined_var);
  auto *listType = cherry::getListType(symbol->type);
  if (!listType)
    return emitError(expr, diag::mismatch_type);

  auto listRank = listType->shape().size();
  if (expr->getIndices().size() != listRank)
    return emitError(expr, diag::mismatch_type);

  for (auto &index : expr->getIndices()) {
    if (sema(index.get()))
      return failure();
    if (!isUInt64Type(index->type()))
      return emitError(index.get(), diag::mismatch_type);
  }

  expr->setType(listType->elementType());
  return success();
}

auto SemaImpl::semaMatmul(CallExpr *node) -> CherryResult {
  auto &expressions = node->expressions();
  if (expressions.size() != 2)
    return emitError(node, diag::wrong_num_arg);

  for (auto &expr : expressions)
    if (sema(expr.get()))
      return failure();

  auto *lhsType = cherry::getListType(expressions[0]->type());
  auto *rhsType = cherry::getListType(expressions[1]->type());
  if (!lhsType || !rhsType)
    return emitError(node, diag::mismatch_type);
  if (!isFloat32ListType(lhsType) || !isFloat32ListType(rhsType))
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
      _typeContext.createListType(
          _typeContext.getBuiltinType(BuiltinTypeKind::Float32),
          std::vector<int64_t>{lhsShape[0], rhsShape[1]});
  node->setType(resultType);

  return success();
}

auto SemaImpl::semaMatadd(CallExpr *node) -> CherryResult {
  auto &expressions = node->expressions();
  if (expressions.size() != 2)
    return emitError(node, diag::wrong_num_arg);

  for (auto &expr : expressions)
    if (sema(expr.get()))
      return failure();

  auto *lhsType = cherry::getListType(expressions[0]->type());
  auto *rhsType = cherry::getListType(expressions[1]->type());
  if (!lhsType || !rhsType)
    return emitError(node, diag::mismatch_type);
  if (!isFloat32ListType(lhsType) || !isFloat32ListType(rhsType))
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

  auto *resultType = _typeContext.createListType(
      _typeContext.getBuiltinType(BuiltinTypeKind::Float32),
      std::move(resultShape));
  node->setType(resultType);

  return success();
}

auto SemaImpl::semaTranspose(CallExpr *node) -> CherryResult {
  auto &expressions = node->expressions();
  if (expressions.size() != 1)
    return emitError(node, diag::wrong_num_arg);

  if (sema(expressions[0].get()))
    return failure();

  auto *inputType = cherry::getListType(expressions[0]->type());
  if (!inputType)
    return emitError(node, diag::mismatch_type);
  if (!isFloat32ListType(inputType))
    return emitError(node, diag::mismatch_type);
  auto &inputShape = inputType->shape();
  if (inputShape.size() != 2)
    return emitError(node, diag::mismatch_type);

  auto *resultType =
      _typeContext.createListType(
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

  auto *inputType = cherry::getListType(expressions[0]->type());
  if (!inputType)
    return emitError(node, diag::mismatch_type);
  if (!isFloat32ListType(inputType))
    return emitError(node, diag::mismatch_type);
  if (inputType->shape().size() != 2)
    return emitError(node, diag::mismatch_type);

  auto *resultType = _typeContext.createListType(
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

  auto *inputType = cherry::getListType(expressions[0]->type());
  if (!inputType)
    return emitError(node, diag::mismatch_type);
  if (!isFloat32ListType(inputType))
    return emitError(node, diag::mismatch_type);
  auto &inputShape = inputType->shape();
  if (inputShape.size() != 1 && inputShape.size() != 2)
    return emitError(node, diag::mismatch_type);
  for (auto dim : inputShape)
    if (dim < 0)
      return emitError(node, diag::mismatch_type);

  setBuiltinType(node, BuiltinTypeKind::UInt64);

  return success();
}

auto SemaImpl::semaSize(CallExpr *node) -> CherryResult {
  auto &expressions = node->expressions();
  if (expressions.size() != 1)
    return emitError(node, diag::wrong_num_arg);

  if (sema(expressions[0].get()))
    return failure();

  auto *listType = cherry::getListType(expressions[0]->type());
  if (!listType)
    return emitError(expressions[0].get(), diag::mismatch_type);

  auto &shape = listType->shape();
  if (shape.empty() || shape.front() < 0)
    return emitError(node, diag::mismatch_type);

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
  if (sema(initExpr))
    return failure();
  if (!sameType(varType, initExpr->type()))
    return emitError(initExpr, diag::mismatch_type);
  if (checkConstListBinding(node, varType))
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

} // end namespace cherry
