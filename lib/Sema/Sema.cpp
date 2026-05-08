//===--- Sema.cpp - Cherry Semantic Analysis ------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/Sema/Sema.h"
#include "Symbols.h"
#include "cherry/AST/AST.h"
#include "cherry/Sema/DiagnosticsSema.h"
#include "llvm/ADT/SmallSet.h"

namespace {
using namespace cherry;
using llvm::cast;

auto listContainsUnitType(llvm::StringRef type) -> bool {
  if (auto listType = parseListTypeName(type))
    return listType->elementType == builtins::UnitType ||
           listContainsUnitType(listType->elementType);
  return false;
}

auto typesMatch(llvm::StringRef expected, llvm::StringRef actual) -> bool {
  auto expectedList = parseListTypeName(expected);
  auto actualList = parseListTypeName(actual);
  if (!expectedList || !actualList)
    return expected == actual;

  if (!typesMatch(expectedList->elementType, actualList->elementType))
    return false;
  if (expectedList->shape.size() != actualList->shape.size())
    return false;

  for (auto [expectedDim, actualDim] :
       llvm::zip(expectedList->shape, actualList->shape)) {
    if (expectedDim >= 0 && actualDim >= 0 && expectedDim != actualDim)
      return false;
  }
  return true;
}

class SemaImpl {
public:
  SemaImpl(const llvm::SourceMgr &sourceManager)
      : _sourceManager{sourceManager} {
    _symbols.addBuiltins();
  }

  auto sema(Module &node) -> CherryResult {
    for (auto &decl : node)
      if (sema(decl.get()))
        return failure();

    llvm::ArrayRef<llvm::StringRef> types;
    llvm::StringRef returnType;
    if (_symbols.getFunction("main", types, returnType) || types.size() != 0 ||
        returnType != builtins::UInt64Type)
      return emitError(llvm::SMLoc{}, diag::undefined_main);
    return success();
  }

private:
  const llvm::SourceMgr &_sourceManager;
  Symbols _symbols;

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
  auto semaStructInitializer(CallExpr *node) -> CherryResult;
  auto sema(VariableExpr *node) -> CherryResult;
  auto sema(DecimalLiteralExpr *node) -> CherryResult;
  auto sema(FloatLiteralExpr *node) -> CherryResult;
  auto sema(BoolLiteralExpr *node) -> CherryResult;
  auto sema(BinaryExpr *node) -> CherryResult;
  auto semaRhsLhsSameType(BinaryExpr *node, llvm::StringRef &type)
      -> CherryResult;
  auto semaStructReadOp(BinaryExpr *node) -> CherryResult;
  auto sema(ListLiteralExpr *expr) -> CherryResult;
  auto sema(ListAccessExpr *expr) -> CherryResult;
  auto semaMatmul(CallExpr *node) -> CherryResult;
  auto semaMatadd(CallExpr *node) -> CherryResult;
  auto semaTranspose(CallExpr *node) -> CherryResult;
  auto sema(IfExpr *node) -> CherryResult;
  auto sema(WhileExpr *node) -> CherryResult;

  // Statements
  auto sema(Stat *node) -> CherryResult;
  auto sema(VariableStat *node) -> CherryResult;
  auto sema(ExprStat *node) -> CherryResult;

  // Errors
  auto emitError(Node *node, const llvm::Twine &msg) -> CherryResult {
    _sourceManager.PrintMessage(node->location(),
                                llvm::SourceMgr::DiagKind::DK_Error, msg);
    return failure();
  }

  auto emitError(llvm::SMLoc loc, const llvm::Twine &msg) -> CherryResult {
    _sourceManager.PrintMessage(loc, llvm::SourceMgr::DiagKind::DK_Error, msg);
    return failure();
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
  llvm::SmallVector<llvm::StringRef, 2> types;
  for (auto &par : node->parameters()) {
    auto type = par->varType().get();
    auto typeName = type->name();
    if (typeName == builtins::UnitType || listContainsUnitType(typeName))
      return emitError(type, diag::unexpected_unit_type);
    if (_symbols.checkType(typeName))
      return emitError(type, diag::undefined_type);
    if (_symbols.declareVariable(par->variable().get(), typeName))
      return emitError(par->variable().get(), diag::redefinition_var);
    types.push_back(typeName);
  }

  auto returnType = node->type()->name();
  if (_symbols.checkType(returnType))
    return emitError(node->type().get(), diag::undefined_type);

  auto name = node->id()->name();
  if (_symbols.declareFunction(name, std::move(types), returnType)) {
    const char *diagnostic = diag::redefinition_func;
    char buffer[50];
    snprintf(buffer, 50, diagnostic, name.str().c_str());
    return emitError(node->id().get(), buffer);
  }
  return success();
}

auto SemaImpl::sema(FunctionDecl *node) -> CherryResult {
  _symbols.resetVariables();
  if (sema(node->proto().get()) || sema(node->body().get()))
    return failure();

  auto expression = node->body()->expression().get();
  auto returnType = expression->type();
  if (!typesMatch(node->proto()->type()->name(), returnType))
    return emitError(expression, diag::wrong_return_type);

  return success();
}

auto SemaImpl::sema(StructDecl *node) -> CherryResult {
  llvm::SmallVector<llvm::StringRef, 2> types;
  llvm::SmallSet<llvm::StringRef, 4> variables;
  for (auto &varDecl : *node) {
    auto type = varDecl->varType().get();
    auto var = varDecl->variable().get();
    if (type->name() == builtins::UnitType ||
        listContainsUnitType(type->name()))
      return emitError(type, diag::unexpected_unit_type);
    if (_symbols.checkType(type->name()))
      return emitError(type, diag::undefined_type);
    if (variables.count(var->name()) > 0)
      return emitError(var, diag::redefinition_var);
    variables.insert(var->name());
    types.push_back(type->name());
  }
  auto id = node->id().get();
  if (_symbols.declareType(node))
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
  case Expr::Expr_Variable:
    return sema(cast<VariableExpr>(node));
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
  default:
    llvm_unreachable("Unexpected expression");
  }
}

auto SemaImpl::sema(UnitExpr *node) -> CherryResult {
  node->setType(builtins::UnitType);
  return success();
}

auto SemaImpl::sema(BlockExpr *node) -> CherryResult {
  for (auto &expr : *node)
    if (sema(expr.get()))
      return failure();

  return sema(node->expression().get());
}

auto SemaImpl::sema(CallExpr *node) -> CherryResult {
  auto name = node->name();

  if (llvm::succeeded(_symbols.checkType(name)))
    return semaStructInitializer(node);

  if (name == nn::matmul) {
    return semaMatmul(node);
  }
  if (name == nn::matadd) {
    return semaMatadd(node);
  }
  if (name == nn::transpose) {
    return semaTranspose(node);
  }

  llvm::ArrayRef<llvm::StringRef> parametersTypes;
  llvm::StringRef returnType;
  if (_symbols.getFunction(name, parametersTypes, returnType)) {
    const char *diagnostic = diag::undefined_func;
    char buffer[50];
    snprintf(buffer, 50, diagnostic, name.str().c_str());
    return emitError(node, buffer);
  }

  auto &expressions = node->expressions();
  if (expressions.size() != parametersTypes.size()) {
    const char *diagnostic = diag::func_param;
    char buffer[50];
    snprintf(buffer, 50, diagnostic, name.str().c_str(),
             parametersTypes.size());
    return emitError(node, buffer);
  }

  for (const auto &expr_type : llvm::zip(expressions, parametersTypes)) {
    auto &expr = std::get<0>(expr_type);
    auto type = std::get<1>(expr_type);
    if (sema(expr.get()))
      return failure();
    if (!typesMatch(type, expr->type()))
      return emitError(expr.get(), diag::mismatch_type);
  }

  node->setType(returnType);
  return success();
}

auto SemaImpl::semaStructInitializer(CallExpr *node) -> CherryResult {
  auto type = node->name();
  const VectorUniquePtr<VariableStat> *fieldsTypes;
  if (_symbols.getType(type, fieldsTypes))
    return emitError(node, diag::undefined_type);

  if (node->expressions().size() != fieldsTypes->size())
    return emitError(node, diag::wrong_num_arg);

  for (const auto &expr_type : llvm::zip(*node, *fieldsTypes)) {
    auto &expr = std::get<0>(expr_type);
    auto fieldType = std::get<1>(expr_type)->varType()->name();
    if (sema(expr.get()))
      return failure();
    if (!typesMatch(fieldType, expr->type()))
      return emitError(expr.get(), diag::mismatch_type);
  }

  node->setType(type);
  return success();
}

auto SemaImpl::sema(VariableExpr *node) -> CherryResult {
  llvm::StringRef type;
  if (_symbols.getVariableType(node, type))
    return emitError(node, diag::undefined_var);
  node->setType(type);
  return success();
}

auto SemaImpl::sema(DecimalLiteralExpr *node) -> CherryResult {
  node->setType(builtins::UInt64Type);
  return success();
}

auto SemaImpl::sema(FloatLiteralExpr *node) -> CherryResult {
  node->setType(builtins::Float32Type);
  return success();
}

auto SemaImpl::sema(BoolLiteralExpr *node) -> CherryResult {
  node->setType(builtins::BoolType);
  return success();
}

auto SemaImpl::sema(BinaryExpr *node) -> CherryResult {
  using Operator = BinaryExpr::Operator;
  switch (node->opEnum()) {
  case Operator::Assign: {
    llvm::StringRef type;
    if (semaRhsLhsSameType(node, type))
      return llvm::failure();
    if (!node->lhs()->isLvalue())
      return emitError(node->lhs().get(), diag::expected_lvalue);
    node->setType(builtins::UnitType);
    return success();
  }
  case Operator::StructRead:
    return semaStructReadOp(node);
  default:
    break;
  }

  llvm::StringRef type;
  if (semaRhsLhsSameType(node, type))
    return llvm::failure();

  switch (node->opEnum()) {
  case Operator::Add:
  case Operator::Mul:
  case Operator::Diff:
  case Operator::Div: {
    if (type != builtins::UInt64Type && type != builtins::Float32Type)
      return emitError(node->lhs().get(), diag::mismatch_type);
    node->setType(type);
    return success();
  }
  case Operator::Rem: {
    if (type != builtins::UInt64Type)
      return emitError(node->lhs().get(), diag::mismatch_type);
    node->setType(builtins::UInt64Type);
    return success();
  }
  case Operator::And:
  case Operator::Or: {
    if (type != builtins::BoolType)
      return emitError(node->lhs().get(), diag::mismatch_type);
    node->setType(builtins::BoolType);
    return success();
  }
  case Operator::EQ:
  case Operator::NEQ: {
    if (type != builtins::UInt64Type && type != builtins::BoolType &&
        type != builtins::Float32Type)
      return emitError(node->lhs().get(), diag::mismatch_type);
    node->setType(builtins::BoolType);
    return success();
  }
  case Operator::LT:
  case Operator::LE:
  case Operator::GT:
  case Operator::GE: {
    if (type != builtins::UInt64Type && type != builtins::Float32Type)
      return emitError(node->lhs().get(), diag::mismatch_type);
    node->setType(builtins::BoolType);
    return success();
  }
  default:
    llvm_unreachable("Unexpected BinaryExpr operator");
  }
}

auto SemaImpl::semaRhsLhsSameType(BinaryExpr *node, llvm::StringRef &type)
    -> CherryResult {
  if (sema(node->lhs().get()) || sema(node->rhs().get()))
    return failure();
  auto lhsType = node->lhs()->type();
  auto rhsType = node->rhs()->type();
  if (!typesMatch(lhsType, rhsType))
    return emitError(node->lhs().get(), diag::mismatch_type);
  type = lhsType;
  return success();
}

auto SemaImpl::semaStructReadOp(BinaryExpr *node) -> CherryResult {
  if (sema(node->lhs().get()))
    return failure();

  if (!node->lhs()->isLvalue())
    return emitError(node->lhs().get(), diag::expected_lvalue);

  VariableExpr *var = llvm::dyn_cast<VariableExpr>(node->rhs().get());
  if (!var)
    return emitError(node->rhs().get(), diag::expected_field);

  auto fieldName = var->name();
  const VectorUniquePtr<VariableStat> *fieldsTypes;
  auto lhsType = node->lhs()->type();
  _symbols.getType(lhsType, fieldsTypes);

  auto index = 0;
  for (auto &f : *fieldsTypes) {
    if (f->variable()->name() == fieldName) {
      node->setType(f->varType()->name());
      node->setIndex(index);
      return success();
    }
    index++;
  }

  return emitError(node->rhs().get(), diag::undefined_field);
}

auto SemaImpl::sema(ListLiteralExpr *expr) -> CherryResult {
  auto &elements = expr->getElements();
  if (elements.empty())
    return emitError(expr, diag::expected_expr);

  if (sema(elements.front().get()))
    return failure();

  auto firstType = elements.front()->type();
  std::string elementType = firstType.str();
  std::vector<int64_t> currentShape{static_cast<int64_t>(elements.size())};

  if (auto nestedList = parseListTypeName(firstType)) {
    elementType = nestedList->elementType;
    currentShape.insert(currentShape.end(), nestedList->shape.begin(),
                        nestedList->shape.end());
  }

  for (auto &element : llvm::drop_begin(elements)) {
    if (sema(element.get()))
      return failure();
    if (!typesMatch(firstType, element->type()))
      return emitError(element.get(), diag::mismatch_type);
  }

  expr->setInferredShape(currentShape);
  expr->setType(formatListTypeName(elementType, currentShape));
  return success();
}

auto SemaImpl::sema(ListAccessExpr *expr) -> CherryResult {
  llvm::StringRef listTypeName;
  if (_symbols.getVariableType(expr->getVarName(), listTypeName))
    return emitError(expr, diag::undefined_var);

  auto listType = parseListTypeName(listTypeName);
  if (!listType)
    return emitError(expr, diag::mismatch_type);

  if (expr->getIndices().size() != listType->shape.size())
    return emitError(expr, diag::mismatch_type);

  for (auto &index : expr->getIndices()) {
    if (sema(index.get()))
      return failure();
    if (index->type() != builtins::UInt64Type)
      return emitError(index.get(), diag::mismatch_type);
  }

  expr->setType(listType->elementType);
  return success();
}

auto SemaImpl::semaMatmul(CallExpr *node) -> CherryResult {
  auto &expressions = node->expressions();
  if (expressions.size() != 2)
    return emitError(node, diag::wrong_num_arg);

  for (auto &expr : expressions)
    if (sema(expr.get()))
      return failure();

  auto lhsType = parseListTypeName(expressions[0]->type());
  auto rhsType = parseListTypeName(expressions[1]->type());
  if (!lhsType || !rhsType)
    return emitError(node, diag::mismatch_type);
  if (lhsType->elementType != builtins::Float32Type ||
      rhsType->elementType != builtins::Float32Type)
    return emitError(node, diag::mismatch_type);
  if (lhsType->shape.size() != 2 || rhsType->shape.size() != 2)
    return emitError(node, diag::mismatch_type);

  auto lhsCols = lhsType->shape[1];
  auto rhsRows = rhsType->shape[0];
  if (lhsCols >= 0 && rhsRows >= 0 && lhsCols != rhsRows)
    return emitError(node, diag::mismatch_type);

  node->setType(formatListTypeName(
      builtins::Float32Type,
      llvm::ArrayRef<int64_t>{lhsType->shape[0], rhsType->shape[1]}));

  return success();
}

auto SemaImpl::semaMatadd(CallExpr *node) -> CherryResult {
  auto &expressions = node->expressions();
  if (expressions.size() != 2)
    return emitError(node, diag::wrong_num_arg);

  for (auto &expr : expressions)
    if (sema(expr.get()))
      return failure();

  auto lhsType = parseListTypeName(expressions[0]->type());
  auto rhsType = parseListTypeName(expressions[1]->type());
  if (!lhsType || !rhsType)
    return emitError(node, diag::mismatch_type);
  if (lhsType->elementType != builtins::Float32Type ||
      rhsType->elementType != builtins::Float32Type)
    return emitError(node, diag::mismatch_type);
  if (lhsType->shape.size() != 2 || rhsType->shape.size() != 2)
    return emitError(node, diag::mismatch_type);

  llvm::SmallVector<int64_t, 2> resultShape;
  resultShape.reserve(lhsType->shape.size());
  for (auto [lhsDim, rhsDim] : llvm::zip(lhsType->shape, rhsType->shape)) {
    if (lhsDim >= 0 && rhsDim >= 0 && lhsDim != rhsDim)
      return emitError(node, diag::mismatch_type);
    resultShape.push_back(lhsDim >= 0 ? lhsDim : rhsDim);
  }

  node->setType(formatListTypeName(builtins::Float32Type, resultShape));

  return success();
}

auto SemaImpl::semaTranspose(CallExpr *node) -> CherryResult {
  auto &expressions = node->expressions();
  if (expressions.size() != 1)
    return emitError(node, diag::wrong_num_arg);

  if (sema(expressions[0].get()))
    return failure();

  auto inputType = parseListTypeName(expressions[0]->type());
  if (!inputType)
    return emitError(node, diag::mismatch_type);
  if (inputType->elementType != builtins::Float32Type)
    return emitError(node, diag::mismatch_type);
  if (inputType->shape.size() != 2)
    return emitError(node, diag::mismatch_type);

  node->setType(formatListTypeName(
      builtins::Float32Type,
      llvm::ArrayRef<int64_t>{inputType->shape[1], inputType->shape[0]}));

  return success();
}

auto SemaImpl::sema(IfExpr *node) -> CherryResult {
  auto conditionExpr = node->conditionExpr().get();
  if (sema(conditionExpr))
    return failure();
  if (conditionExpr->type() != builtins::BoolType)
    return emitError(conditionExpr, diag::expected_bool);

  auto thenBlock = node->thenBlock().get();
  auto elseBlock = node->elseBlock().get();
  if (sema(thenBlock) || sema(elseBlock))
    return failure();

  auto elseExpr = elseBlock->expression().get();
  auto elseType = elseExpr->type();
  auto thenType = thenBlock->expression()->type();
  if (!typesMatch(thenType, elseType))
    return emitError(elseExpr, diag::mismatch_type_then_else);

  node->setType(elseType);
  return success();
}

auto SemaImpl::sema(WhileExpr *node) -> CherryResult {
  auto conditionExpr = node->conditionExpr().get();
  if (sema(conditionExpr))
    return failure();
  if (conditionExpr->type() != builtins::BoolType)
    return emitError(conditionExpr, diag::expected_bool);

  auto bodyBlock = node->bodyBlock().get();
  if (sema(bodyBlock))
    return failure();

  auto bodyType = bodyBlock->expression()->type();
  if (bodyType != builtins::UnitType)
    return emitError(bodyBlock->expression().get(), diag::mismatch_type);

  node->setType(builtins::UnitType);
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
  auto varType = node->varType().get();
  auto varTypeName = varType->name();
  if (listContainsUnitType(varTypeName))
    return emitError(varType, diag::unexpected_unit_type);
  if (_symbols.checkType(varTypeName))
    return emitError(varType, diag::undefined_type);
  if (_symbols.declareVariable(var, varTypeName))
    return emitError(var, diag::redefinition_var);

  auto initExpr = node->init().get();
  if (sema(initExpr))
    return failure();
  if (!typesMatch(varTypeName, initExpr->type()))
    return emitError(initExpr, diag::mismatch_type);
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
