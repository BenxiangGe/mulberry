//===--- MLIRGen.cpp - MLIR Generator -------------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/MLIRGen.h"
#include "cherry/AST/AST.h"
#include "cherry/Basic/Logging.h"
#include "cherry/Basic/Builtins.h"
#include "cherry/Basic/CherryResult.h"
#include "cherry/MLIRGen/IR/CherryOps.h"
#include "cherry/MLIRGen/IR/CherryTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/TypeSwitch.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

#include <map>

namespace {
using namespace mlir::cherry;
using namespace mlir::arith;
using namespace cherry;
using llvm::cast;
using llvm::failure;
using llvm::success;

#undef DEBUG_TYPE
#define DEBUG_TYPE "MLIRGen"

class MLIRGenImpl {
public:
  MLIRGenImpl(const llvm::SourceMgr &sourceManager, mlir::MLIRContext &context)
      : _sourceManager{sourceManager}, _builder(&context),
        _fileNameIdentifier{
            _sourceManager.getMemoryBuffer(_sourceManager.getMainFileID())
                ->getBufferIdentifier()} {}

  auto gen(const Module &node) -> CherryResult;

  mlir::ModuleOp module;

private:
  const llvm::SourceMgr &_sourceManager;
  mlir::OpBuilder _builder;
  std::map<llvm::StringRef, mlir::Value> _variableSymbols;
  std::map<llvm::StringRef, mlir::Type> _typeSymbols;
  std::map<llvm::StringRef, mlir::Type> _functionSymbols;
  std::map<llvm::StringRef, mlir::func::FuncOp> _functionOps;
  std::map<llvm::StringRef, const StructDecl *> _structNodes;
  llvm::StringRef _fileNameIdentifier;

  // Declarations
  auto gen(const Decl *node) -> mlir::Operation *;
  auto gen(const Prototype *node) -> mlir::func::FuncOp;
  auto gen(const FunctionDecl *node) -> mlir::func::FuncOp;
  auto gen(const StructDecl *node) -> void;

  // Expressions
  auto gen(const Expr *node) -> mlir::Value;
  auto gen(const UnitExpr *node) -> mlir::Value;
  auto gen(const BlockExpr *node) -> mlir::Value;
  auto gen(const IfExpr *node) -> mlir::Value;
  auto gen(const WhileExpr *node) -> mlir::Value;
  auto genPrint(const CallExpr *node) -> mlir::Value;
  auto gen(const CallExpr *node) -> mlir::Value;
  auto gen(const VariableExpr *node) -> mlir::Value;
  auto gen(const DecimalLiteralExpr *node) -> mlir::Value;
  auto gen(const BoolLiteralExpr *node) -> mlir::Value;
  auto gen(const BinaryExpr *node) -> mlir::Value;
  auto genAssignOp(const BinaryExpr *node) -> mlir::Value;

  // Statements
  auto gen(const Stat *node) -> void;
  auto gen(const VariableStat *node) -> void;
  auto gen(const ExprStat *node) -> void;

  auto getAlignOne() -> mlir::IntegerAttr;
  auto genStructLiteral(const CallExpr *callExpr, llvm::StringRef typeName,
                        mlir::Value targetPtr) -> mlir::Value;

  // Utility
  auto loc(const Node *node) -> mlir::Location {
    auto [line, col] = _sourceManager.getLineAndColumn(node->location());
    return mlir::FileLineColLoc::get(
        _builder.getStringAttr(_fileNameIdentifier), line, col);
  }

  auto getType(llvm::StringRef name) -> mlir::Type {
    if (name == builtins::UnitType) {
      return _builder.getNoneType();
    } else if (name == builtins::UInt64Type) {
      return cir::IntType::get(_builder.getContext(), 64, /*isSigned=*/false);
    } else if (name == builtins::BoolType) {
      return cir::BoolType::get(_builder.getContext());
    } else {
      return _typeSymbols[name];
    }
  }

  auto createEntryBlockAlloca(mlir::Type mlirType, mlir::Location loc)
      -> mlir::Value {
    if (mlirType == getType(builtins::UnitType))
      return nullptr;

    mlir::Type ptrTy = cir::PointerType::get(mlirType);
    cir::AllocaOp alloca = cir::AllocaOp::create(_builder, loc, ptrTy, mlirType,
                                                 "", getAlignOne());

    auto *parentBlock = alloca.getOperation()->getBlock();
    alloca.getOperation()->moveBefore(&parentBlock->front());

    return alloca;
  }
};

} // end namespace

auto MLIRGenImpl::gen(const Module &node) -> CherryResult {
  module = mlir::ModuleOp::create(_builder.getUnknownLoc());

  for (auto &decl : node) {
    if (auto *op = gen(decl.get()))
      module.push_back(op);
  }

  if (failed(mlir::verify(module))) {
    module.emitError("module verification error");
    return failure();
  }

  return success();
}

auto MLIRGenImpl::gen(const Decl *node) -> mlir::Operation * {
  switch (node->getKind()) {
  case Decl::Decl_Function: {
    return gen(cast<FunctionDecl>(node));
  }
  case Decl::Decl_Struct: {
    gen(cast<StructDecl>(node));
    return nullptr;
  }
  }
}

auto MLIRGenImpl::gen(const Prototype *node) -> mlir::func::FuncOp {
  llvm::SmallVector<mlir::Type, 3> arg_types;
  arg_types.reserve(node->parameters().size());
  for (auto &param : node->parameters())
    arg_types.push_back(getType(param->varType()->name()));

  llvm::SmallVector<mlir::Type, 1> result_types;
  auto type = getType(node->type()->name());
  if (type != _builder.getNoneType())
    result_types.push_back(type);

  auto funcName = node->id()->name();
  auto funcType = _builder.getFunctionType(arg_types, result_types);
  auto func = mlir::func::FuncOp::create(loc(node), funcName, funcType);

  auto &entryBlock = *func.addEntryBlock();
  _builder.setInsertionPointToStart(&entryBlock);
  for (const auto &var_value :
       llvm::zip(node->parameters(), entryBlock.getArguments())) {
    auto &var = std::get<0>(var_value);
    auto varName = var->variable()->name();
    auto typeName = var->varType()->name();
    auto value = std::get<1>(var_value);
    auto alloca = createEntryBlockAlloca(getType(typeName), loc(node));
    _variableSymbols[varName] = alloca;
    cir::StoreOp::create(_builder, loc(node), value, alloca,
                         /*isVolatile=*/false,
                         /*alignment=*/mlir::IntegerAttr{},
                         /*sync_scope=*/cir::SyncScopeKindAttr(),
                         /*mem-order=*/cir::MemOrderAttr());
  }

  _functionSymbols[funcName] = getType(node->type()->name());
  _functionOps[funcName] = func;

  DBG("funcName: {0}", funcName);

  return func;
}

auto MLIRGenImpl::gen(const FunctionDecl *node) -> mlir::func::FuncOp {
  _variableSymbols = {};
  auto func = gen(node->proto().get());

  auto value = gen(node->body().get());

  auto location = loc(node->body()->expression().get());
  if (value)
    mlir::func::ReturnOp::create(_builder, location, value);
  else
    mlir::func::ReturnOp::create(_builder, location);

  return func;
}

auto MLIRGenImpl::gen(const StructDecl *node) -> void {
  auto &variables = node->variables();

  llvm::SmallVector<mlir::Type, 2> elementTypes;
  elementTypes.reserve(variables.size());
  for (auto &variable : variables) {
    elementTypes.push_back(getType(variable->varType()->name()));
  }

  mlir::StringAttr attr = _builder.getStringAttr(node->id()->name());
  cir::RecordType structTy =
      cir::RecordType::get(_builder.getContext(), attr, cir::RecordType::RecordKind::Struct);
  structTy.complete(elementTypes, false, false);

  _typeSymbols[node->id()->name()] = structTy;
  _structNodes[node->id()->name()] = node;
}

auto MLIRGenImpl::gen(const Expr *node) -> mlir::Value {
  switch (node->getKind()) {
  case Expr::Expr_Unit:
    return gen(cast<UnitExpr>(node));
  case Expr::Expr_DecimalLiteral:
    return gen(cast<DecimalLiteralExpr>(node));
  case Expr::Expr_BoolLiteral:
    return gen(cast<BoolLiteralExpr>(node));
  case Expr::Expr_Call:
    return gen(cast<CallExpr>(node));
  case Expr::Expr_Variable:
    return gen(cast<VariableExpr>(node));
  case Expr::Expr_Binary:
    return gen(cast<BinaryExpr>(node));
  case Expr::Expr_If:
    return gen(cast<IfExpr>(node));
  case Expr::Expr_While:
    return gen(cast<WhileExpr>(node));
  default:
    llvm_unreachable("Unexpected expression");
  }
}

auto MLIRGenImpl::gen(const UnitExpr *node) -> mlir::Value { return nullptr; }

auto MLIRGenImpl::gen(const BlockExpr *node) -> mlir::Value {
  for (auto &expr : *node)
    gen(expr.get());
  return gen(node->expression().get());
}

auto MLIRGenImpl::gen(const IfExpr *node) -> mlir::Value {
  DBG("IfExpr node type: {0}, then type: {1}, else type: {2}", node->type(), node->thenBlock()->type(), node->elseBlock()->type());
  auto cond = gen(node->conditionExpr().get());

  auto resultType = getType(node->type());
  auto ptrType = cir::PointerType::get(_builder.getContext(), resultType);
  cir::AllocaOp ifYieldResult = cir::AllocaOp::create(_builder,
      loc(node), ptrType, resultType,
      "if_yield_result", getAlignOne());

  cir::IfOp::create(
      _builder, loc(node), cond,
      /*withElseRegion=*/true,

      /*thenBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        auto thenBlock = node->thenBlock().get();
        auto thenValue = gen(thenBlock);
        cir::StoreOp::create(b, loc, thenValue, ifYieldResult,
                         /*isVolatile=*/false,
                         /*alignment=*/mlir::IntegerAttr{},
                         /*sync_scope=*/cir::SyncScopeKindAttr(),
                         /*mem-order=*/cir::MemOrderAttr());

        cir::YieldOp::create(b, loc);
      },

      /*elseBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        auto elseBlock = node->elseBlock().get();
        auto elseValue = gen(elseBlock);
        cir::StoreOp::create(b, loc, elseValue, ifYieldResult,
                         /*isVolatile=*/false,
                         /*alignment=*/mlir::IntegerAttr{},
                         /*sync_scope=*/cir::SyncScopeKindAttr(),
                         /*mem-order=*/cir::MemOrderAttr());

        cir::YieldOp::create(b, loc);
      });

  // return cir::LoadOp::create(_builder, loc(node), resultType, ifYieldResult);
  return cir::LoadOp::create(_builder, loc(node), {ifYieldResult});
}

auto MLIRGenImpl::gen(const WhileExpr *node) -> mlir::Value {
  auto conditionExprBuilder = [&](mlir::OpBuilder &builder, mlir::Location location) {
    auto cond = gen(node->conditionExpr().get());
    cir::ConditionOp::create(builder, loc(node->conditionExpr().get()), cond);
  };

  auto bodyBlock = node->bodyBlock().get();
  auto bodyExprBuilder = [&](mlir::OpBuilder &builder, mlir::Location location) {
    gen(bodyBlock);
    cir::YieldOp::create(builder, loc(bodyBlock->expression().get()));
  };

  cir::WhileOp::create(_builder, loc(node), conditionExprBuilder, bodyExprBuilder);

  return nullptr;
}

auto MLIRGenImpl::gen(const CallExpr *node) -> mlir::Value {
  auto functionName = node->name();
  if (functionName == builtins::print)
    return genPrint(node);

  DBG("gen(CallExpr). functionName: {0}", functionName);

  if (auto type = getType(functionName)) {
    if (auto recordType = llvm::dyn_cast<cir::RecordType>(type)) {
      return genStructLiteral(node, functionName, nullptr);
    }
  }

  llvm::SmallVector<mlir::Value, 4> operands;
  for (auto &expr : *node) {
    auto value = gen(expr.get());
    DBG("gen(CallExpr). value: {0}", value);
    operands.push_back(value);
  }

  if (functionName == builtins::boolToUInt64)
    return CastOp::create(_builder, loc(node), operands.front());

  auto result = _functionSymbols[functionName];
  llvm::SmallVector<mlir::Type, 4> results;
  if (result != _builder.getNoneType())
    results.push_back(result);

  auto calleeOpIter = _functionOps.find(node->name());
  if (calleeOpIter == _functionOps.end()) {
    // TODO: placeholder for functions implemented after the caller
    ERR("callee {0} DOESN'T exist.", node->name());
    return nullptr;
  }

  auto callOp = mlir::func::CallOp::create(_builder, loc(node), calleeOpIter->second, operands);

  return node->type() == builtins::UnitType ? nullptr : callOp.getResult(0);
}

auto MLIRGenImpl::genPrint(const CallExpr *node) -> mlir::Value {
  auto &expressions = node->expressions();
  auto operand = gen(expressions.front().get());
  return PrintOp::create(_builder, loc(node), operand);
}

auto MLIRGenImpl::gen(const VariableExpr *node) -> mlir::Value {
  auto address = _variableSymbols[node->name()];
  return cir::LoadOp::create(_builder, loc(node), address);
}

auto MLIRGenImpl::gen(const DecimalLiteralExpr *node) -> mlir::Value {
  mlir::Type type = getType(node->type());
  cir::IntAttr attr = cir::IntAttr::get(type, node->value());
  DBG("type: {0}, attr: {1}", type, attr);
  return cir::ConstantOp::create(_builder, loc(node), attr);
}

auto MLIRGenImpl::gen(const BoolLiteralExpr *node) -> mlir::Value {
  cir::BoolAttr attr = cir::BoolAttr::get(_builder.getContext(), node->value());
  DBG("attr: {0}", attr);
  return cir::ConstantOp::create(_builder, loc(node), attr);
}

auto MLIRGenImpl::gen(const BinaryExpr *node) -> mlir::Value {
  using Operator = BinaryExpr::Operator;
  auto op = node->opEnum();
  switch (op) {
  case Operator::Assign:
    return genAssignOp(node);
  case Operator::StructRead: {
    auto structValue = gen(node->lhs().get());
    auto index = node->index();
    return StructReadOp::create(_builder, loc(node), structValue, index);
  }
  default:
    break;
  }

  auto lhs = gen(node->lhs().get());
  auto rhs = gen(node->rhs().get());
  switch (op) {
  case Operator::Add:
    return cir::BinOp::create(_builder, loc(node), cir::BinOpKind::Add, lhs, rhs);
  case Operator::Diff:
    return cir::BinOp::create(_builder, loc(node), cir::BinOpKind::Sub, lhs, rhs);
  case Operator::Mul:
    return cir::BinOp::create(_builder, loc(node), cir::BinOpKind::Mul, lhs, rhs);
  case Operator::Div:
    return cir::BinOp::create(_builder, loc(node), cir::BinOpKind::Div, lhs, rhs);
  case Operator::Rem:
    return cir::BinOp::create(_builder, loc(node), cir::BinOpKind::Rem, lhs, rhs);
  case Operator::And:
    return cir::BinOp::create(_builder, loc(node), cir::BinOpKind::And, lhs, rhs);
  case Operator::Or:
    return cir::BinOp::create(_builder, loc(node), cir::BinOpKind::Or, lhs, rhs);
  case Operator::EQ:
    return cir::CmpOp::create(_builder, loc(node), cir::CmpOpKind::eq, lhs, rhs);
  case Operator::NEQ:
    return cir::CmpOp::create(_builder, loc(node), cir::CmpOpKind::ne, lhs, rhs);
  case Operator::LT:
    return cir::CmpOp::create(_builder, loc(node), cir::CmpOpKind::lt, lhs, rhs);
  case Operator::LE:
    return cir::CmpOp::create(_builder, loc(node), cir::CmpOpKind::le, lhs, rhs);
  case Operator::GT:
    return cir::CmpOp::create(_builder, loc(node), cir::CmpOpKind::gt, lhs, rhs);
  case Operator::GE:
    return cir::CmpOp::create(_builder, loc(node), cir::CmpOpKind::ge, lhs, rhs);
  default:
    llvm_unreachable("Unexpected statement");
  }
}

auto MLIRGenImpl::genAssignOp(const BinaryExpr *node) -> mlir::Value {
  auto rhs = gen(node->rhs().get());

  llvm::TypeSwitch<const Expr *>(node->lhs().get())
      .Case<VariableExpr>([&](const auto *var) {
        auto name = var->name();
        auto address = _variableSymbols[name];
        if (node->lhs()->type() != builtins::UnitType)
          cir::StoreOp::create(_builder, loc(node), rhs, address,
                               /*isVolatile=*/false,
                               /*alignment=*/mlir::IntegerAttr{},
                               /*sync_scope=*/cir::SyncScopeKindAttr(),
                               /*mem-order=*/cir::MemOrderAttr());
      })
      .Case<BinaryExpr>([&](const auto *structRead) {
        llvm::SmallVector<int64_t, 3> indexes = {};
        auto variable = structRead->lhs().get();
        indexes.push_back(structRead->index());
        while ((structRead = llvm::dyn_cast<BinaryExpr>(variable))) {
          variable = structRead->lhs().get();
          indexes.push_back(structRead->index());
        }
        std::reverse(indexes.begin(), indexes.end());

        auto memref =
            _variableSymbols[static_cast<VariableExpr *>(variable)->name()];
        auto structValue =
            mlir::memref::LoadOp::create(_builder, loc(node), memref);
        auto valueToStore = StructWriteOp::create(_builder,
            loc(node), structValue, indexes, rhs);
        mlir::memref::StoreOp::create(_builder, loc(node), valueToStore, memref);
#if 0
        cir::StoreOp::create(_builder, loc(node), valueToStore, memref,
                             /*isVolatile=*/false,
                             /*alignment=*/mlir::IntegerAttr{},
                             /*sync_scope=*/cir::SyncScopeKindAttr(),
                             /*mem-order=*/cir::MemOrderAttr());
#endif
      })
      .Default(
          [&](const Expr *) { llvm_unreachable("Unexpected expression"); });
  return nullptr;
}

auto MLIRGenImpl::gen(const Stat *node) -> void {
  switch (node->getKind()) {
  case Stat::Stat_VariableDecl:
    return gen(cast<VariableStat>(node));
  case Stat::Stat_Expression:
    return gen(cast<ExprStat>(node));
  }
}

auto MLIRGenImpl::getAlignOne() -> mlir::IntegerAttr {
  // Note that mlir::IntegerType is used instead of cir::IntType here because
  // we don't need sign information for this to be useful, so keep it simple.
  clang::CharUnits align = clang::CharUnits::One();
  return _builder.getI64IntegerAttr(align.getQuantity());
}

auto MLIRGenImpl::genStructLiteral(const CallExpr* callExpr, llvm::StringRef typeName, mlir::Value targetPtr) -> mlir::Value {

  DBG("typeName: {0}, callExpr->name(): {1}", typeName, callExpr->name());

  auto recordType = llvm::dyn_cast<cir::RecordType>(getType(typeName));
  if (!recordType) {
    ERR("NOT a RecordType. typeName: {0}", typeName);
    return nullptr;
  }
  if (recordType.getKind() != cir::RecordType::RecordKind::Struct) {
    ERR("record type kind: {0}", recordType.getKind());
    return nullptr;
  }
  mlir::Type recordPtrTy = cir::PointerType::get(recordType);

  mlir::Value varPtrOp;
  if (!targetPtr) {
    auto alloca = cir::AllocaOp::create(_builder, loc(callExpr), recordPtrTy, recordType,
                                     typeName, getAlignOne());
    DBG("typeName: {0}, alloca: {1}", typeName, varPtrOp);

    auto *parentBlock = alloca->getBlock();
    alloca->moveBefore(&parentBlock->front());

    varPtrOp = alloca;
  } else {
    varPtrOp = llvm::dyn_cast<cir::GetMemberOp>(targetPtr.getDefiningOp());
  }

  const StructDecl *declNode = _structNodes[typeName];
  auto& variables = declNode->variables();
  // DBG("typeName: {0}, declNode: {1}", typeName, declNode);

  int index = 0;
  for (auto& expr : *callExpr) {
    mlir::Type fieldTy = getType(expr->type());
    cir::PointerType fieldPtrTy = cir::PointerType::get(_builder.getContext(), fieldTy);

    auto& variable = variables[index]->variable();

    DBG("index: {0}, variable name: {1}, expr->type(): {2}", index, variable->name(), expr->type());
    auto memberPtr =
        cir::GetMemberOp::create(_builder, loc(callExpr), fieldPtrTy, varPtrOp, variable->name(), index);

    DBG();
    if (auto nestedCallExpr = llvm::dyn_cast<CallExpr>(expr.get())) {
      DBG("index: {0}, expr->type(): {1}, type: {2}", index, expr->type(), getType(expr->type()));
      genStructLiteral(nestedCallExpr, expr->type(), memberPtr);
    } else {
      mlir::Value val = gen(expr.get());
      cir::StoreOp::create(_builder, loc(callExpr), val, memberPtr,
                           /*isVolatile=*/false,
                           /*alignment=*/mlir::IntegerAttr{},
                           /*sync_scope=*/cir::SyncScopeKindAttr(),
                           /*mem-order=*/cir::MemOrderAttr());
    }

    index++;
  }

  return varPtrOp;
}

auto MLIRGenImpl::gen(const VariableStat *node) -> void {
  auto typeName = node->varType()->name();
  auto varName = node->variable()->name();

  if (auto recordType = llvm::dyn_cast<cir::RecordType>(getType(typeName))) {
    CallExpr *callExpr = llvm::dyn_cast<CallExpr>(node->init().get());
    if (!callExpr) {
      ERR("{0} is NOT a CallExpr.", varName);
      return;
    }

    auto structLiteral = genStructLiteral(callExpr, typeName, nullptr);
    _variableSymbols[varName] = structLiteral;
    return;
  }

  auto alloca = createEntryBlockAlloca(getType(typeName), loc(node));
  _variableSymbols[varName] = alloca;

  auto initValue = gen(node->init().get());

  if (typeName != builtins::UnitType)
    cir::StoreOp::create(_builder, loc(node), initValue, alloca,
                         /*isVolatile=*/false,
                         /*alignment=*/mlir::IntegerAttr{},
                         /*sync_scope=*/cir::SyncScopeKindAttr(),
                         /*mem-order=*/cir::MemOrderAttr());
}

auto MLIRGenImpl::gen(const ExprStat *node) -> void {
  gen(node->expression().get());
}

namespace cherry {

auto mlirGen(const llvm::SourceMgr &sourceManager, mlir::MLIRContext &context,
             const Module &moduleAST, mlir::OwningOpRef<mlir::ModuleOp> &module)
    -> CherryResult {
  auto generator = MLIRGenImpl(sourceManager, context);
  auto result = generator.gen(moduleAST);
  module = generator.module;
  return result;
}

} // end namespace cherry
