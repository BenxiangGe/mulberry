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
      return _builder.getI64Type();
    } else if (name == builtins::BoolType) {
      return _builder.getI1Type();
    } else {
      return _typeSymbols[name];
    }
  }

  auto createEntryBlockAlloca(mlir::Type mlirType, mlir::Location loc)
      -> mlir::Value {
    if (mlirType == getType(builtins::UnitType))
      return nullptr;
    auto memRefType = mlir::MemRefType::get({}, mlirType);
    auto alloca = _builder.create<mlir::memref::AllocaOp>(loc, memRefType);
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
    _builder.create<mlir::memref::StoreOp>(loc(node), value, alloca);
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
    _builder.create<mlir::func::ReturnOp>(location, value);
  else
    _builder.create<mlir::func::ReturnOp>(location, std::nullopt);

  return func;
}

auto MLIRGenImpl::gen(const StructDecl *node) -> void {
  auto &variables = node->variables();
  llvm::SmallVector<mlir::Type, 2> elementTypes;
  elementTypes.reserve(variables.size());
  for (auto &variable : variables) {
    auto type = getType(variable->varType()->name());
    elementTypes.push_back(type);
  }

  _typeSymbols[node->id()->name()] =
      mlir::cherry::CherryStructType::get(_builder.getContext(), elementTypes);
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
  auto cond = gen(node->conditionExpr().get());

  auto thenBlock = node->thenBlock().get();

  auto ifOp = _builder.create<mlir::scf::IfOp>(loc(node), getType(node->type()), cond, true);
  {
    mlir::OpBuilder::InsertionGuard guard(_builder);
    _builder.setInsertionPointToStart(ifOp.thenBlock());

    auto thenValue = gen(thenBlock);
    _builder.create<mlir::scf::YieldOp>(loc(thenBlock->expression().get()), thenValue);
  }

  auto elseBlock = node->elseBlock().get();
  {
    mlir::OpBuilder::InsertionGuard guard(_builder);
    _builder.setInsertionPointToStart(ifOp.elseBlock());
    auto elseValue = gen(elseBlock);
    _builder.create<mlir::scf::YieldOp>(loc(elseBlock->expression().get()), elseValue);
  }

  return ifOp.getResult(0);
}

auto MLIRGenImpl::gen(const WhileExpr *node) -> mlir::Value {
  auto conditionExprBuilder = [&](mlir::OpBuilder &builder, mlir::Location) {
    auto cond = gen(node->conditionExpr().get());
    builder.create<YieldWhileOp>(loc(node->conditionExpr().get()), cond);
  };

  auto bodyBlock = node->bodyBlock().get();
  auto bodyExprBuilder = [&](mlir::OpBuilder &builder, mlir::Location) {
    gen(bodyBlock);
    builder.create<YieldWhileOp>(loc(bodyBlock->expression().get()),
                                 std::nullopt);
  };

  _builder.create<WhileOp>(loc(node), conditionExprBuilder, bodyExprBuilder);
  return nullptr;
}

auto MLIRGenImpl::gen(const CallExpr *node) -> mlir::Value {
  auto functionName = node->name();
  if (functionName == builtins::print)
    return genPrint(node);

  llvm::SmallVector<mlir::Value, 4> operands;
  for (auto &expr : *node) {
    auto value = gen(expr.get());
    operands.push_back(value);
  }

  if (auto type = getType(functionName))
    return _builder.create<StructInitOp>(
        loc(node), llvm::cast<CherryStructType>(type), operands);

  if (functionName == builtins::boolToUInt64)
    return _builder.create<CastOp>(loc(node), operands.front());

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

  auto callOp = _builder.create<mlir::func::CallOp>(loc(node), calleeOpIter->second, operands);

  return node->type() == builtins::UnitType ? nullptr : callOp.getResult(0);
}

auto MLIRGenImpl::genPrint(const CallExpr *node) -> mlir::Value {
  auto &expressions = node->expressions();
  auto operand = gen(expressions.front().get());
  return _builder.create<PrintOp>(loc(node), operand);
}

auto MLIRGenImpl::gen(const VariableExpr *node) -> mlir::Value {
  auto address = _variableSymbols[node->name()];
  return _builder.create<mlir::memref::LoadOp>(loc(node), address);
}

auto MLIRGenImpl::gen(const DecimalLiteralExpr *node) -> mlir::Value {
  mlir::Type type = getType(node->type());
  DBG("type: {0}", type);
  mlir::TypedAttr attr = _builder.getIntegerAttr(type, node->value());
  DBG("attr: {0}", attr);
  return _builder.create<mlir::arith::ConstantOp>(loc(node), attr);
}

auto MLIRGenImpl::gen(const BoolLiteralExpr *node) -> mlir::Value {
  mlir::Type type = getType(node->type());
  DBG("type: {0}", type);
  mlir::TypedAttr attr = _builder.getIntegerAttr(type, node->value());
  DBG("attr: {0}", attr);
  return _builder.create<mlir::arith::ConstantOp>(loc(node), attr);
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
    return _builder.create<StructReadOp>(loc(node), structValue, index);
  }
  default:
    break;
  }

  auto lhs = gen(node->lhs().get());
  auto rhs = gen(node->rhs().get());
  switch (op) {
  case Operator::Add:
    return _builder.create<mlir::arith::AddIOp>(loc(node), lhs, rhs);
  case Operator::Diff:
    return _builder.create<mlir::arith::SubIOp>(loc(node), lhs, rhs);
  case Operator::Mul:
    return _builder.create<mlir::arith::MulIOp>(loc(node), lhs, rhs);
  case Operator::Div:
    return _builder.create<mlir::arith::DivUIOp>(loc(node), lhs, rhs);
  case Operator::Rem:
    return _builder.create<mlir::arith::RemUIOp>(loc(node), lhs, rhs);
  case Operator::And:
    return _builder.create<mlir::arith::AndIOp>(loc(node), lhs, rhs);
  case Operator::Or:
    return _builder.create<mlir::arith::OrIOp>(loc(node), lhs, rhs);
  case Operator::EQ:
    return _builder.create<mlir::arith::CmpIOp>(loc(node), mlir::arith::CmpIPredicate::eq, lhs, rhs);
  case Operator::NEQ:
    return _builder.create<mlir::arith::CmpIOp>(loc(node), mlir::arith::CmpIPredicate::ne, lhs, rhs);
  case Operator::LT:
    return _builder.create<mlir::arith::CmpIOp>(loc(node), mlir::arith::CmpIPredicate::ult, lhs, rhs);
  case Operator::LE:
    return _builder.create<mlir::arith::CmpIOp>(loc(node), mlir::arith::CmpIPredicate::ule, lhs, rhs);
  case Operator::GT:
    return _builder.create<mlir::arith::CmpIOp>(loc(node), mlir::arith::CmpIPredicate::ugt, lhs, rhs);
  case Operator::GE:
    return _builder.create<mlir::arith::CmpIOp>(loc(node), mlir::arith::CmpIPredicate::uge, lhs, rhs);
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
          _builder.create<mlir::memref::StoreOp>(loc(node), rhs, address);
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
            _builder.create<mlir::memref::LoadOp>(loc(node), memref);
        auto valueToStore = _builder.create<StructWriteOp>(
            loc(node), structValue, indexes, rhs);
        _builder.create<mlir::memref::StoreOp>(loc(node), valueToStore, memref);
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

auto MLIRGenImpl::gen(const VariableStat *node) -> void {
  auto typeName = node->varType()->name();
  auto varName = node->variable()->name();
  auto alloca = createEntryBlockAlloca(getType(typeName), loc(node));
  _variableSymbols[varName] = alloca;

  auto initValue = gen(node->init().get());

  if (typeName != builtins::UnitType)
    _builder.create<mlir::memref::StoreOp>(loc(node), initValue, alloca);
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
