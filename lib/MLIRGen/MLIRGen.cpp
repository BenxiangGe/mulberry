//===--- MLIRGen.cpp - MLIR Generator -------------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/MLIRGen.h"
#include "cherry/AST/AST.h"
#include "cherry/Basic/Builtins.h"
#include "cherry/Basic/CherryResult.h"
#include "cherry/Basic/Logging.h"
#include "cherry/Basic/ScopeStack.h"
#include "cherry/MLIRGen/IR/CherryNNOps.h"
#include "cherry/MLIRGen/IR/CherryOps.h"
#include "cherry/MLIRGen/IR/MulberryOps.h"
#include "cherry/MLIRGen/IR/MulberryTypes.h"
#include "cherry/MLIRGen/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/TypeSwitch.h"

#include <functional>
#include <map>
#include <string>
#include <string_view>

namespace {
using namespace mlir::cherry;
using namespace mlir::arith;
using namespace cherry;
using llvm::cast;
using llvm::failure;
using llvm::success;

#undef DEBUG_TYPE
#define DEBUG_TYPE "MLIRGen"

constexpr int64_t kGlobalTensorLiteralElementThreshold = 64;

template <typename T>
using NameMap = std::map<std::string, T, std::less<>>;

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
  ScopeStack<NameMap<mlir::Value>> _variableScopes;
  NameMap<mlir::func::FuncOp> _functionsByName;
  llvm::StringRef _fileNameIdentifier;
  int _globalTensorCounter = 0;
  MLIRTypeConverter _typeConverter{_builder};

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
  auto gen(const ForExpr *node) -> mlir::Value;
  auto genPrint(const CallExpr *node) -> mlir::Value;
  auto genMatmul(const CallExpr *node) -> mlir::Value;
  auto genMatadd(const CallExpr *node) -> mlir::Value;
  auto genTranspose(const CallExpr *node) -> mlir::Value;
  auto genElementwiseNN(const CallExpr *node) -> mlir::Value;
  auto genArgmax(const CallExpr *node) -> mlir::Value;
  auto genSize(const CallExpr *node) -> mlir::Value;
  auto gen(const CallExpr *node) -> mlir::Value;
  auto gen(const StructLiteralExpr *node) -> mlir::Value;
  auto gen(const VariableExpr *node) -> mlir::Value;
  auto gen(const MemberExpr *node) -> mlir::Value;
  auto gen(const DecimalLiteralExpr *node) -> mlir::Value;
  auto gen(const FloatLiteralExpr *node) -> mlir::Value;
  auto gen(const BoolLiteralExpr *node) -> mlir::Value;
  auto gen(const AssignExpr *node) -> mlir::Value;
  auto gen(const BinaryExpr *node) -> mlir::Value;

  // Statements
  auto gen(const Stat *node) -> void;
  auto gen(const VariableStat *node) -> void;
  auto gen(const ExprStat *node) -> void;

  template <typename T>
  void setSymbol(NameMap<T> &symbols, std::string_view name, T value) {
    symbols[std::string(name)] = value;
  }

  void resetVariableScopes() {
    _variableScopes.reset();
    enterVariableScope();
  }

  void enterVariableScope() { _variableScopes.enterScope(); }

  void leaveVariableScope() { _variableScopes.leaveScope(); }

  void setVariable(std::string_view name, mlir::Value value) {
    if (_variableScopes.empty())
      enterVariableScope();
    setSymbol(_variableScopes.currentScope(), name, value);
  }

  auto getVariable(std::string_view name) -> mlir::Value {
    if (auto *variable = _variableScopes.lookup(name))
      return *variable;
    return {};
  }

  void setFunction(std::string_view name, mlir::func::FuncOp func) {
    setSymbol(_functionsByName, name, func);
  }

  auto findFunction(std::string_view name) {
    return _functionsByName.find(name);
  }

  auto genStructLiteral(const StructLiteralExpr *structLiteral,
                        const StructType *structType,
                        mlir::Value targetPtr) -> mlir::Value;
  auto getPtrType(mlir::Type elementType) const -> mlir::mulberry::PtrType;
  auto createAlloca(mlir::Type mlirType, mlir::Location location)
      -> mlir::Value;
  auto createLoad(mlir::Value ptr, mlir::Type type, mlir::Location location)
      -> mlir::Value;
  void createStore(mlir::Value value, mlir::Value ptr,
                   mlir::Location location);
  auto createStructFieldPtr(mlir::Value recordPtr, const StructField& field,
                            mlir::Location location) -> mlir::Value;
  auto gen(const TensorLiteralExpr *expr) -> mlir::Value;
  auto genGlobalTensorLiteral(const TensorLiteralExpr *expr,
                              std::string_view varName) -> mlir::Value;
  void storeTensorElements(const TensorLiteralExpr *expr, mlir::Value memref,
                           mlir::Type elementType,
                           llvm::SmallVectorImpl<mlir::Value> &indices);
  mlir::Value gen(const TensorAccessExpr *expr, bool isLValue = false);
  void genAssignment(const TensorAccessExpr *lhs, const Expr *rhs);

  auto genLValue(const Expr *node) -> mlir::Value;
  auto genRValue(const Expr *node) -> mlir::Value;
  auto getStructField(const MemberExpr *memberExpr) const
      -> const StructField *;
  auto genIndexValue(const Expr *node) -> mlir::Value;
  auto genMemRefElementValue(const Expr *node, mlir::Type elementType)
      -> mlir::Value;
  auto genMemRefLoadValue(const TensorAccessExpr *expr) -> mlir::Value;
  auto castToType(mlir::Value value, mlir::Type type, mlir::Location location)
      -> mlir::Value;
  auto getMLIRType(const Type *type) const -> mlir::Type;
  auto getMLIRType(const Expr *expr) const -> mlir::Type;
  auto getMemRefType(const Type *type) const -> mlir::MemRefType;
  auto getMemRefType(const Expr *expr) const -> mlir::MemRefType;
  auto getTensorElementCount(mlir::MemRefType memRefType) -> int64_t;
  auto isConstantTensorData(const TensorLiteralExpr *expr) -> bool;
  auto shouldPromoteConstTensorData(const TensorLiteralExpr *expr) -> bool;
  auto getTensorDataAttr(const Expr *expr, mlir::Type elementType)
      -> mlir::Attribute;
  void collectTensorDataAttrs(const TensorLiteralExpr *expr,
                              mlir::Type elementType,
                              llvm::SmallVectorImpl<mlir::Attribute> &attrs);
  // Utility
  auto loc(const Node *node) -> mlir::Location {
    auto [line, col] = _sourceManager.getLineAndColumn(node->location());
    return mlir::FileLineColLoc::get(
        _builder.getStringAttr(_fileNameIdentifier), line, col);
  }

};

} // end namespace

auto MLIRGenImpl::gen(const Module &node) -> CherryResult {
  module = mlir::ModuleOp::create(_builder.getUnknownLoc());

  for (auto &decl : node) {
    if (auto *op = gen(decl.get()))
      module.push_back(op);
  }

#if 1
  if (failed(mlir::verify(module))) {
    module.emitError("module verification error");
    return failure();
  }
#endif

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
  llvm::SmallVector<mlir::Type, 3> argTypes;
  for (auto &param : node->parameters()) {
    auto *paramType = param->type();
    argTypes.push_back(getMLIRType(paramType));
  }

  auto funcName = node->id()->name();
  auto *returnType = node->type();
  llvm::SmallVector<mlir::Type, 1> resultTypes;
  if (!cherry::isUnitType(returnType))
    resultTypes.push_back(getMLIRType(returnType));
  auto funcType = _builder.getFunctionType(argTypes, resultTypes);
  mlir::OperationState state(loc(node), mlir::func::FuncOp::getOperationName());
  mlir::func::FuncOp::build(_builder, state, funcName, funcType);
  auto func = llvm::cast<mlir::func::FuncOp>(mlir::Operation::create(state));

  auto &entryBlock = *func.addEntryBlock();
  _builder.setInsertionPointToStart(&entryBlock);
  for (const auto &varValue :
       llvm::zip(node->parameters(), entryBlock.getArguments())) {
    auto &var = std::get<0>(varValue);
    auto varName = var->variable()->name();
    auto value = std::get<1>(varValue);
    auto *paramType = var->type();
    if (cherry::isTensorType(paramType)) {
      setVariable(varName, value);
      continue;
    }
    if (cherry::isUnitType(paramType)) {
      setVariable(varName, nullptr);
      continue;
    }

    auto alloca = createAlloca(getMLIRType(paramType), loc(node));
    setVariable(varName, alloca);
    createStore(value, alloca, loc(node));
  }

  setFunction(funcName, func);

  DBG("funcName: {0}", funcName);

  return func;
}

auto MLIRGenImpl::gen(const FunctionDecl *node) -> mlir::func::FuncOp {
  resetVariableScopes();
  auto func = gen(node->proto().get());

  auto value = gen(node->body().get());

  auto location = loc(node->body()->expression().get());
  if (value) {
    auto *returnType = node->proto()->type();
    value = castToType(value, getMLIRType(returnType), location);
    llvm::SmallVector<mlir::Value, 1> returnValues{value};
    mlir::func::ReturnOp::create(_builder, location, returnValues);
  } else {
    mlir::func::ReturnOp::create(_builder, location);
  }

  return func;
}

auto MLIRGenImpl::gen(const StructDecl *node) -> void {
  if (auto *structType = cherry::getStructType(node->id()->type())) {
    getMLIRType(structType);
    return;
  }

  ERR("struct `{0}` has no Cherry type", node->id()->name());
}

auto MLIRGenImpl::gen(const Expr *node) -> mlir::Value {
  switch (node->getKind()) {
  case Expr::Expr_Unit:
    return gen(cast<UnitExpr>(node));
  case Expr::Expr_DecimalLiteral:
    return gen(cast<DecimalLiteralExpr>(node));
  case Expr::Expr_FloatLiteral:
    return gen(cast<FloatLiteralExpr>(node));
  case Expr::Expr_BoolLiteral:
    return gen(cast<BoolLiteralExpr>(node));
  case Expr::Expr_Call:
    return gen(cast<CallExpr>(node));
  case Expr::Expr_StructLiteral:
    return gen(cast<StructLiteralExpr>(node));
  case Expr::Expr_Variable:
    return gen(cast<VariableExpr>(node));
  case Expr::Expr_Member:
    return gen(cast<MemberExpr>(node));
  case Expr::Expr_TensorLiteral:
    return gen(cast<TensorLiteralExpr>(node));
  case Expr::Expr_TensorAccess:
    return gen(cast<TensorAccessExpr>(node));
  case Expr::Expr_Assign:
    return gen(cast<AssignExpr>(node));
  case Expr::Expr_Binary:
    return gen(cast<BinaryExpr>(node));
  case Expr::Expr_If:
    return gen(cast<IfExpr>(node));
  case Expr::Expr_While:
    return gen(cast<WhileExpr>(node));
  case Expr::Expr_For:
    return gen(cast<ForExpr>(node));
  default:
    llvm_unreachable("Unexpected expression");
  }
}

auto MLIRGenImpl::gen(const UnitExpr *node) -> mlir::Value { return nullptr; }

auto MLIRGenImpl::gen(const BlockExpr *node) -> mlir::Value {
  enterVariableScope();
  for (auto &expr : *node)
    gen(expr.get());
  auto value = gen(node->expression().get());
  leaveVariableScope();
  return value;
}

auto MLIRGenImpl::gen(const IfExpr *node) -> mlir::Value {
  DBG("IfExpr Cherry type: {0}, then type: {1}, else type: {2}",
      formatType(node->type()),
      formatType(node->thenBlock()->type()),
      formatType(node->elseBlock()->type()));
  auto cond = gen(node->conditionExpr().get());

  if (cherry::isUnitType(node->type())) {
    mlir::scf::IfOp::create(
        _builder, loc(node), cond,
        [&](mlir::OpBuilder &builder, mlir::Location location) {
          gen(node->thenBlock().get());
          mlir::scf::YieldOp::create(builder, location);
        },
        [&](mlir::OpBuilder &builder, mlir::Location location) {
          gen(node->elseBlock().get());
          mlir::scf::YieldOp::create(builder, location);
        });
    return nullptr;
  }

  auto resultType = getMLIRType(node);
  auto ifOp = mlir::scf::IfOp::create(
      _builder, loc(node), mlir::TypeRange{resultType}, cond,
      /*withElseRegion=*/true);

  _builder.setInsertionPointToStart(ifOp.thenBlock());
  auto thenValue = gen(node->thenBlock().get());
  mlir::scf::YieldOp::create(_builder, loc(node->thenBlock().get()),
                             thenValue);

  _builder.setInsertionPointToStart(ifOp.elseBlock());
  auto elseValue = gen(node->elseBlock().get());
  mlir::scf::YieldOp::create(_builder, loc(node->elseBlock().get()),
                             elseValue);

  _builder.setInsertionPointAfter(ifOp);

  return ifOp.getResult(0);
}

auto MLIRGenImpl::gen(const WhileExpr *node) -> mlir::Value {
  auto bodyBlock = node->bodyBlock().get();
  mlir::scf::WhileOp::create(
      _builder, loc(node), mlir::TypeRange{}, mlir::ValueRange{},
      [&](mlir::OpBuilder &builder, mlir::Location location,
          mlir::ValueRange args) {
        auto cond = gen(node->conditionExpr().get());
        mlir::scf::ConditionOp::create(builder, location, cond,
                                       mlir::ValueRange{});
      },
      [&](mlir::OpBuilder &builder, mlir::Location location,
          mlir::ValueRange args) {
        gen(bodyBlock);
        mlir::scf::YieldOp::create(builder, location);
      });

  return nullptr;
}

auto MLIRGenImpl::gen(const ForExpr *node) -> mlir::Value {
  auto forLocation = loc(node);
  auto inductionType = getMLIRType(node->startExpr().get());
  auto inductionPtr = createAlloca(inductionType, forLocation);

  auto startValue =
      castToType(gen(node->startExpr().get()), inductionType, forLocation);
  auto endValue =
      castToType(gen(node->endExpr().get()), inductionType, forLocation);
  auto intType = llvm::cast<mlir::IntegerType>(inductionType);
  auto oneValue = mlir::arith::ConstantIntOp::create(
      _builder, forLocation, 1, intType.getWidth());

  enterVariableScope();
  setVariable(node->variableName(), inductionPtr);

  auto bodyBlock = node->bodyBlock().get();
  mlir::scf::ForOp::create(
      _builder, forLocation, startValue, endValue, oneValue,
      mlir::ValueRange{},
      [&](mlir::OpBuilder &builder, mlir::Location location,
          mlir::Value inductionValue, mlir::ValueRange args) {
        createStore(inductionValue, inductionPtr, location);
        gen(bodyBlock);
        mlir::scf::YieldOp::create(builder, location);
      },
      /*unsignedCmp=*/true);
  leaveVariableScope();
  return nullptr;
}

auto MLIRGenImpl::gen(const CallExpr *node) -> mlir::Value {
  auto name = node->name();
  DBG("gen(CallExpr). functionName: {0}", name);

  if (name == builtins::print)
    return genPrint(node);
  if (name == nn::matmul)
    return genMatmul(node);
  if (name == nn::matadd)
    return genMatadd(node);
  if (name == nn::transpose)
    return genTranspose(node);
  if (name == nn::exp || name == nn::sigmoid)
    return genElementwiseNN(node);
  if (name == nn::argmax)
    return genArgmax(node);
  if (name == builtins::size)
    return genSize(node);

  llvm::SmallVector<mlir::Value, 4> operands;
  for (auto &expr : *node) {
    auto value = gen(expr.get());
    DBG("gen(CallExpr). value: {0}", value);
    operands.push_back(value);
  }

  if (name == builtins::boolToUInt64)
    return mlir::arith::ExtUIOp::create(_builder, loc(node),
                                        _builder.getI64Type(),
                                        operands.front());

  auto calleeOpIter = findFunction(node->name());
  if (calleeOpIter == _functionsByName.end()) {
    // TODO: placeholder for functions implemented after the caller
    ERR("callee {0} DOESN'T exist.", node->name());
    return nullptr;
  }

  auto isUnitCall = cherry::isUnitType(node->type());
  auto callResultType = isUnitCall ? mlir::Type{} : getMLIRType(node);
  llvm::SmallVector<mlir::Type, 1> resultTypes;
  if (!isUnitCall)
    resultTypes.push_back(callResultType);
  auto callOp = mlir::func::CallOp::create(_builder, loc(node), name,
                                           resultTypes, operands);

  return isUnitCall ? nullptr : callOp.getResult(0);
}

auto MLIRGenImpl::gen(const StructLiteralExpr *node) -> mlir::Value {
  auto *structType = node->structType();
  if (!structType) {
    ERR("struct literal `{0}` has no Cherry struct type", node->name());
    return nullptr;
  }

  DBG("use Cherry struct literal `{0}`",
      formatType(structType));
  auto ptr = genStructLiteral(node, structType, nullptr);
  return createLoad(ptr, getMLIRType(structType), loc(node));
}

auto MLIRGenImpl::genPrint(const CallExpr *node) -> mlir::Value {
  auto &expressions = node->expressions();
  auto *expr = expressions.front().get();
  auto operand = llvm::isa<TensorAccessExpr>(expr)
                     ? genMemRefLoadValue(llvm::cast<TensorAccessExpr>(expr))
                     : gen(expr);
  return PrintOp::create(_builder, loc(node), operand);
}

auto MLIRGenImpl::genMatmul(const CallExpr *node) -> mlir::Value {
  auto &expressions = node->expressions();
  auto lhs = gen(expressions[0].get());
  auto rhs = gen(expressions[1].get());
  auto outType = getMemRefType(node);
  auto out = mlir::memref::AllocOp::create(_builder, loc(node), outType);
  mlir::cherry_nn::MatmulOp::create(_builder, loc(node), lhs, rhs, out);
  return out;
}

auto MLIRGenImpl::genMatadd(const CallExpr *node) -> mlir::Value {
  auto &expressions = node->expressions();
  auto lhs = gen(expressions[0].get());
  auto rhs = gen(expressions[1].get());
  auto outType = getMemRefType(node);
  auto out = mlir::memref::AllocOp::create(_builder, loc(node), outType);
  mlir::cherry_nn::MataddOp::create(_builder, loc(node), lhs, rhs, out);
  return out;
}

auto MLIRGenImpl::genTranspose(const CallExpr *node) -> mlir::Value {
  auto &expressions = node->expressions();
  auto input = gen(expressions[0].get());
  auto outType = getMemRefType(node);
  auto out = mlir::memref::AllocOp::create(_builder, loc(node), outType);
  mlir::cherry_nn::TransposeOp::create(_builder, loc(node), input, out);
  return out;
}

auto MLIRGenImpl::genElementwiseNN(const CallExpr *node) -> mlir::Value {
  auto &expressions = node->expressions();
  auto input = gen(expressions[0].get());
  auto outType = getMemRefType(node);
  auto out = mlir::memref::AllocOp::create(_builder, loc(node), outType);

  if (node->name() == nn::exp) {
    mlir::cherry_nn::ExpOp::create(_builder, loc(node), input, out);
    return out;
  }

  mlir::cherry_nn::SigmoidOp::create(_builder, loc(node), input, out);
  return out;
}

auto MLIRGenImpl::genArgmax(const CallExpr *node) -> mlir::Value {
  auto &expressions = node->expressions();
  auto input = gen(expressions[0].get());
  auto op = mlir::cherry_nn::ArgmaxOp::create(_builder, loc(node),
                                              _builder.getI64Type(), input);
  return op.getResult();
}

auto MLIRGenImpl::genSize(const CallExpr *node) -> mlir::Value {
  auto &expressions = node->expressions();
  auto *tensorType = cherry::getTensorType(expressions.front()->type());
  if (!tensorType) {
    ERR("size() argument has no Cherry tensor type");
    return nullptr;
  }

  auto size = tensorType->shape().front();
  DBG("size() static tensor size: {0}", size);
  return mlir::arith::ConstantIntOp::create(_builder, loc(node), size, 64);
}

auto MLIRGenImpl::gen(const VariableExpr *node) -> mlir::Value {
  auto address = getVariable(node->name());
  if (cherry::isTensorType(node->type()))
    return address;
  return createLoad(address, getMLIRType(node), loc(node));
}

auto MLIRGenImpl::gen(const MemberExpr *node) -> mlir::Value {
  auto ptr = genLValue(node);
  return createLoad(ptr, getMLIRType(node), loc(node));
}

auto MLIRGenImpl::gen(const DecimalLiteralExpr *node) -> mlir::Value {
  mlir::Type type = getMLIRType(node);
  auto intType = llvm::cast<mlir::IntegerType>(type);
  return mlir::arith::ConstantIntOp::create(_builder, loc(node),
                                            node->value(),
                                            intType.getWidth());
}

auto MLIRGenImpl::gen(const FloatLiteralExpr *node) -> mlir::Value {
  mlir::Type type = getMLIRType(node);
  return mlir::arith::ConstantFloatOp::create(
      _builder, loc(node), llvm::cast<mlir::FloatType>(type), node->value());
}

auto MLIRGenImpl::gen(const BoolLiteralExpr *node) -> mlir::Value {
  return mlir::arith::ConstantIntOp::create(_builder, loc(node),
                                            node->value(), 1);
}

auto MLIRGenImpl::genLValue(const Expr *node) -> mlir::Value {
  if (auto varExpr = llvm::dyn_cast<VariableExpr>(node)) {
    DBG("varExpr->name(): {0}", varExpr->name());
    auto parentAddress = getVariable(varExpr->name());
    return parentAddress;
  }

  if (auto *memberExpr = llvm::dyn_cast<MemberExpr>(node)) {
    auto base = memberExpr->base().get();
    mlir::Value basePtr = genLValue(base);

    if (auto *field = getStructField(memberExpr)) {
      return createStructFieldPtr(basePtr, *field, loc(node));
    }

    ERR("struct member access has no Cherry field information");
    return nullptr;
  }

  ERR("unknown EXPR");
  exit(-1);

  return nullptr;
}

auto MLIRGenImpl::getStructField(const MemberExpr *memberExpr) const
    -> const StructField * {
  auto *base = memberExpr->base().get();
  auto *structType = cherry::getStructType(base->type());
  if (!structType)
    return nullptr;

  auto index = memberExpr->fieldIndex();
  auto &fields = structType->fields();
  if (index >= fields.size()) {
    DBG("Cherry struct field index `{0}` out of bounds for `{1}`", index,
        formatType(structType));
    return nullptr;
  }

  auto *field = &fields[index];
  if (!field->type()) {
    DBG("Cherry struct field `{0}` has no Cherry type",
        field->name());
    return nullptr;
  }

  DBG("use Cherry struct field `{0}` from `{1}`", field->name(),
      formatType(structType));
  return field;
}

auto MLIRGenImpl::genRValue(const Expr *node) -> mlir::Value {
  if (auto *memberExpr = llvm::dyn_cast<MemberExpr>(node)) {
    mlir::Value ptr = genLValue(memberExpr);
    mlir::Value val = createLoad(ptr, getMLIRType(memberExpr),
                                 loc(memberExpr));
    return val;
  }

  return gen(node);
}

auto MLIRGenImpl::gen(const BinaryExpr *node) -> mlir::Value {
  using Operator = BinaryExpr::Operator;
  auto op = node->opEnum();

  auto lhs =
      castToType(gen(node->lhs().get()),
                 getMLIRType(node->lhs().get()), loc(node->lhs().get()));
  auto rhs =
      castToType(gen(node->rhs().get()),
                 getMLIRType(node->rhs().get()), loc(node->rhs().get()));
  if (llvm::isa<mlir::FloatType>(lhs.getType())) {
    switch (op) {
    case Operator::Add:
      return mlir::arith::AddFOp::create(_builder, loc(node), lhs, rhs);
    case Operator::Diff:
      return mlir::arith::SubFOp::create(_builder, loc(node), lhs, rhs);
    case Operator::Mul:
      return mlir::arith::MulFOp::create(_builder, loc(node), lhs, rhs);
    case Operator::Div:
      return mlir::arith::DivFOp::create(_builder, loc(node), lhs, rhs);
    case Operator::Rem:
      return mlir::arith::RemFOp::create(_builder, loc(node), lhs, rhs);
    case Operator::EQ:
      return mlir::arith::CmpFOp::create(
          _builder, loc(node), mlir::arith::CmpFPredicate::OEQ, lhs, rhs);
    case Operator::NEQ:
      return mlir::arith::CmpFOp::create(
          _builder, loc(node), mlir::arith::CmpFPredicate::ONE, lhs, rhs);
    case Operator::LT:
      return mlir::arith::CmpFOp::create(
          _builder, loc(node), mlir::arith::CmpFPredicate::OLT, lhs, rhs);
    case Operator::LE:
      return mlir::arith::CmpFOp::create(
          _builder, loc(node), mlir::arith::CmpFPredicate::OLE, lhs, rhs);
    case Operator::GT:
      return mlir::arith::CmpFOp::create(
          _builder, loc(node), mlir::arith::CmpFPredicate::OGT, lhs, rhs);
    case Operator::GE:
      return mlir::arith::CmpFOp::create(
          _builder, loc(node), mlir::arith::CmpFPredicate::OGE, lhs, rhs);
    default:
      break;
    }
  }

  switch (op) {
  case Operator::Add:
    return mlir::arith::AddIOp::create(_builder, loc(node), lhs, rhs);
  case Operator::Diff:
    return mlir::arith::SubIOp::create(_builder, loc(node), lhs, rhs);
  case Operator::Mul:
    return mlir::arith::MulIOp::create(_builder, loc(node), lhs, rhs);
  case Operator::Div:
    return mlir::arith::DivUIOp::create(_builder, loc(node), lhs, rhs);
  case Operator::Rem:
    return mlir::arith::RemUIOp::create(_builder, loc(node), lhs, rhs);
  case Operator::And:
    return mlir::arith::AndIOp::create(_builder, loc(node), lhs, rhs);
  case Operator::Or:
    return mlir::arith::OrIOp::create(_builder, loc(node), lhs, rhs);
  case Operator::EQ:
    return mlir::arith::CmpIOp::create(
        _builder, loc(node), mlir::arith::CmpIPredicate::eq, lhs, rhs);
  case Operator::NEQ:
    return mlir::arith::CmpIOp::create(
        _builder, loc(node), mlir::arith::CmpIPredicate::ne, lhs, rhs);
  case Operator::LT:
    return mlir::arith::CmpIOp::create(
        _builder, loc(node), mlir::arith::CmpIPredicate::ult, lhs, rhs);
  case Operator::LE:
    return mlir::arith::CmpIOp::create(
        _builder, loc(node), mlir::arith::CmpIPredicate::ule, lhs, rhs);
  case Operator::GT:
    return mlir::arith::CmpIOp::create(
        _builder, loc(node), mlir::arith::CmpIPredicate::ugt, lhs, rhs);
  case Operator::GE:
    return mlir::arith::CmpIOp::create(
        _builder, loc(node), mlir::arith::CmpIPredicate::uge, lhs, rhs);
  }

  llvm_unreachable("Unexpected statement");
}

auto MLIRGenImpl::gen(const AssignExpr *node) -> mlir::Value {
  llvm::TypeSwitch<const Expr *>(node->lhs().get())
      .Case<VariableExpr>([&](const auto *var) {
        auto name = var->name();
        auto rhs = gen(node->rhs().get());
        if (cherry::isTensorType(var->type())) {
          setVariable(name, rhs);
          return;
        }

        auto address = getVariable(name);
        if (!cherry::isUnitType(node->lhs()->type())) {
          rhs = castToType(rhs, getMLIRType(node->lhs().get()), loc(node));
          createStore(rhs, address, loc(node));
        }
      })
      .Case<MemberExpr>([&](const auto *member) {
        mlir::Value lhsPtr = genLValue(member);
        auto rhs = castToType(gen(node->rhs().get()),
                              getMLIRType(member), loc(node));

        createStore(rhs, lhsPtr, loc(node));
      })
      .Case<TensorAccessExpr>([&](const auto *tensorAccess) {
        genAssignment(tensorAccess, node->rhs().get());
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

auto MLIRGenImpl::getPtrType(mlir::Type elementType) const
    -> mlir::mulberry::PtrType {
  return mlir::mulberry::PtrType::get(_builder.getContext(), elementType);
}

auto MLIRGenImpl::createAlloca(mlir::Type mlirType,
                               mlir::Location location) -> mlir::Value {
  auto alloca = mlir::mulberry::AllocaOp::create(
      _builder, location, getPtrType(mlirType), mlirType);
  auto *parentBlock = alloca.getOperation()->getBlock();
  alloca.getOperation()->moveBefore(&parentBlock->front());
  return alloca;
}

auto MLIRGenImpl::createLoad(mlir::Value ptr, mlir::Type type,
                             mlir::Location location) -> mlir::Value {
  return mlir::mulberry::LoadOp::create(_builder, location, type, ptr);
}

void MLIRGenImpl::createStore(mlir::Value value, mlir::Value ptr,
                              mlir::Location location) {
  mlir::mulberry::StoreOp::create(_builder, location, value, ptr);
}

auto MLIRGenImpl::createStructFieldPtr(mlir::Value recordPtr,
                                       const StructField& field,
                                       mlir::Location location)
    -> mlir::Value {
  auto fieldType = getMLIRType(field.type());
  auto fieldPtrType = getPtrType(fieldType);
  return mlir::mulberry::RecordGetFieldOp::create(
      _builder, location, fieldPtrType, recordPtr,
      std::string(field.name()));
}

auto MLIRGenImpl::genStructLiteral(const StructLiteralExpr *structLiteral,
                                   const StructType *structType,
                                   mlir::Value targetPtr) -> mlir::Value {
  DBG("structType: {0}, structLiteral->name(): {1}",
      formatType(structType),
      structLiteral->name());

  auto recordType =
      llvm::dyn_cast<mlir::mulberry::RecordType>(getMLIRType(structType));
  if (!recordType) {
    ERR("NOT a RecordType. structType: {0}",
        formatType(structType));
    return nullptr;
  }

  auto &fields = structType->fields();
  if (fields.size() != structLiteral->expressions().size()) {
    ERR("Cherry struct literal field count mismatch for `{0}`",
        formatType(structType));
    return nullptr;
  }

  for (auto &field : fields) {
    if (!field.type()) {
      ERR("Cherry struct literal field `{0}` has no Cherry type",
          field.name());
      return nullptr;
    }
  }

  mlir::Value varPtrOp;
  if (!targetPtr) {
    auto alloca = createAlloca(recordType, loc(structLiteral));
    DBG("structType: {0}, alloca: {1}", formatType(structType),
        alloca);

    varPtrOp = alloca;
  } else {
    varPtrOp = targetPtr;
  }

  unsigned index = 0;
  for (auto &expr : *structLiteral) {
    auto &field = fields[index];
    DBG("Cherry struct literal field index: {0}, field name: {1}, "
        "expr type: {2}",
        field.index(), field.name(),
        formatType(expr->type()));
    auto memberPtr = createStructFieldPtr(varPtrOp, field,
                                          loc(structLiteral));

    if (auto *nestedStructLiteral =
            llvm::dyn_cast<StructLiteralExpr>(expr.get())) {
      auto *nestedStructType = nestedStructLiteral->structType();
      if (nestedStructType) {
        DBG("Cherry nested struct literal index: {0}, expr type: {1}, "
            "type: {2}",
            field.index(), formatType(expr->type()),
            getMLIRType(expr.get()));
        genStructLiteral(nestedStructLiteral, nestedStructType, memberPtr);
      } else {
        ERR("nested struct literal has no Cherry struct type");
        return nullptr;
      }
    } else {
      mlir::Value val = castToType(gen(expr.get()), getMLIRType(field.type()),
                                   loc(expr.get()));
      createStore(val, memberPtr, loc(structLiteral));
    }

    index++;
  }

  return varPtrOp;
}

auto MLIRGenImpl::getMLIRType(const Type *type) const -> mlir::Type {
  if (!type)
    return {};

  DBG("convert Cherry type `{0}` to MLIR type", formatType(type));
  return _typeConverter.convert(type);
}

auto MLIRGenImpl::getMLIRType(const Expr *expr) const -> mlir::Type {
  return getMLIRType(expr->type());
}

auto MLIRGenImpl::getMemRefType(const Type *type) const
    -> mlir::MemRefType {
  return llvm::dyn_cast_if_present<mlir::MemRefType>(getMLIRType(type));
}

auto MLIRGenImpl::getMemRefType(const Expr *expr) const -> mlir::MemRefType {
  return getMemRefType(expr->type());
}

auto MLIRGenImpl::getTensorElementCount(mlir::MemRefType memRefType)
    -> int64_t {
  int64_t count = 1;
  for (auto dim : memRefType.getShape()) {
    if (dim < 0)
      return -1;
    count *= dim;
  }
  return count;
}

auto MLIRGenImpl::isConstantTensorData(const TensorLiteralExpr *expr) -> bool {
  for (auto &element : expr->getElements()) {
    auto *childExpr = element.get();
    if (auto *nestedTensor = llvm::dyn_cast<TensorLiteralExpr>(childExpr)) {
      if (!isConstantTensorData(nestedTensor))
        return false;
      continue;
    }

    if (!llvm::isa<DecimalLiteralExpr, FloatLiteralExpr, BoolLiteralExpr>(
            childExpr))
      return false;
  }
  return true;
}

auto MLIRGenImpl::shouldPromoteConstTensorData(const TensorLiteralExpr *expr)
    -> bool {
  auto memRefType = getMemRefType(expr);
  auto elementCount = getTensorElementCount(memRefType);
  if (elementCount < kGlobalTensorLiteralElementThreshold)
    return false;
  return isConstantTensorData(expr);
}

auto MLIRGenImpl::getTensorDataAttr(const Expr *expr, mlir::Type elementType)
    -> mlir::Attribute {
  if (auto *decimal = llvm::dyn_cast<DecimalLiteralExpr>(expr)) {
    if (auto intType = llvm::dyn_cast<mlir::IntegerType>(elementType))
      return mlir::IntegerAttr::get(
          elementType, llvm::APInt(intType.getWidth(), decimal->value()));
  }

  if (auto *boolean = llvm::dyn_cast<BoolLiteralExpr>(expr)) {
    auto intType = llvm::cast<mlir::IntegerType>(elementType);
    return mlir::IntegerAttr::get(
        elementType, llvm::APInt(intType.getWidth(), boolean->value()));
  }

  if (auto *floating = llvm::dyn_cast<FloatLiteralExpr>(expr))
    return mlir::FloatAttr::get(elementType, floating->value());

  llvm_unreachable("expected constant tensor element");
}

void MLIRGenImpl::collectTensorDataAttrs(
    const TensorLiteralExpr *expr, mlir::Type elementType,
    llvm::SmallVectorImpl<mlir::Attribute> &attrs) {
  for (auto &element : expr->getElements()) {
    auto *childExpr = element.get();
    if (auto *nestedTensor = llvm::dyn_cast<TensorLiteralExpr>(childExpr)) {
      collectTensorDataAttrs(nestedTensor, elementType, attrs);
      continue;
    }

    attrs.push_back(getTensorDataAttr(childExpr, elementType));
  }
}

auto MLIRGenImpl::castToType(mlir::Value value, mlir::Type type,
                             mlir::Location location) -> mlir::Value {
  if (!value || value.getType() == type)
    return value;

  return nullptr;
}

auto MLIRGenImpl::genIndexValue(const Expr *node) -> mlir::Value {
  if (auto *decimal = llvm::dyn_cast<DecimalLiteralExpr>(node))
    return mlir::arith::ConstantIndexOp::create(_builder, loc(node),
                                                decimal->value());

  auto value = gen(node);
  if (value.getType().isIndex())
    return value;

  if (!llvm::isa<mlir::IntegerType>(value.getType()))
    value = castToType(value, _builder.getI64Type(), loc(node));
  return mlir::arith::IndexCastOp::create(_builder, loc(node),
                                          _builder.getIndexType(), value);
}

auto MLIRGenImpl::genMemRefElementValue(const Expr *node,
                                        mlir::Type elementType) -> mlir::Value {
  if (auto *decimal = llvm::dyn_cast<DecimalLiteralExpr>(node)) {
    auto intType = llvm::cast<mlir::IntegerType>(elementType);
    return mlir::arith::ConstantIntOp::create(
        _builder, loc(node), decimal->value(), intType.getWidth());
  }

  if (auto *boolean = llvm::dyn_cast<BoolLiteralExpr>(node)) {
    auto intType = llvm::cast<mlir::IntegerType>(elementType);
    return mlir::arith::ConstantIntOp::create(
        _builder, loc(node), boolean->value(), intType.getWidth());
  }

  if (auto *floating = llvm::dyn_cast<FloatLiteralExpr>(node)) {
    return mlir::arith::ConstantFloatOp::create(
        _builder, loc(node), llvm::cast<mlir::FloatType>(elementType),
        floating->value());
  }

  if (auto *tensorAccess = llvm::dyn_cast<TensorAccessExpr>(node))
    return genMemRefLoadValue(tensorAccess);

  return castToType(gen(node), elementType, loc(node));
}

auto MLIRGenImpl::gen(const TensorLiteralExpr *expr) -> mlir::Value {
  auto memRefType = getMemRefType(expr);
  mlir::Value allocatedTensor =
      mlir::memref::AllocOp::create(_builder, loc(expr), memRefType);

  llvm::SmallVector<mlir::Value, 4> currentIndices;
  storeTensorElements(expr, allocatedTensor, memRefType.getElementType(),
                      currentIndices);

  return allocatedTensor;
}

auto MLIRGenImpl::genGlobalTensorLiteral(const TensorLiteralExpr *expr,
                                         std::string_view varName)
    -> mlir::Value {
  auto memRefType = getMemRefType(expr);
  llvm::SmallVector<mlir::Attribute, 8> attrs;
  collectTensorDataAttrs(expr, memRefType.getElementType(), attrs);

  auto tensorType = mlir::RankedTensorType::get(
      memRefType.getShape(), memRefType.getElementType());
  auto initialValue = mlir::DenseElementsAttr::get(tensorType, attrs);
  std::string symbolName = "__cherry_global_" + std::string(varName) + "_" +
                           std::to_string(_globalTensorCounter++);

  {
    mlir::OpBuilder::InsertionGuard guard(_builder);
    auto &body = module.getBodyRegion().front();
    auto insertPoint = body.begin();
    while (insertPoint != body.end() &&
           llvm::isa<mlir::memref::GlobalOp>(*insertPoint))
      ++insertPoint;
    _builder.setInsertionPoint(&body, insertPoint);
    mlir::memref::GlobalOp::create(
        _builder, loc(expr), symbolName, _builder.getStringAttr("private"),
        memRefType, initialValue, /*constant=*/true,
        /*alignment=*/mlir::IntegerAttr{});
  }

  return mlir::memref::GetGlobalOp::create(_builder, loc(expr), memRefType,
                                           symbolName);
}

void MLIRGenImpl::storeTensorElements(
    const TensorLiteralExpr *expr, mlir::Value memref, mlir::Type elementType,
    llvm::SmallVectorImpl<mlir::Value> &indices) {
  for (size_t i = 0; i < expr->getElements().size(); ++i) {
    mlir::Value indexVal =
        mlir::arith::ConstantIndexOp::create(_builder, loc(expr), i);
    indices.push_back(indexVal);

    auto *childExpr = expr->getElements()[i].get();
    if (auto *nestedTensor = llvm::dyn_cast<TensorLiteralExpr>(childExpr)) {
      storeTensorElements(nestedTensor, memref, elementType, indices);
    } else {
      mlir::Value val = genMemRefElementValue(childExpr, elementType);
      mlir::memref::StoreOp::create(_builder, loc(childExpr), val, memref,
                                    indices);
    }
    indices.pop_back();
  }
}

mlir::Value MLIRGenImpl::genMemRefLoadValue(const TensorAccessExpr *expr) {
  mlir::Value memref = getVariable(expr->getVarName());

  llvm::SmallVector<mlir::Value, 4> mlirIndices;
  for (auto &idxExpr : expr->getIndices())
    mlirIndices.push_back(genIndexValue(idxExpr.get()));

  return mlir::memref::LoadOp::create(_builder, loc(expr), memref,
                                      mlirIndices);
}

mlir::Value MLIRGenImpl::gen(const TensorAccessExpr *expr, bool isLValue) {
  if (isLValue)
    return nullptr;

  auto loaded = genMemRefLoadValue(expr);
  return castToType(loaded, getMLIRType(expr), loc(expr));
}

void MLIRGenImpl::genAssignment(const TensorAccessExpr *lhs, const Expr *rhs) {
  mlir::Value memref = getVariable(lhs->getVarName());
  auto memRefType = llvm::cast<mlir::MemRefType>(memref.getType());

  llvm::SmallVector<mlir::Value, 4> mlirIndices;
  for (auto &idxExpr : lhs->getIndices())
    mlirIndices.push_back(genIndexValue(idxExpr.get()));

  auto rhsValue = genMemRefElementValue(rhs, memRefType.getElementType());
  mlir::memref::StoreOp::create(_builder, loc(lhs), rhsValue, memref,
                                mlirIndices);
}

auto MLIRGenImpl::gen(const VariableStat *node) -> void {
  auto *varType = node->type();
  auto *tensorType = cherry::getTensorType(varType);
  auto *structType = cherry::getStructType(varType);
  auto varName = node->variable()->name();

  if (tensorType) {
    DBG("use Cherry variable tensor type `{0}`",
        formatType(tensorType));

    if (node->isConst()) {
      if (auto *tensorLiteral =
              llvm::dyn_cast<TensorLiteralExpr>(node->init().get())) {
        if (shouldPromoteConstTensorData(tensorLiteral)) {
          setVariable(varName, genGlobalTensorLiteral(tensorLiteral, varName));
          return;
        }
      }
    }

    setVariable(varName, gen(node->init().get()));
    return;
  }

  if (structType) {
    auto *structLiteral = llvm::dyn_cast<StructLiteralExpr>(node->init().get());
    if (structLiteral) {
      DBG("use Cherry variable struct literal `{0}`",
          formatType(structType));
      setVariable(varName, genStructLiteral(structLiteral, structType, nullptr));
      return;
    }

    DBG("use Cherry variable struct value `{0}`",
        formatType(structType));
    auto mlirType = getMLIRType(varType);
    auto alloca = createAlloca(mlirType, loc(node));
    setVariable(varName, alloca);

    auto initValue = gen(node->init().get());
    createStore(initValue, alloca, loc(node));
    return;
  }

  if (cherry::isUnitType(varType)) {
    setVariable(varName, nullptr);
    gen(node->init().get());
    return;
  }

  auto mlirType = getMLIRType(varType);
  auto alloca = createAlloca(mlirType, loc(node));
  setVariable(varName, alloca);

  auto initValue = gen(node->init().get());

  initValue = castToType(initValue, mlirType, loc(node));
  createStore(initValue, alloca, loc(node));
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
