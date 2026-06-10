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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/TypeSwitch.h"

#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using namespace mlir::cherry;
using namespace mlir::arith;
using namespace cherry;
using llvm::cast;
using llvm::failure;
using llvm::success;

#undef DEBUG_TYPE
#define DEBUG_TYPE "MLIRGen"

template <typename T>
using NameMap = std::map<std::string, T, std::less<>>;

struct VariableBinding {
  // Scalar, struct, and mutable tensor variables are address-bound. List
  // variables bind descriptor values; assignment replaces the binding.
  enum Kind {
    Address,
    Value,
    Unit,
  };

  static auto address(mlir::Value value) -> VariableBinding {
    return {Address, value};
  }

  static auto value(mlir::Value value) -> VariableBinding {
    return {Value, value};
  }

  static auto unit() -> VariableBinding {
    return {Unit, nullptr};
  }

  auto isAddress() const -> bool { return kind == Address; }

  auto isValue() const -> bool { return kind == Value; }

  Kind kind;
  mlir::Value mlirValue;
};

struct TensorDimSource {
  mlir::Value tensor;
  int64_t dimension;
};

auto isValueType(const Type *type) -> bool {
  return cherry::isTensorType(type) || cherry::isListType(type);
}

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
  ScopeStack<NameMap<VariableBinding>> _variableScopes;
  NameMap<mlir::func::FuncOp> _functionsByName;
  llvm::StringRef _fileNameIdentifier;
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
  auto genTensorBinaryNN(const CallExpr *node) -> mlir::Value;
  auto genMatscale(const CallExpr *node) -> mlir::Value;
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

  void setVariable(std::string_view name, VariableBinding binding) {
    if (_variableScopes.empty())
      enterVariableScope();
    setSymbol(_variableScopes.currentScope(), name, binding);
  }

  void setVariableAddress(std::string_view name, mlir::Value address) {
    setVariable(name, VariableBinding::address(address));
  }

  void setVariableValue(std::string_view name, mlir::Value value) {
    setVariable(name, VariableBinding::value(value));
  }

  void setUnitVariable(std::string_view name) {
    setVariable(name, VariableBinding::unit());
  }

  auto getVariableBinding(std::string_view name) -> VariableBinding * {
    return _variableScopes.lookup(name);
  }

  auto getVariableAddress(std::string_view name) -> mlir::Value {
    auto *binding = getVariableBinding(name);
    if (!binding)
      llvm_unreachable("unknown variable");
    if (!binding->isAddress())
      llvm_unreachable("variable is not address-bound");
    return binding->mlirValue;
  }

  auto getVariableValue(std::string_view name,
                        mlir::Location location) -> mlir::Value {
    auto *binding = getVariableBinding(name);
    if (!binding)
      llvm_unreachable("unknown variable");
    if (binding->isAddress()) {
      auto ptrType =
          llvm::cast<mlir::mulberry::PtrType>(binding->mlirValue.getType());
      return createLoad(binding->mlirValue, ptrType.getPointeeType(),
                        location);
    }
    if (!binding->isValue())
      llvm_unreachable("variable has no value");
    return binding->mlirValue;
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
  auto createTensorAlloc(mlir::mulberry::TensorType tensorType,
                         const std::vector<mlir::Value> &dynamicSizes,
                         mlir::Location location) -> mlir::Value;
  auto createTensorDim(mlir::Value tensor, int64_t dimension,
                       mlir::Location location) -> mlir::Value;
  auto createListSize(mlir::Value list, mlir::Location location) -> mlir::Value;
  auto literalDynamicSizes(const ArrayLiteralExpr *expr,
                           mlir::mulberry::TensorType tensorType)
      -> std::vector<mlir::Value>;
  auto sourceDynamicSizes(mlir::mulberry::TensorType tensorType,
                          const std::vector<TensorDimSource> &sources,
                          mlir::Location location) -> std::vector<mlir::Value>;
  auto getPtrType(mlir::Type pointeeType) const -> mlir::mulberry::PtrType;
  auto createAlloca(mlir::Type mlirType, mlir::Location location)
      -> mlir::Value;
  auto createLoad(mlir::Value ptr, mlir::Type type, mlir::Location location)
      -> mlir::Value;
  void createStore(mlir::Value value, mlir::Value ptr,
                   mlir::Location location);
  auto createStructFieldPtr(mlir::Value recordPtr, const StructField& field,
                            mlir::Location location) -> mlir::Value;
  auto gen(const ArrayLiteralExpr *expr) -> mlir::Value;
  auto genTensorLiteral(const ArrayLiteralExpr *expr,
                        mlir::mulberry::TensorType tensorType) -> mlir::Value;
  auto genListLiteral(const ArrayLiteralExpr *expr,
                      mlir::mulberry::ListType listType) -> mlir::Value;
  void storeTensorElements(const ArrayLiteralExpr *expr, mlir::Value tensor,
                           mlir::Type elementType,
                           std::vector<mlir::Value> &indices);
  mlir::Value gen(const IndexExpr *expr);
  void genAssignment(const IndexExpr *lhs, const Expr *rhs);

  auto genLValue(const Expr *node) -> mlir::Value;
  auto getStructField(const MemberExpr *memberExpr) const
      -> const StructField *;
  auto genIndexValue(const Expr *node) -> mlir::Value;
  auto genTensorElementValue(const Expr *node, mlir::Type elementType)
      -> mlir::Value;
  auto genTensorGet(const IndexExpr *expr, mlir::Value tensor) -> mlir::Value;
  auto genListGet(const IndexExpr *expr, mlir::Value list) -> mlir::Value;
  auto castToType(mlir::Value value, mlir::Type type, mlir::Location location)
      -> mlir::Value;
  auto getMLIRType(const Type *type) const -> mlir::Type;
  auto getMLIRType(const Expr *expr) const -> mlir::Type;
  auto getMemRefType(const Type *type) const -> mlir::MemRefType;
  auto getMemRefType(const Expr *expr) const -> mlir::MemRefType;
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
    if (isValueType(paramType)) {
      setVariableValue(varName, value);
      continue;
    }
    if (cherry::isUnitType(paramType)) {
      setUnitVariable(varName);
      continue;
    }

    auto alloca = createAlloca(getMLIRType(paramType), loc(node));
    setVariableAddress(varName, alloca);
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
  case Expr::Expr_ArrayLiteral:
    return gen(cast<ArrayLiteralExpr>(node));
  case Expr::Expr_Index:
    return gen(cast<IndexExpr>(node));
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
  setVariableAddress(node->variableName(), inductionPtr);

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
  if (name == nn::matadd || name == nn::matsub || name == nn::hadamard)
    return genTensorBinaryNN(node);
  if (name == nn::matscale)
    return genMatscale(node);
  if (name == nn::transpose)
    return genTranspose(node);
  if (name == nn::exp || name == nn::sigmoid || name == nn::sigmoidPrime)
    return genElementwiseNN(node);
  if (name == nn::argmax)
    return genArgmax(node);
  if (name == builtins::size)
    return genSize(node);
  if (name == builtins::boolToUInt64) {
    auto value = gen(node->expressions().front().get());
    return mlir::arith::ExtUIOp::create(_builder, loc(node),
                                        _builder.getI64Type(), value);
  }

  auto calleeOpIter = findFunction(node->name());
  if (calleeOpIter == _functionsByName.end()) {
    // TODO: placeholder for functions implemented after the caller
    ERR("callee {0} DOESN'T exist.", node->name());
    return nullptr;
  }
  auto calleeType = calleeOpIter->second.getFunctionType();

  llvm::SmallVector<mlir::Value, 4> operands;
  for (const auto &indexedExpr : llvm::enumerate(*node)) {
    auto *expr = indexedExpr.value().get();
    auto value = gen(expr);
    value = castToType(value, calleeType.getInput(indexedExpr.index()),
                       loc(expr));
    DBG("gen(CallExpr). value: {0}", value);
    operands.push_back(value);
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
  auto operand = gen(expr);
  return PrintOp::create(_builder, loc(node), operand);
}

auto MLIRGenImpl::genMatmul(const CallExpr *node) -> mlir::Value {
  auto &expressions = node->expressions();
  auto lhs = gen(expressions[0].get());
  auto rhs = gen(expressions[1].get());
  auto outType = llvm::cast<mlir::mulberry::TensorType>(getMLIRType(node));
  auto out = createTensorAlloc(
      outType, sourceDynamicSizes(outType, {{lhs, 0}, {rhs, 1}}, loc(node)),
      loc(node));
  mlir::cherry_nn::MatmulOp::create(_builder, loc(node), lhs, rhs, out);
  return out;
}

auto MLIRGenImpl::genTensorBinaryNN(const CallExpr *node) -> mlir::Value {
  auto &expressions = node->expressions();
  auto lhs = gen(expressions[0].get());
  auto rhs = gen(expressions[1].get());
  auto outType = llvm::cast<mlir::mulberry::TensorType>(getMLIRType(node));
  auto out = createTensorAlloc(
      outType, sourceDynamicSizes(outType, {{lhs, 0}, {lhs, 1}}, loc(node)),
      loc(node));
  if (node->name() == nn::matadd) {
    mlir::cherry_nn::MataddOp::create(_builder, loc(node), lhs, rhs, out);
    return out;
  }
  if (node->name() == nn::matsub) {
    mlir::cherry_nn::MatsubOp::create(_builder, loc(node), lhs, rhs, out);
    return out;
  }
  if (node->name() == nn::hadamard) {
    mlir::cherry_nn::HadamardOp::create(_builder, loc(node), lhs, rhs, out);
    return out;
  }

  llvm_unreachable("unexpected binary cherry_nn op");
  return out;
}

auto MLIRGenImpl::genMatscale(const CallExpr *node) -> mlir::Value {
  auto &expressions = node->expressions();
  auto input = gen(expressions[0].get());
  auto scale = gen(expressions[1].get());
  auto outType = llvm::cast<mlir::mulberry::TensorType>(getMLIRType(node));
  auto out = createTensorAlloc(
      outType, sourceDynamicSizes(outType, {{input, 0}, {input, 1}},
                                  loc(node)),
      loc(node));
  mlir::cherry_nn::MatscaleOp::create(_builder, loc(node), input, scale, out);
  return out;
}

auto MLIRGenImpl::genTranspose(const CallExpr *node) -> mlir::Value {
  auto &expressions = node->expressions();
  auto input = gen(expressions[0].get());
  auto outType = llvm::cast<mlir::mulberry::TensorType>(getMLIRType(node));
  auto out = createTensorAlloc(
      outType, sourceDynamicSizes(outType, {{input, 1}, {input, 0}},
                                  loc(node)),
      loc(node));
  mlir::cherry_nn::TransposeOp::create(_builder, loc(node), input, out);
  return out;
}

auto MLIRGenImpl::genElementwiseNN(const CallExpr *node) -> mlir::Value {
  auto &expressions = node->expressions();
  auto input = gen(expressions[0].get());
  auto outType = llvm::cast<mlir::mulberry::TensorType>(getMLIRType(node));
  auto out = createTensorAlloc(
      outType, sourceDynamicSizes(outType, {{input, 0}, {input, 1}},
                                  loc(node)),
      loc(node));

  if (node->name() == nn::exp) {
    mlir::cherry_nn::ExpOp::create(_builder, loc(node), input, out);
    return out;
  }
  if (node->name() == nn::sigmoid) {
    mlir::cherry_nn::SigmoidOp::create(_builder, loc(node), input, out);
    return out;
  }
  if (node->name() == nn::sigmoidPrime) {
    mlir::cherry_nn::SigmoidPrimeOp::create(_builder, loc(node), input, out);
    return out;
  }

  llvm_unreachable("unexpected elementwise cherry_nn op");
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
  if (cherry::isListType(expressions.front()->type()))
    return createListSize(gen(expressions.front().get()), loc(node));

  auto *tensorType = cherry::getTensorType(expressions.front()->type());
  if (!tensorType) {
    ERR("size() argument has no Cherry tensor/list type");
    return nullptr;
  }

  auto size = tensorType->shape().front();
  if (size < 0) {
    auto tensor = gen(expressions.front().get());
    auto dynamicSize = createTensorDim(tensor, 0, loc(node));
    return mlir::arith::IndexCastOp::create(_builder, loc(node),
                                            _builder.getI64Type(), dynamicSize);
  }

  DBG("size() static tensor size: {0}", size);
  return mlir::arith::ConstantIntOp::create(_builder, loc(node), size, 64);
}

auto MLIRGenImpl::gen(const VariableExpr *node) -> mlir::Value {
  if (cherry::isUnitType(node->type()))
    return nullptr;
  return getVariableValue(node->name(), loc(node));
}

auto MLIRGenImpl::gen(const MemberExpr *node) -> mlir::Value {
  auto *field = getStructField(node);
  if (!field) {
    ERR("struct member access has no Cherry field information");
    return nullptr;
  }

  if (node->base()->isLvalue()) {
    auto ptr = genLValue(node);
    return createLoad(ptr, getMLIRType(node), loc(node));
  }

  auto record = gen(node->base().get());
  return mlir::mulberry::RecordExtractOp::create(
      _builder, loc(node), getMLIRType(node), record,
      std::string(field->name()));
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
    auto parentAddress = getVariableAddress(varExpr->name());
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
  llvm_unreachable("unexpected lvalue expression");

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
        if (isValueType(var->type())) {
          rhs = castToType(rhs, getMLIRType(var), loc(node));
          auto *binding = getVariableBinding(name);
          if (binding && binding->isAddress())
            createStore(rhs, binding->mlirValue, loc(node));
          else
            setVariableValue(name, rhs);
          return;
        }

        auto address = getVariableAddress(name);
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
      .Case<IndexExpr>([&](const auto *index) {
        genAssignment(index, node->rhs().get());
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

auto MLIRGenImpl::getPtrType(mlir::Type pointeeType) const
    -> mlir::mulberry::PtrType {
  return mlir::mulberry::PtrType::get(_builder.getContext(), pointeeType);
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

auto MLIRGenImpl::createTensorAlloc(
    mlir::mulberry::TensorType tensorType,
    const std::vector<mlir::Value> &dynamicSizes,
    mlir::Location location) -> mlir::Value {
  return mlir::mulberry::TensorAllocOp::create(_builder, location, tensorType,
                                               dynamicSizes);
}

auto MLIRGenImpl::createTensorDim(mlir::Value tensor, int64_t dimension,
                                  mlir::Location location) -> mlir::Value {
  auto index = mlir::arith::ConstantIndexOp::create(_builder, location,
                                                    dimension);
  return mlir::mulberry::TensorDimOp::create(_builder, location,
                                             _builder.getIndexType(), tensor,
                                             index);
}

auto MLIRGenImpl::createListSize(mlir::Value list,
                                 mlir::Location location) -> mlir::Value {
  return mlir::mulberry::ListSizeOp::create(_builder, location,
                                            _builder.getI64Type(), list);
}

auto MLIRGenImpl::literalDynamicSizes(
    const ArrayLiteralExpr *expr,
    mlir::mulberry::TensorType tensorType) -> std::vector<mlir::Value> {
  std::vector<mlir::Value> dynamicSizes;
  auto shape = tensorType.getShape();
  auto &inferredShape = expr->getInferredShape();

  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] >= 0)
      continue;
    dynamicSizes.push_back(mlir::arith::ConstantIndexOp::create(
        _builder, loc(expr), inferredShape[i]));
  }

  return dynamicSizes;
}

auto MLIRGenImpl::sourceDynamicSizes(
    mlir::mulberry::TensorType tensorType,
    const std::vector<TensorDimSource> &sources,
    mlir::Location location) -> std::vector<mlir::Value> {
  std::vector<mlir::Value> dynamicSizes;
  auto shape = tensorType.getShape();

  // Dynamic alloc operands are ordered by dynamic result dimensions, matching
  // MLIR shaped allocation conventions.
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] >= 0)
      continue;
    assert(i < sources.size());
    dynamicSizes.push_back(createTensorDim(sources[i].tensor,
                                           sources[i].dimension, location));
  }

  return dynamicSizes;
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
  auto *tensorType = cherry::getTensorType(type);
  if (!tensorType)
    return {};

  return _typeConverter.convertTensorStorage(*tensorType);
}

auto MLIRGenImpl::getMemRefType(const Expr *expr) const -> mlir::MemRefType {
  return getMemRefType(expr->type());
}

auto MLIRGenImpl::castToType(mlir::Value value, mlir::Type type,
                             mlir::Location location) -> mlir::Value {
  if (!value || value.getType() == type)
    return value;

  if (llvm::isa<mlir::mulberry::TensorType>(value.getType()) &&
      llvm::isa<mlir::mulberry::TensorType>(type))
    return mlir::mulberry::TensorCastOp::create(_builder, location, type,
                                                value);

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

auto MLIRGenImpl::genTensorElementValue(const Expr *node,
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

  if (auto *index = llvm::dyn_cast<IndexExpr>(node))
    return gen(index);

  return castToType(gen(node), elementType, loc(node));
}

auto MLIRGenImpl::gen(const ArrayLiteralExpr *expr) -> mlir::Value {
  auto type = getMLIRType(expr);
  if (auto tensorType = llvm::dyn_cast<mlir::mulberry::TensorType>(type))
    return genTensorLiteral(expr, tensorType);
  if (auto listType = llvm::dyn_cast<mlir::mulberry::ListType>(type))
    return genListLiteral(expr, listType);

  llvm_unreachable("array literal must lower to tensor or list");
}

auto MLIRGenImpl::genTensorLiteral(
    const ArrayLiteralExpr *expr,
    mlir::mulberry::TensorType tensorType) -> mlir::Value {
  auto allocatedTensor =
      createTensorAlloc(tensorType, literalDynamicSizes(expr, tensorType),
                        loc(expr));
  std::vector<mlir::Value> currentIndices;
  storeTensorElements(expr, allocatedTensor, tensorType.getElementType(),
                      currentIndices);

  return allocatedTensor;
}

auto MLIRGenImpl::genListLiteral(
    const ArrayLiteralExpr *expr,
    mlir::mulberry::ListType listType) -> mlir::Value {
  std::vector<mlir::Value> elements;
  for (auto &element : expr->getElements()) {
    auto value = gen(element.get());
    value = castToType(value, listType.getElementType(), loc(element.get()));
    elements.push_back(value);
  }

  return mlir::mulberry::ListCreateOp::create(_builder, loc(expr), listType,
                                              elements);
}

void MLIRGenImpl::storeTensorElements(
    const ArrayLiteralExpr *expr, mlir::Value tensor, mlir::Type elementType,
    std::vector<mlir::Value> &indices) {
  for (size_t i = 0; i < expr->getElements().size(); ++i) {
    mlir::Value indexVal =
        mlir::arith::ConstantIndexOp::create(_builder, loc(expr), i);
    indices.push_back(indexVal);

    auto *childExpr = expr->getElements()[i].get();
    if (auto *nestedTensor = llvm::dyn_cast<ArrayLiteralExpr>(childExpr)) {
      storeTensorElements(nestedTensor, tensor, elementType, indices);
    } else {
      mlir::Value val = genTensorElementValue(childExpr, elementType);
      mlir::mulberry::TensorStoreOp::create(_builder, loc(childExpr), val,
                                            tensor, mlir::ValueRange(indices));
    }
    indices.pop_back();
  }
}

auto MLIRGenImpl::genTensorGet(const IndexExpr *expr,
                               mlir::Value tensor) -> mlir::Value {
  llvm::SmallVector<mlir::Value, 4> mlirIndices;
  for (auto &idxExpr : expr->indices())
    mlirIndices.push_back(genIndexValue(idxExpr.get()));

  return mlir::mulberry::TensorLoadOp::create(
      _builder, loc(expr), getMLIRType(expr), tensor, mlirIndices);
}

auto MLIRGenImpl::genListGet(const IndexExpr *expr,
                             mlir::Value list) -> mlir::Value {
  auto index = genIndexValue(expr->indices().front().get());
  return mlir::mulberry::ListGetOp::create(_builder, loc(expr),
                                           getMLIRType(expr), list, index);
}

mlir::Value MLIRGenImpl::gen(const IndexExpr *expr) {
  auto source = gen(expr->base().get());
  if (llvm::isa<mlir::mulberry::ListType>(source.getType()))
    return genListGet(expr, source);

  auto loaded = genTensorGet(expr, source);
  return castToType(loaded, getMLIRType(expr), loc(expr));
}

void MLIRGenImpl::genAssignment(const IndexExpr *lhs, const Expr *rhs) {
  mlir::Value tensor = gen(lhs->base().get());
  auto tensorType =
      llvm::cast<mlir::mulberry::TensorType>(tensor.getType());

  llvm::SmallVector<mlir::Value, 4> mlirIndices;
  for (auto &idxExpr : lhs->indices())
    mlirIndices.push_back(genIndexValue(idxExpr.get()));

  auto rhsValue = genTensorElementValue(rhs, tensorType.getElementType());
  mlir::mulberry::TensorStoreOp::create(_builder, loc(lhs), rhsValue, tensor,
                                        mlirIndices);
}

auto MLIRGenImpl::gen(const VariableStat *node) -> void {
  auto *varType = node->type();
  auto *tensorType = cherry::getTensorType(varType);
  auto *listType = cherry::getListType(varType);
  auto *structType = cherry::getStructType(varType);
  auto varName = node->variable()->name();

  if (tensorType) {
    DBG("use Cherry variable tensor type `{0}`",
        formatType(tensorType));

    auto targetType =
        llvm::cast<mlir::mulberry::TensorType>(getMLIRType(varType));
    mlir::Value value;
    if (auto *literal =
            llvm::dyn_cast<ArrayLiteralExpr>(node->init().get())) {
      value = genTensorLiteral(literal, targetType);
    } else {
      value = castToType(gen(node->init().get()), targetType, loc(node));
    }

    if (node->isConst()) {
      setVariableValue(varName, value);
      return;
    }

    auto alloca = createAlloca(targetType, loc(node));
    setVariableAddress(varName, alloca);
    createStore(value, alloca, loc(node));
    return;
  }

  if (listType) {
    DBG("use Cherry variable list type `{0}`", formatType(listType));
    auto targetType =
        llvm::cast<mlir::mulberry::ListType>(getMLIRType(varType));
    auto value = gen(node->init().get());
    value = castToType(value, targetType, loc(node));
    setVariableValue(varName, value);
    return;
  }

  if (structType) {
    auto *structLiteral = llvm::dyn_cast<StructLiteralExpr>(node->init().get());
    if (structLiteral) {
      DBG("use Cherry variable struct literal `{0}`",
          formatType(structType));
      setVariableAddress(varName,
                         genStructLiteral(structLiteral, structType, nullptr));
      return;
    }

    DBG("use Cherry variable struct value `{0}`",
        formatType(structType));
    auto mlirType = getMLIRType(varType);
    auto alloca = createAlloca(mlirType, loc(node));
    setVariableAddress(varName, alloca);

    auto initValue = gen(node->init().get());
    createStore(initValue, alloca, loc(node));
    return;
  }

  if (cherry::isUnitType(varType)) {
    setUnitVariable(varName);
    gen(node->init().get());
    return;
  }

  auto mlirType = getMLIRType(varType);
  auto alloca = createAlloca(mlirType, loc(node));
  setVariableAddress(varName, alloca);

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
