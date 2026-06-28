//===--- MLIRGen.cpp - MLIR Generator -------------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/MLIRGen.h"
#include "cherry/AST/AST.h"
#include "cherry/Basic/CherryResult.h"
#include "cherry/Basic/Logging.h"
#include "cherry/Basic/ScopeStack.h"
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
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/TypeSwitch.h"

#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
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
  // Mutable variables are address-bound so assignment inside scf regions
  // writes through the variable slot instead of only replacing a local symbol.
  // Function parameters and const bindings may still be SSA value-bound.
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

struct WhileControl {
  mlir::Value breakFlag;
  mlir::Value continueFlag;
};

auto isParameterValueType(const Type *type) -> bool {
  return cherry::isTensorType(type) || cherry::isPtrType(type);
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
  std::vector<WhileControl> _whileControls;
  NameMap<mlir::func::FuncOp> _functionsByName;
  llvm::StringRef _fileNameIdentifier;
  MLIRTypeConverter _typeConverter{_builder};

  // Declarations
  auto declareFunction(const FunctionDecl *node) -> mlir::func::FuncOp;
  auto gen(const Prototype *node) -> mlir::func::FuncOp;
  auto gen(const FunctionDecl *node) -> mlir::func::FuncOp;
  auto gen(const StructDecl *node) -> void;

  // Expressions
  auto gen(const Expr *node) -> mlir::Value;
  auto gen(const UnitExpr *node) -> mlir::Value;
  auto gen(const BlockExpr *node) -> mlir::Value;
  auto gen(const IfExpr *node) -> mlir::Value;
  auto gen(const WhileExpr *node) -> mlir::Value;
  auto gen(const BreakExpr *node) -> mlir::Value;
  auto gen(const ContinueExpr *node) -> mlir::Value;
  auto gen(const ForExpr *node) -> mlir::Value;
  auto genDeclaredCall(std::string_view name, llvm::ArrayRef<const Expr *> args,
                       mlir::Location location) -> mlir::func::CallOp;
  auto genDeclaredCall(std::string_view name, mlir::ValueRange args,
                       mlir::Location location) -> mlir::func::CallOp;
  auto genNormalCall(const CallExpr *node) -> mlir::Value;
  auto gen(const CallExpr *node) -> mlir::Value;
  auto gen(const StructLiteralExpr *node) -> mlir::Value;
  auto gen(const VariableExpr *node) -> mlir::Value;
  auto gen(const MemberExpr *node) -> mlir::Value;
  auto gen(const DecimalLiteralExpr *node) -> mlir::Value;
  auto gen(const FloatLiteralExpr *node) -> mlir::Value;
  auto gen(const BoolLiteralExpr *node) -> mlir::Value;
  auto gen(const StringLiteralExpr *node) -> mlir::Value;
  auto gen(const CharLiteralExpr *node) -> mlir::Value;
  auto gen(const TypeLayoutExpr *node) -> mlir::Value;
  auto gen(const HeapAllocExpr *node) -> mlir::Value;
  auto gen(const DerefExpr *node) -> mlir::Value;
  auto gen(const TensorZerosExpr *node) -> mlir::Value;
  auto gen(const TensorPackExpr *node) -> mlir::Value;
  auto gen(const TensorViewExpr *node) -> mlir::Value;
  auto gen(const AssignExpr *node) -> mlir::Value;
  auto gen(const BinaryExpr *node) -> mlir::Value;

  // Statements
  auto gen(const Stat *node) -> void;
  auto gen(const VariableStat *node) -> void;
  auto gen(const ExprStat *node) -> void;
  auto declareLocalVariableSlot(const VariableStat *node) -> void;
  auto genStatements(const VectorUniquePtr<Stat> &statements) -> void;
  auto genGuardedWhileStatement(const Stat *node) -> void;
  auto genGuardedWhileExpression(const Expr *node) -> mlir::Value;
  auto currentWhileContinueAllowed() -> mlir::Value;

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

  auto getCurrentVariableBinding(std::string_view name) -> VariableBinding * {
    return _variableScopes.lookupCurrent(name);
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
  auto genStdlibListLiteral(const ArrayLiteralExpr *expr) -> mlir::Value;
  void storeTensorElements(const ArrayLiteralExpr *expr, mlir::Value tensor,
                           mlir::Type elementType,
                           std::vector<mlir::Value> &indices);
  auto zeroValue(mlir::Type type, mlir::Location location) -> mlir::Value;
  void zeroFill(mlir::Value tensor, mlir::Type elementType,
                const std::vector<int64_t> &shape, size_t depth,
                std::vector<mlir::Value> &indices, mlir::Location location);
  mlir::Value gen(const IndexExpr *expr);
  void genAssignment(const IndexExpr *lhs, const Expr *rhs);

  auto genLValue(const Expr *node) -> mlir::Value;
  auto genRecordPtrForMember(const MemberExpr *memberExpr) -> mlir::Value;
  auto getStructField(const MemberExpr *memberExpr) const
      -> const StructField *;
  auto genIndexValue(const Expr *node) -> mlir::Value;
  auto genPtrIndex(const IndexExpr *expr, mlir::Value ptr) -> mlir::Value;
  auto genStdlibListGet(const IndexExpr *expr) -> mlir::Value;
  auto genAddressableValue(const Expr *expr, mlir::Type valueType)
      -> mlir::Value;
  auto genArgumentValue(const Expr *expr, mlir::Type parameterType)
      -> mlir::Value;
  auto genTensorElementValue(const Expr *node, mlir::Type elementType)
      -> mlir::Value;
  auto genTensorGet(const IndexExpr *expr, mlir::Value tensor) -> mlir::Value;
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
    if (auto *structDecl = llvm::dyn_cast<StructDecl>(decl.get()))
      gen(structDecl);
  }

  // Sema appends instantiated generic functions after their call sites.  Declare
  // all functions first so call lowering can always use the callee signature
  // instead of manufacturing a loosely typed func.call.
  for (auto &decl : node) {
    auto *function = llvm::dyn_cast<FunctionDecl>(decl.get());
    if (function && !function->proto()->isGeneric()) {
      module.push_back(declareFunction(function));
      continue;
    }

    auto *structDecl = llvm::dyn_cast<StructDecl>(decl.get());
    if (!structDecl)
      continue;
    for (auto &method : structDecl->methods()) {
      if (!method->proto()->isGeneric())
        module.push_back(declareFunction(method.get()));
    }
  }

  for (auto &decl : node) {
    auto *function = llvm::dyn_cast<FunctionDecl>(decl.get());
    if (function && !function->proto()->isGeneric() && !function->isExtern()) {
      gen(function);
      continue;
    }

    auto *structDecl = llvm::dyn_cast<StructDecl>(decl.get());
    if (!structDecl)
      continue;
    for (auto &method : structDecl->methods()) {
      if (!method->proto()->isGeneric() && !method->isExtern())
        gen(method.get());
    }
  }

  if (failed(mlir::verify(module))) {
    module.emitError("module verification error");
    return failure();
  }

  return success();
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

  setFunction(funcName, func);

  DBG("funcName: {0}", funcName);

  return func;
}

auto MLIRGenImpl::declareFunction(const FunctionDecl *node)
    -> mlir::func::FuncOp {
  auto func = gen(node->proto().get());
  if (node->isExtern()) {
    mlir::SymbolTable::setSymbolVisibility(
        func, mlir::SymbolTable::Visibility::Private);
  }
  return func;
}

auto MLIRGenImpl::gen(const FunctionDecl *node) -> mlir::func::FuncOp {
  resetVariableScopes();
  auto funcIter = findFunction(node->proto()->id()->name());
  if (funcIter == _functionsByName.end()) {
    ERR("function `{0}` was not declared before body generation",
        node->proto()->id()->name());
    return {};
  }
  auto func = funcIter->second;
  if (node->isExtern()) {
    return func;
  }

  auto &entryBlock = *func.addEntryBlock();
  _builder.setInsertionPointToStart(&entryBlock);
  for (const auto &varValue : llvm::zip(node->proto()->parameters(),
                                        entryBlock.getArguments())) {
    auto &var = std::get<0>(varValue);
    auto varName = var->variable()->name();
    auto value = std::get<1>(varValue);
    auto *paramType = var->type();
    if (isParameterValueType(paramType)) {
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
  case Expr::Expr_StringLiteral:
    return gen(cast<StringLiteralExpr>(node));
  case Expr::Expr_CharLiteral:
    return gen(cast<CharLiteralExpr>(node));
  case Expr::Expr_TypeLayout:
    return gen(cast<TypeLayoutExpr>(node));
  case Expr::Expr_HeapAlloc:
    return gen(cast<HeapAllocExpr>(node));
  case Expr::Expr_Deref:
    return gen(cast<DerefExpr>(node));
  case Expr::Expr_TensorZeros:
    return gen(cast<TensorZerosExpr>(node));
  case Expr::Expr_TensorPack:
    return gen(cast<TensorPackExpr>(node));
  case Expr::Expr_TensorView:
    return gen(cast<TensorViewExpr>(node));
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
  case Expr::Expr_Break:
    return gen(cast<BreakExpr>(node));
  case Expr::Expr_Continue:
    return gen(cast<ContinueExpr>(node));
  case Expr::Expr_For:
    return gen(cast<ForExpr>(node));
  default:
    llvm_unreachable("Unexpected expression");
  }
}

auto MLIRGenImpl::gen(const UnitExpr *node) -> mlir::Value { return nullptr; }

auto MLIRGenImpl::gen(const BlockExpr *node) -> mlir::Value {
  enterVariableScope();
  genStatements(node->statements());
  auto *expression = node->expression().get();
  mlir::Value value = nullptr;
  if (cherry::isUnitType(node->type())) {
    if (!_whileControls.empty())
      genGuardedWhileExpression(expression);
    else
      gen(expression);
  } else {
    value = gen(expression);
  }
  leaveVariableScope();
  return value;
}

auto MLIRGenImpl::gen(const IfExpr *node) -> mlir::Value {
  DBG("IfExpr Cherry type: {0}, then type: {1}, else type: {2}",
      formatType(node->type()),
      formatType(node->thenBlock()->type()),
      node->hasElseBlock() ? formatType(node->elseBlock()->type()) : "none");
  auto cond = gen(node->conditionExpr().get());

  if (cherry::isUnitType(node->type())) {
    auto ifOp = mlir::scf::IfOp::create(_builder, loc(node), cond,
                                        node->hasElseBlock());
    _builder.setInsertionPointToStart(ifOp.thenBlock());
    gen(node->thenBlock().get());

    if (node->hasElseBlock()) {
      _builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
      gen(node->elseBlock().get());
    }
    _builder.setInsertionPointAfter(ifOp);
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
  auto location = loc(node);
  auto flagType = _builder.getI1Type();
  auto falseValue =
      mlir::arith::ConstantIntOp::create(_builder, location, 0, 1);
  auto trueValue =
      mlir::arith::ConstantIntOp::create(_builder, location, 1, 1);
  auto breakFlag = createAlloca(flagType, location);
  auto continueFlag = createAlloca(flagType, location);
  createStore(falseValue, breakFlag, location);
  createStore(falseValue, continueFlag, location);

  mlir::scf::WhileOp::create(
      _builder, location, mlir::TypeRange{}, mlir::ValueRange{},
      [&](mlir::OpBuilder &builder, mlir::Location location,
          mlir::ValueRange args) {
        createStore(falseValue, continueFlag, location);
        auto cond = gen(node->conditionExpr().get());
        auto shouldBreak = createLoad(breakFlag, flagType, location);
        auto notBreak = mlir::arith::XOrIOp::create(builder, location,
                                                    shouldBreak, trueValue);
        auto loopCond =
            mlir::arith::AndIOp::create(builder, location, cond, notBreak);
        mlir::scf::ConditionOp::create(builder, location, loopCond,
                                       mlir::ValueRange{});
      },
      [&](mlir::OpBuilder &builder, mlir::Location location,
          mlir::ValueRange args) {
        _whileControls.push_back({breakFlag, continueFlag});
        gen(bodyBlock);
        _whileControls.pop_back();
        mlir::scf::YieldOp::create(builder, location);
      });

  return nullptr;
}

auto MLIRGenImpl::gen(const BreakExpr *node) -> mlir::Value {
  auto control = _whileControls.back();
  auto trueValue =
      mlir::arith::ConstantIntOp::create(_builder, loc(node), 1, 1);
  createStore(trueValue, control.breakFlag, loc(node));
  return nullptr;
}

auto MLIRGenImpl::gen(const ContinueExpr *node) -> mlir::Value {
  auto control = _whileControls.back();
  auto trueValue =
      mlir::arith::ConstantIntOp::create(_builder, loc(node), 1, 1);
  createStore(trueValue, control.continueFlag, loc(node));
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
  DBG("gen(CallExpr). functionName: {0}", node->name());
  return genNormalCall(node);
}

auto MLIRGenImpl::genDeclaredCall(std::string_view name,
                                  mlir::ValueRange args,
                                  mlir::Location location)
    -> mlir::func::CallOp {
  auto calleeOpIter = findFunction(name);
  if (calleeOpIter == _functionsByName.end()) {
    ERR("call `{0}` has no declared callee", name);
    return {};
  }
  auto calleeType = calleeOpIter->second.getFunctionType();

  llvm::SmallVector<mlir::Value, 4> operands;
  for (const auto &indexedArg : llvm::enumerate(args)) {
    auto value = castToType(indexedArg.value(),
                            calleeType.getInput(indexedArg.index()), location);
    DBG("genDeclaredCall. value: {0}", value);
    operands.push_back(value);
  }

  auto resultTypes = calleeType.getResults();
  return mlir::func::CallOp::create(_builder, location, name,
                                    resultTypes, operands);
}

auto MLIRGenImpl::genDeclaredCall(std::string_view name,
                                  llvm::ArrayRef<const Expr *> args,
                                  mlir::Location location)
    -> mlir::func::CallOp {
  auto calleeOpIter = findFunction(name);
  if (calleeOpIter == _functionsByName.end()) {
    ERR("call `{0}` has no declared callee", name);
    return {};
  }
  auto calleeType = calleeOpIter->second.getFunctionType();

  llvm::SmallVector<mlir::Value, 4> operands;
  for (const auto &indexedArg : llvm::enumerate(args)) {
    auto value =
        genArgumentValue(indexedArg.value(),
                         calleeType.getInput(indexedArg.index()));
    DBG("genDeclaredCall. value: {0}", value);
    operands.push_back(value);
  }

  auto resultTypes = calleeType.getResults();
  return mlir::func::CallOp::create(_builder, location, name,
                                    resultTypes, operands);
}

auto MLIRGenImpl::genNormalCall(const CallExpr *node) -> mlir::Value {
  llvm::SmallVector<const Expr *, 4> args;
  for (auto &expr : node->expressions())
    args.push_back(expr.get());
  auto callOp = genDeclaredCall(node->name(), args, loc(node));

  return cherry::isUnitType(node->type()) ? nullptr : callOp.getResult(0);
}

auto MLIRGenImpl::gen(const StructLiteralExpr *node) -> mlir::Value {
  auto *structType = node->structType();
  if (!structType) {
    ERR("struct literal has no Cherry struct type");
    return nullptr;
  }

  DBG("use Cherry struct literal `{0}`",
      formatType(structType));
  auto ptr = genStructLiteral(node, structType, nullptr);
  return createLoad(ptr, getMLIRType(structType), loc(node));
}

auto MLIRGenImpl::gen(const TensorZerosExpr *node) -> mlir::Value {
  auto tensorType = llvm::cast<mlir::mulberry::TensorType>(getMLIRType(node));
  auto tensor = createTensorAlloc(tensorType, {}, loc(node));
  std::vector<mlir::Value> indices;
  zeroFill(tensor, tensorType.getElementType(), tensorType.getShape(), 0,
           indices, loc(node));
  return tensor;
}

auto MLIRGenImpl::gen(const TensorPackExpr *node) -> mlir::Value {
  auto tensor = gen(node->tensor().get());
  auto resultType = llvm::cast<mlir::mulberry::RecordType>(
      getMLIRType(node));
  return mlir::mulberry::TensorPackOp::create(_builder, loc(node),
                                              resultType, tensor);
}

auto MLIRGenImpl::gen(const TensorViewExpr *node) -> mlir::Value {
  auto tensorRecord = gen(node->tensorRecord().get());
  auto resultType = llvm::cast<mlir::mulberry::TensorType>(
      getMLIRType(node));
  return mlir::mulberry::TensorViewOp::create(_builder, loc(node),
                                              resultType, tensorRecord);
}

auto MLIRGenImpl::gen(const VariableExpr *node) -> mlir::Value {
  if (cherry::isUnitType(node->type()))
    return nullptr;
  return getVariableValue(node->name(), loc(node));
}

auto MLIRGenImpl::gen(const DerefExpr *node) -> mlir::Value {
  auto ptr = gen(node->pointer().get());
  return createLoad(ptr, getMLIRType(node), loc(node));
}

auto MLIRGenImpl::gen(const MemberExpr *node) -> mlir::Value {
  auto *field = getStructField(node);
  if (!field) {
    ERR("struct member access has no Cherry field information");
    return nullptr;
  }

  if (node->isLvalue()) {
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

auto MLIRGenImpl::gen(const CharLiteralExpr *node) -> mlir::Value {
  return mlir::arith::ConstantIntOp::create(_builder, loc(node),
                                            node->value(), 8);
}

auto MLIRGenImpl::gen(const StringLiteralExpr *node) -> mlir::Value {
  auto location = loc(node);
  auto storageType = llvm::cast<mlir::mulberry::RecordType>(
      getMLIRType(node));

  // Materialize the literal bytes as a normal heap buffer, then wrap that
  // buffer in the stdlib String value. String is a plain record wrapper around
  // `{length, data}`, so there is no second heap object around the byte buffer.
  auto bytes = node->value();
  auto byteCount = mlir::arith::ConstantIndexOp::create(
      _builder, location, bytes.size() + 1);
  auto dataBuffer = mlir::mulberry::HeapAllocOp::create(
                        _builder, location, getPtrType(_builder.getI8Type()),
                        _builder.getI8Type(), byteCount)
                        .getResult();

  for (size_t index = 0; index < bytes.size(); ++index) {
    auto byteIndex = mlir::arith::ConstantIndexOp::create(
        _builder, location, static_cast<int64_t>(index));
    auto bytePtr = mlir::mulberry::PtrIndexOp::create(
        _builder, location, getPtrType(_builder.getI8Type()), dataBuffer,
        byteIndex);
    auto byteValue = mlir::arith::ConstantIntOp::create(
        _builder, location,
        static_cast<int64_t>(static_cast<unsigned char>(bytes[index])), 8);
    createStore(byteValue, bytePtr, location);
  }

  auto nulIndex = mlir::arith::ConstantIndexOp::create(
      _builder, location, static_cast<int64_t>(bytes.size()));
  auto nulPtr = mlir::mulberry::PtrIndexOp::create(
      _builder, location, getPtrType(_builder.getI8Type()), dataBuffer,
      nulIndex);
  auto nulValue = mlir::arith::ConstantIntOp::create(_builder, location, 0, 8);
  createStore(nulValue, nulPtr, location);

  auto storage = createAlloca(storageType, location);
  auto lengthPtr = mlir::mulberry::RecordGetFieldOp::create(
      _builder, location, getPtrType(_builder.getI64Type()),
      storage, "length");
  auto lengthValue = mlir::arith::ConstantIntOp::create(
      _builder, location, static_cast<int64_t>(bytes.size()), 64);
  createStore(lengthValue, lengthPtr, location);

  auto dataPtr = mlir::mulberry::RecordGetFieldOp::create(
      _builder, location,
      getPtrType(getPtrType(_builder.getI8Type())), storage, "data");
  createStore(dataBuffer, dataPtr, location);

  return createLoad(storage, storageType, location);
}

auto MLIRGenImpl::gen(const TypeLayoutExpr *node) -> mlir::Value {
  return mlir::arith::ConstantIntOp::create(_builder, loc(node),
                                            node->value(), 64);
}

auto MLIRGenImpl::gen(const HeapAllocExpr *node) -> mlir::Value {
  auto allocatedType = getMLIRType(node->allocatedType());
  auto resultType = llvm::cast<mlir::mulberry::PtrType>(getMLIRType(node));
  auto count = node->count()
                   ? genIndexValue(node->count().get())
                   : mlir::arith::ConstantIndexOp::create(_builder, loc(node),
                                                          1);
  return mlir::mulberry::HeapAllocOp::create(_builder, loc(node), resultType,
                                             allocatedType, count);
}

auto MLIRGenImpl::genLValue(const Expr *node) -> mlir::Value {
  if (auto varExpr = llvm::dyn_cast<VariableExpr>(node)) {
    DBG("varExpr->name(): {0}", varExpr->name());
    auto parentAddress = getVariableAddress(varExpr->name());
    return parentAddress;
  }

  if (auto *memberExpr = llvm::dyn_cast<MemberExpr>(node)) {
    mlir::Value basePtr = genRecordPtrForMember(memberExpr);

    if (auto *field = getStructField(memberExpr)) {
      return createStructFieldPtr(basePtr, *field, loc(node));
    }

    ERR("struct member access has no Cherry field information");
    return nullptr;
  }

  if (auto *derefExpr = llvm::dyn_cast<DerefExpr>(node))
    return gen(derefExpr->pointer().get());

  if (auto *indexExpr = llvm::dyn_cast<IndexExpr>(node)) {
    auto source = gen(indexExpr->base().get());
    if (llvm::isa<mlir::mulberry::PtrType>(source.getType()))
      return genPtrIndex(indexExpr, source);
  }

  ERR("unknown EXPR");
  llvm_unreachable("unexpected lvalue expression");

  return nullptr;
}

auto MLIRGenImpl::genRecordPtrForMember(const MemberExpr *memberExpr)
    -> mlir::Value {
  auto *base = memberExpr->base().get();
  if (auto *ptrType = cherry::getPtrType(base->type())) {
    if (cherry::getStructType(ptrType->pointeeType()))
      return gen(base);
  }

  return genLValue(base);
}

auto MLIRGenImpl::getStructField(const MemberExpr *memberExpr) const
    -> const StructField * {
  auto *base = memberExpr->base().get();
  auto *baseType = base->type();
  auto *ptrType = cherry::getPtrType(baseType);
  auto *structType = ptrType ? cherry::getStructType(ptrType->pointeeType())
                             : cherry::getStructType(baseType);
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
        if (cherry::isTensorType(var->type())) {
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
      .Case<DerefExpr>([&](const auto *deref) {
        mlir::Value lhsPtr = genLValue(deref);
        auto rhs = castToType(gen(node->rhs().get()),
                              getMLIRType(deref), loc(node));

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

auto MLIRGenImpl::genStatements(const VectorUniquePtr<Stat> &statements)
    -> void {
  for (auto &statement : statements) {
    if (_whileControls.empty())
      gen(statement.get());
    else
      genGuardedWhileStatement(statement.get());
  }
}

auto MLIRGenImpl::currentWhileContinueAllowed() -> mlir::Value {
  auto location = _builder.getUnknownLoc();
  auto trueValue =
      mlir::arith::ConstantIntOp::create(_builder, location, 1, 1);
  auto flagType = _builder.getI1Type();
  auto control = _whileControls.back();
  auto shouldBreak = createLoad(control.breakFlag, flagType, location);
  auto shouldContinue = createLoad(control.continueFlag, flagType, location);
  auto stopped = mlir::arith::OrIOp::create(
      _builder, location, shouldBreak, shouldContinue);
  return mlir::arith::XOrIOp::create(_builder, location, stopped, trueValue);
}

auto MLIRGenImpl::genGuardedWhileStatement(const Stat *node) -> void {
  if (auto *variable = llvm::dyn_cast<VariableStat>(node))
    declareLocalVariableSlot(variable);

  auto condition = currentWhileContinueAllowed();
  auto ifOp = mlir::scf::IfOp::create(_builder, loc(node), condition,
                                      /*withElseRegion=*/false);
  _builder.setInsertionPointToStart(ifOp.thenBlock());
  gen(node);
  _builder.setInsertionPointAfter(ifOp);
}

auto MLIRGenImpl::declareLocalVariableSlot(const VariableStat *node) -> void {
  auto *varType = node->type();
  if (cherry::isUnitType(varType)) {
    setUnitVariable(node->variable()->name());
    return;
  }

  if (auto *structType = cherry::getStructType(varType)) {
    auto *structLiteral = llvm::dyn_cast<StructLiteralExpr>(node->init().get());
    if (structLiteral) {
      auto ptr = genStructLiteral(structLiteral, structType, nullptr);
      setVariableAddress(node->variable()->name(), ptr);
      return;
    }
  }

  auto mlirType = getMLIRType(varType);
  auto alloca = createAlloca(mlirType, loc(node));
  setVariableAddress(node->variable()->name(), alloca);
}

auto MLIRGenImpl::genGuardedWhileExpression(const Expr *node) -> mlir::Value {
  auto condition = currentWhileContinueAllowed();
  auto ifOp = mlir::scf::IfOp::create(_builder, loc(node), condition,
                                      /*withElseRegion=*/false);
  _builder.setInsertionPointToStart(ifOp.thenBlock());
  gen(node);
  _builder.setInsertionPointAfter(ifOp);
  return nullptr;
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
  DBG("struct literal type: {0}", formatType(structType));

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

  return _typeConverter.convertTensorToMemRefType(*tensorType);
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

  if (llvm::isa<mlir::mulberry::PtrType>(value.getType()) &&
      llvm::isa<mlir::mulberry::PtrType>(type))
    return mlir::mulberry::PtrCastOp::create(_builder, location, type, value);

  auto sourceIntType = llvm::dyn_cast<mlir::IntegerType>(value.getType());
  auto targetIntType = llvm::dyn_cast<mlir::IntegerType>(type);
  if (sourceIntType && targetIntType) {
    // Cherry only has unsigned integer scalars today, so integer casts use
    // zero-extension and truncation instead of signed variants.
    if (sourceIntType.getWidth() < targetIntType.getWidth()) {
      DBG("castToType zero-extend integer {0} -> {1}", value.getType(), type);
      return mlir::arith::ExtUIOp::create(_builder, location, type, value);
    }
    DBG("castToType truncate integer {0} -> {1}", value.getType(), type);
    return mlir::arith::TruncIOp::create(_builder, location, type, value);
  }

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

auto MLIRGenImpl::genPtrIndex(const IndexExpr *expr, mlir::Value ptr)
    -> mlir::Value {
  auto ptrType = llvm::cast<mlir::mulberry::PtrType>(ptr.getType());
  auto index = genIndexValue(expr->indices().front().get());
  return mlir::mulberry::PtrIndexOp::create(_builder, loc(expr), ptrType, ptr,
                                            index);
}

auto MLIRGenImpl::genStdlibListGet(const IndexExpr *expr) -> mlir::Value {
  auto list = genAddressableValue(expr->base().get(),
                                  getMLIRType(expr->base().get()));
  auto index = castToType(gen(expr->indices().front().get()),
                          _builder.getI64Type(), loc(expr));
  auto call =
      genDeclaredCall(expr->getFunctionName(), mlir::ValueRange{list, index},
                      loc(expr));
  return call.getResult(0);
}

auto MLIRGenImpl::genAddressableValue(const Expr *expr,
                                      mlir::Type valueType) -> mlir::Value {
  if (auto *variable = llvm::dyn_cast<VariableExpr>(expr)) {
    auto *binding = getVariableBinding(variable->name());
    if (binding && binding->isAddress())
      return binding->mlirValue;
  } else if (expr->isLvalue()) {
    return genLValue(expr);
  }

  auto ptr = createAlloca(valueType, loc(expr));
  createStore(castToType(gen(expr), valueType, loc(expr)), ptr, loc(expr));
  return ptr;
}

auto MLIRGenImpl::genArgumentValue(const Expr *expr,
                                   mlir::Type parameterType) -> mlir::Value {
  auto exprType = getMLIRType(expr);
  if (exprType == parameterType)
    return gen(expr);

  if (auto ptrType = llvm::dyn_cast<mlir::mulberry::PtrType>(parameterType)) {
    auto pointeeType = ptrType.getPointeeType();
    if (exprType == pointeeType) {
      // Methods such as List<T>.push take Ptr<List<T>> receivers.  Passing the
      // lvalue address keeps receiver mutation visible to the caller while
      // normal struct values still use shallow C-style copies elsewhere.
      return genAddressableValue(expr, pointeeType);
    }
  }

  return castToType(gen(expr), parameterType, loc(expr));
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
  switch (expr->literalKind()) {
  case ArrayLiteralExpr::LiteralKind::Tensor: {
    auto tensorType =
        llvm::cast<mlir::mulberry::TensorType>(getMLIRType(expr));
    return genTensorLiteral(expr, tensorType);
  }
  case ArrayLiteralExpr::LiteralKind::StdlibList:
    return genStdlibListLiteral(expr);
  case ArrayLiteralExpr::LiteralKind::Unknown:
    break;
  }

  auto type = getMLIRType(expr);
  if (auto tensorType = llvm::dyn_cast<mlir::mulberry::TensorType>(type))
    return genTensorLiteral(expr, tensorType);

  llvm_unreachable("array literal must lower to tensor or stdlib List");
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

auto MLIRGenImpl::genStdlibListLiteral(const ArrayLiteralExpr *expr)
    -> mlir::Value {
  auto location = loc(expr);
  auto listType = getMLIRType(expr);
  auto elementType = getMLIRType(expr->stdlibListElementType());

  auto capacity = mlir::arith::ConstantIntOp::create(
      _builder, location, expr->getElements().size(), 64);
  auto list = genDeclaredCall(expr->withCapacityFunctionName(),
                              mlir::ValueRange{capacity}, location);
  auto listPtr = createAlloca(listType, location);
  createStore(list.getResult(0), listPtr, location);

  for (auto &element : expr->getElements()) {
    auto value = castToType(gen(element.get()), elementType,
                            loc(element.get()));
    genDeclaredCall(expr->pushFunctionName(), mlir::ValueRange{listPtr, value},
                    loc(element.get()));
  }

  return createLoad(listPtr, listType, location);
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

auto MLIRGenImpl::zeroValue(mlir::Type type,
                            mlir::Location location) -> mlir::Value {
  if (auto intType = llvm::dyn_cast<mlir::IntegerType>(type))
    return mlir::arith::ConstantIntOp::create(_builder, location, 0,
                                              intType.getWidth());

  if (auto floatType = llvm::dyn_cast<mlir::FloatType>(type)) {
    auto zero = llvm::APFloat::getZero(floatType.getFloatSemantics());
    return mlir::arith::ConstantFloatOp::create(_builder, location, floatType,
                                                zero);
  }

  ERR("zeros() does not support tensor element type `{0}`", type);
  return nullptr;
}

void MLIRGenImpl::zeroFill(mlir::Value tensor, mlir::Type elementType,
                           const std::vector<int64_t> &shape, size_t depth,
                           std::vector<mlir::Value> &indices,
                           mlir::Location location) {
  if (depth == shape.size()) {
    auto zero = zeroValue(elementType, location);
    mlir::mulberry::TensorStoreOp::create(_builder, location, zero, tensor,
                                          mlir::ValueRange(indices));
    return;
  }

  // Static zeros() is expanded to stores now; dynamic zeros() should become a
  // loop-based fill when the language needs dynamic Tensor allocation syntax.
  for (int64_t i = 0; i < shape[depth]; ++i) {
    auto index = mlir::arith::ConstantIndexOp::create(_builder, location, i);
    indices.push_back(index);
    zeroFill(tensor, elementType, shape, depth + 1, indices, location);
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

mlir::Value MLIRGenImpl::gen(const IndexExpr *expr) {
  if (expr->indexKind() == IndexExpr::IndexKind::StdlibList)
    return genStdlibListGet(expr);

  auto source = gen(expr->base().get());
  if (llvm::isa<mlir::mulberry::PtrType>(source.getType())) {
    auto ptr = genPtrIndex(expr, source);
    return createLoad(ptr, getMLIRType(expr), loc(expr));
  }

  auto loaded = genTensorGet(expr, source);
  return castToType(loaded, getMLIRType(expr), loc(expr));
}

void MLIRGenImpl::genAssignment(const IndexExpr *lhs, const Expr *rhs) {
  if (lhs->indexKind() == IndexExpr::IndexKind::StdlibList) {
    auto list = genAddressableValue(lhs->base().get(),
                                    getMLIRType(lhs->base().get()));
    auto index = castToType(gen(lhs->indices().front().get()),
                            _builder.getI64Type(), loc(lhs));
    auto rhsValue = castToType(gen(rhs), getMLIRType(lhs), loc(lhs));
    genDeclaredCall(lhs->setFunctionName(),
                    mlir::ValueRange{list, index, rhsValue}, loc(lhs));
    return;
  }

  mlir::Value source = gen(lhs->base().get());
  if (llvm::isa<mlir::mulberry::PtrType>(source.getType())) {
    auto ptr = genPtrIndex(lhs, source);
    auto rhsValue = castToType(gen(rhs), getMLIRType(lhs), loc(lhs));
    createStore(rhsValue, ptr, loc(lhs));
    return;
  }

  mlir::Value tensor = source;
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
  auto *ptrType = cherry::getPtrType(varType);
  auto *structType = cherry::getStructType(varType);
  auto varName = node->variable()->name();
  auto *predeclaredBinding = getCurrentVariableBinding(varName);

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

    if (predeclaredBinding && predeclaredBinding->isAddress()) {
      createStore(value, predeclaredBinding->mlirValue, loc(node));
      return;
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

  if (ptrType) {
    DBG("use Cherry variable ptr type `{0}`", formatType(ptrType));
    auto targetType = getMLIRType(varType);
    auto value = gen(node->init().get());
    value = castToType(value, targetType, loc(node));

    if (predeclaredBinding && predeclaredBinding->isAddress()) {
      createStore(value, predeclaredBinding->mlirValue, loc(node));
      return;
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

  if (structType) {
    auto *structLiteral = llvm::dyn_cast<StructLiteralExpr>(node->init().get());
    if (structLiteral) {
      if (predeclaredBinding && predeclaredBinding->isAddress()) {
        genStructLiteral(structLiteral, structType,
                         predeclaredBinding->mlirValue);
        return;
      }
      DBG("use Cherry variable struct literal `{0}`",
          formatType(structType));
      setVariableAddress(varName,
                         genStructLiteral(structLiteral, structType, nullptr));
      return;
    }

    DBG("use Cherry variable struct value `{0}`",
        formatType(structType));
    auto mlirType = getMLIRType(varType);
    if (predeclaredBinding && predeclaredBinding->isAddress()) {
      auto initValue = gen(node->init().get());
      createStore(initValue, predeclaredBinding->mlirValue, loc(node));
      return;
    }

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
  if (predeclaredBinding && predeclaredBinding->isAddress()) {
    auto initValue = gen(node->init().get());
    initValue = castToType(initValue, mlirType, loc(node));
    createStore(initValue, predeclaredBinding->mlirValue, loc(node));
    return;
  }

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
