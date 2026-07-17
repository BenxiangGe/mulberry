//===--- MLIRGen.cpp - MLIR Generator -------------------------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "mulberry/MLIRGen/MLIRGen.h"
#include "mulberry/AST/AST.h"
#include "mulberry/Basic/MulberryResult.h"
#include "mulberry/Basic/Logging.h"
#include "mulberry/Basic/ScopeStack.h"
#include "mulberry/MLIRGen/IR/MulberryOps.h"
#include "mulberry/MLIRGen/IR/MulberryTypes.h"
#include "mulberry/MLIRGen/TypeConverter.h"
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
namespace arith = mlir::arith;
namespace func = mlir::func;
namespace mulberry_core = mlir::mulberry_core;
namespace scf = mlir::scf;

using namespace mulberry;
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
    ObjectReference,
    Value,
    Unit,
  };

  static auto address(mlir::Value value) -> VariableBinding {
    return {Address, value};
  }

  static auto value(mlir::Value value) -> VariableBinding {
    return {Value, value};
  }

  static auto objectReference(mlir::Value value) -> VariableBinding {
    return {ObjectReference, value};
  }

  static auto unit() -> VariableBinding {
    return {Unit, nullptr};
  }

  auto isAddress() const -> bool { return kind == Address; }

  auto isObjectReference() const -> bool { return kind == ObjectReference; }

  auto isValue() const -> bool { return kind == Value; }

  Kind kind;
  mlir::Value mlirValue;
};

struct TensorLocal {
  mlir::Value referenceSlot;
  mlir::Value initializedFlag;
  bool escaped = false;
};

struct WhileControl {
  mlir::Value breakFlag;
  mlir::Value continueFlag;
};

struct FunctionReturnControl {
  const Type *sourceReturnType;
  mlir::Type returnType;
  mlir::Value returnedFlag;
  mlir::Value returnSlot;
};

struct DeclaredFunction {
  func::FuncOp operation;
  bool isExtern;
};

auto containsReturnStat(const BlockExpr *node) -> bool;

auto containsReturnStat(const Expr *node) -> bool {
  if (auto *block = llvm::dyn_cast<BlockExpr>(node))
    return containsReturnStat(block);
  if (auto *matchExpr = llvm::dyn_cast<MatchExpr>(node)) {
    for (auto &arm : matchExpr->arms())
      if (containsReturnStat(arm->bodyBlock().get()) ||
          containsReturnStat(arm->resultExpr().get()))
        return true;
  }
  return false;
}

auto containsReturnStat(const Stat *node) -> bool {
  if (llvm::dyn_cast<ReturnStat>(node))
    return true;
  if (auto *ifStat = llvm::dyn_cast<IfStat>(node)) {
    if (containsReturnStat(ifStat->thenBlock().get()))
      return true;
    return ifStat->hasElseBlock() &&
           containsReturnStat(ifStat->elseBlock().get());
  }
  if (auto *matchStat = llvm::dyn_cast<MatchStat>(node)) {
    for (auto &arm : matchStat->arms())
      if (containsReturnStat(arm->bodyBlock().get()))
        return true;
    return false;
  }
  if (auto *whileStat = llvm::dyn_cast<WhileStat>(node))
    return containsReturnStat(whileStat->bodyBlock().get());
  if (auto *forStat = llvm::dyn_cast<ForStat>(node))
    return containsReturnStat(forStat->bodyBlock().get());
  if (auto *exprStat = llvm::dyn_cast<ExprStat>(node))
    return containsReturnStat(exprStat->expression().get());
  return false;
}

auto containsReturnStat(const BlockExpr *node) -> bool {
  for (auto &statement : node->statements())
    if (containsReturnStat(statement.get()))
      return true;
  return false;
}

auto isParameterValueType(const Type *type) -> bool {
  return mulberry::isPtrType(type) || mulberry::getFunctionType(type);
}

auto isObjectReferenceType(const Type *type) -> bool {
  return mulberry::getStructType(type) || mulberry::getDataType(type) ||
         mulberry::getArrayType(type);
}

auto isTensorObjectType(const Type *type) -> bool {
  auto *structType = mulberry::getStructType(type);
  auto *origin = structType ? structType->origin() : nullptr;
  return origin && origin->aliasName() == "std.tensor.Tensor";
}

auto startsWith(std::string_view value, std::string_view prefix) -> bool {
  return value.size() >= prefix.size() &&
         value.substr(0, prefix.size()) == prefix;
}

auto isObjectReferenceParameter(const Type *type, bool isExtern) -> bool {
  return !isExtern && isObjectReferenceType(type);
}

auto isObjectReferenceReturn(const Type *type, bool isExtern) -> bool {
  return !isExtern && isObjectReferenceType(type);
}

class MLIRGenImpl {
public:
  MLIRGenImpl(const llvm::SourceMgr &sourceManager, mlir::MLIRContext &context)
      : _sourceManager{sourceManager}, _builder(&context),
        _fileNameIdentifier{
            _sourceManager.getMemoryBuffer(_sourceManager.getMainFileID())
                ->getBufferIdentifier()} {
    registerBuiltinHandlers();
  }

  auto gen(const Module &node) -> MulberryResult;

  mlir::ModuleOp module;

private:
  using BuiltinHandler = std::function<mlir::Value(const Expr *)>;

  const llvm::SourceMgr &_sourceManager;
  mlir::OpBuilder _builder;
  ScopeStack<NameMap<VariableBinding>> _variableScopes;
  ScopeStack<NameMap<TensorLocal>> _tensorLocalScopes;
  std::vector<WhileControl> _whileControls;
  std::vector<FunctionReturnControl> _functionReturnControls;
  NameMap<DeclaredFunction> _functionsByName;
  NameMap<BuiltinHandler> _builtinHandlers;
  llvm::StringRef _fileNameIdentifier;
  MLIRTypeConverter _typeConverter{_builder};

  // Declarations
  auto declareFunction(const FunctionDecl *node) -> func::FuncOp;
  auto gen(const Prototype *node, bool isExtern) -> func::FuncOp;
  auto gen(const FunctionDecl *node) -> func::FuncOp;
  auto gen(const StructDecl *node) -> void;

  // Expressions
  auto gen(const Expr *node) -> mlir::Value;
  auto gen(const UnitExpr *node) -> mlir::Value;
  auto gen(const BlockExpr *node) -> mlir::Value;
  auto gen(const LambdaExpr *node) -> mlir::Value;
  auto gen(const MatchExpr *node) -> mlir::Value;
  auto gen(const TryExpr *node) -> mlir::Value;
  auto genMatchExprArm(const MatchExpr *node, size_t armIndex,
                       mlir::Value value, mlir::Value tag,
                       mlir::Type resultType) -> mlir::Value;
  auto genMatchExprArmResult(const MatchExprArm *arm,
                             const DataType *dataType,
                             mlir::Value value) -> mlir::Value;
  auto genDeclaredCall(std::string_view name, llvm::ArrayRef<const Expr *> args,
                       mlir::Location location,
                       std::vector<mlir::Value>* disposableTensorArguments =
                           nullptr) -> func::CallOp;
  auto genDeclaredLoweredCall(std::string_view name, mlir::ValueRange args,
                              mlir::Location location)
      -> func::CallOp;
  auto genNormalCall(const CallExpr *node) -> mlir::Value;
  auto genIndirectCall(const CallExpr *node) -> mlir::Value;
  auto gen(const CallExpr *node) -> mlir::Value;
  auto gen(const DataConstructorExpr *node) -> mlir::Value;
  auto gen(const StructLiteralExpr *node) -> mlir::Value;
  auto gen(const VariableExpr *node) -> mlir::Value;
  auto gen(const MemberExpr *node) -> mlir::Value;
  auto gen(const DecimalLiteralExpr *node) -> mlir::Value;
  auto gen(const FloatLiteralExpr *node) -> mlir::Value;
  auto gen(const BoolLiteralExpr *node) -> mlir::Value;
  auto gen(const StringLiteralExpr *node) -> mlir::Value;
  auto gen(const InterpolatedStringExpr *node) -> mlir::Value;
  auto gen(const ObjectIdentityExpr *node) -> mlir::Value;
  auto genStringLiteral(const StringLiteralExpr *node,
                        mulberry_core::RecordType storageType)
      -> mlir::Value;
  auto gen(const CharLiteralExpr *node) -> mlir::Value;
  auto gen(const TypeLayoutExpr *node) -> mlir::Value;
  auto gen(const HeapAllocExpr *node) -> mlir::Value;
  auto gen(const AssignExpr *node) -> mlir::Value;
  auto gen(const BinaryExpr *node) -> mlir::Value;

  // Compiler builtins
  auto registerBuiltinHandlers() -> void;
  auto registerBuiltinHandler(std::string_view name, BuiltinHandler handler)
      -> void;
  auto lookupBuiltinHandler(std::string_view name) const
      -> const BuiltinHandler *;

  // Statements
  auto gen(const Stat *node) -> void;
  auto gen(const VariableStat *node) -> void;
  auto gen(const ExprStat *node) -> void;
  auto gen(const IfStat *node) -> void;
  auto gen(const MatchStat *node) -> void;
  auto bindMatchPattern(const DataPattern *pattern,
                        const DataConstructor &constructor,
                        mlir::Value value) -> void;
  auto gen(const WhileStat *node) -> void;
  auto gen(const ForStat *node) -> void;
  auto gen(const BreakStat *node) -> void;
  auto gen(const ContinueStat *node) -> void;
  auto gen(const ReturnStat *node) -> void;
  auto declareLocalVariableSlot(const VariableStat *node) -> void;
  auto genStatements(const VectorUniquePtr<Stat> &statements) -> void;
  auto genGuardedStatement(const Stat *node) -> void;
  auto currentExecutionAllowed() -> mlir::Value;
  auto currentFunctionReturnAllowed(mlir::Location location) -> mlir::Value;
  auto createTrueValue(mlir::Location location) -> mlir::Value;
  auto createFalseValue(mlir::Location location) -> mlir::Value;

  template <typename T>
  void setSymbol(NameMap<T> &symbols, std::string_view name, T value) {
    symbols[std::string(name)] = value;
  }

  void resetVariableScopes() {
    _variableScopes.reset();
    _tensorLocalScopes.reset();
    enterVariableScope();
  }

  void enterVariableScope() {
    _variableScopes.enterScope();
    _tensorLocalScopes.enterScope();
  }

  void leaveVariableScope(mlir::Location location) {
    disposeCurrentTensorLocals(location);
    _tensorLocalScopes.leaveScope();
    _variableScopes.leaveScope();
  }

  void setVariable(std::string_view name, VariableBinding binding) {
    if (_variableScopes.empty())
      enterVariableScope();
    setSymbol(_variableScopes.currentScope(), name, binding);
  }

  void setVariableAddress(std::string_view name, mlir::Value address) {
    setVariable(name, VariableBinding::address(address));
  }

  void setVariableObjectReference(std::string_view name,
                                  mlir::Value referenceSlot) {
    setVariable(name, VariableBinding::objectReference(referenceSlot));
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

  auto getTensorLocal(std::string_view name) -> TensorLocal * {
    return _tensorLocalScopes.lookup(name);
  }

  void registerTensorLocal(std::string_view name, mlir::Value referenceSlot,
                           mlir::Value initializedFlag) {
    setSymbol(_tensorLocalScopes.currentScope(), name,
              TensorLocal{referenceSlot, initializedFlag});
  }

  void disposeCurrentTensorLocals(mlir::Location location);
  void disposeTensorLocal(std::string_view name, const TensorLocal& local,
                          mlir::Location location);
  void markTensorReferencesEscaped(const Expr *expr);
  auto isFreshTensorExpression(const Expr *expr) const -> bool;
  auto tensorCallDoesNotRetain(std::string_view name) const -> bool;

  auto getVariableAddress(std::string_view name,
                          mlir::Location location) -> mlir::Value {
    auto *binding = getVariableBinding(name);
    if (!binding)
      llvm_unreachable("unknown variable");
    if (binding->isObjectReference()) {
      auto referenceSlotType =
          llvm::cast<mulberry_core::PtrType>(binding->mlirValue.getType());
      auto objectReferenceType = referenceSlotType.getPointeeType();
      return createLoad(binding->mlirValue, objectReferenceType, location);
    }
    if (!binding->isAddress())
      llvm_unreachable("variable is not address-bound");
    return binding->mlirValue;
  }

  auto getVariableValue(std::string_view name,
                        mlir::Location location) -> mlir::Value {
    auto *binding = getVariableBinding(name);
    if (!binding)
      llvm_unreachable("unknown variable");
    if (binding->isObjectReference()) {
      auto referenceSlotType =
          llvm::cast<mulberry_core::PtrType>(binding->mlirValue.getType());
      auto objectReferenceType = referenceSlotType.getPointeeType();
      return createLoad(binding->mlirValue, objectReferenceType, location);
    }
    if (binding->isAddress()) {
      auto ptrType =
          llvm::cast<mulberry_core::PtrType>(binding->mlirValue.getType());
      return createLoad(binding->mlirValue, ptrType.getPointeeType(),
                        location);
    }
    if (!binding->isValue())
      llvm_unreachable("variable has no value");
    return binding->mlirValue;
  }

  void setFunction(std::string_view name, func::FuncOp func,
                   bool isExtern) {
    setSymbol(_functionsByName, name, DeclaredFunction{func, isExtern});
  }

  auto findFunction(std::string_view name) {
    return _functionsByName.find(name);
  }

  auto genStructLiteral(const StructLiteralExpr *structLiteral,
                        const StructType *structType) -> mlir::Value;
  auto getPtrType(mlir::Type pointeeType) const -> mulberry_core::PtrType;
  auto createIndexConstant(int64_t value, mlir::Location location)
      -> mlir::Value;
  auto createUInt64Constant(int64_t value, mlir::Location location)
      -> mlir::Value;
  auto createAlloca(mlir::Type mlirType, mlir::Location location)
      -> mlir::Value;
  auto createHeapObject(mlir::Type mlirType, mlir::Location location)
      -> mlir::Value;
  auto createLoad(mlir::Value ptr, mlir::Type type, mlir::Location location)
      -> mlir::Value;
  void createStore(mlir::Value value, mlir::Value ptr,
                   mlir::Location location);
  auto createRecordFieldPtr(mlir::Value recordPtr,
                            mulberry_core::RecordType recordType,
                            std::string_view fieldName,
                            mlir::Location location) -> mlir::Value;
  auto loadRecordFieldValue(mlir::Value recordPtr,
                            mulberry_core::RecordType recordType,
                            std::string_view fieldName,
                            mlir::Location location) -> mlir::Value;
  void storeRecordFieldValue(mlir::Value recordPtr,
                             mulberry_core::RecordType recordType,
                             std::string_view fieldName, mlir::Value value,
                             mlir::Location location);
  auto createStructFieldPtr(mlir::Value recordPtr, const StructField& field,
                            mlir::Location location) -> mlir::Value;
  auto gen(const ArrayLiteralExpr *expr) -> mlir::Value;
  auto genArrayLiteral(const ArrayLiteralExpr *expr,
                       mulberry_core::RecordType arrayType)
      -> mlir::Value;
  auto genTensorDispose(const CallExpr *expr) -> mlir::Value;
  auto genTensorStorageAlloc(const CallExpr *expr) -> mlir::Value;
  void createTensorAssertAlive(mlir::Value tensor,
                               mlir::Location location);
  void storeArrayElements(const ArrayLiteralExpr *expr,
                          const ArrayType *arrayType,
                          mlir::Value dataPtr, mlir::Type elementType);
  mlir::Value gen(const IndexExpr *expr);
  void genAssignment(const IndexExpr *lhs, const Expr *rhs);

  auto genLValue(const Expr *node) -> mlir::Value;
  auto genRecordPtrForMember(const MemberExpr *memberExpr) -> mlir::Value;
  auto getStructField(const MemberExpr *memberExpr) const
      -> const StructField *;
  auto genIndexValue(const Expr *node) -> mlir::Value;
  auto genPtrIndex(const IndexExpr *expr, mlir::Value ptr) -> mlir::Value;
  auto genArrayElementPtr(const IndexExpr *expr) -> mlir::Value;
  auto genStdlibTensorElementPtr(const IndexExpr *expr) -> mlir::Value;
  auto genStdlibListGet(const IndexExpr *expr) -> mlir::Value;
  auto genAddressableValue(const Expr *expr, mlir::Type valueType)
      -> mlir::Value;
  auto genObjectReference(const Expr *expr) -> mlir::Value;
  auto genValueForStorage(const Expr *expr, const Type *type,
                          mlir::Location location) -> mlir::Value;
  auto genCallArgumentValue(const Expr *expr, mlir::Type parameterType,
                            bool isExtern) -> mlir::Value;
  auto loadObjectHeaderForExternArgument(const Expr *expr,
                                         mlir::Type parameterType)
      -> mlir::Value;
  auto boxExternObjectResult(mlir::Value value, const Type *type,
                             mlir::Location location) -> mlir::Value;
  auto loadObjectReferenceFromStorage(mlir::Value storagePtr,
                                      const Type *type,
                                      mlir::Location location) -> mlir::Value;
  auto loadSourceValueFromStorage(mlir::Value storagePtr, const Type *type,
                                  mlir::Location location) -> mlir::Value;
  auto castToType(mlir::Value value, mlir::Type type, mlir::Location location)
      -> mlir::Value;
  auto getParameterMLIRType(const Type *type, bool isExtern) const
      -> mlir::Type;
  auto getReturnMLIRType(const Type *type, bool isExtern) const -> mlir::Type;
  auto getLayoutMLIRType(const Type *type) const -> mlir::Type;
  auto getLayoutMLIRType(const Expr *expr) const -> mlir::Type;
  auto getSourceMLIRType(const Type *type) const -> mlir::Type;
  auto getSourceMLIRType(const Expr *expr) const -> mlir::Type;
  auto getStorageMLIRType(const Type *type) const -> mlir::Type;
  // Utility
  auto loc(const Node *node) -> mlir::Location {
    auto [line, col] = _sourceManager.getLineAndColumn(node->location());
    return mlir::FileLineColLoc::get(
        _builder.getStringAttr(_fileNameIdentifier), line, col);
  }

};

} // end namespace

auto MLIRGenImpl::registerBuiltinHandlers() -> void {
  registerBuiltinHandler(
      "std.tensor.__allocate",
      [this](const Expr *node) {
        return genTensorStorageAlloc(cast<CallExpr>(node));
      });
  registerBuiltinHandler(
      "std.tensor.__dispose",
      [this](const Expr *node) {
        return genTensorDispose(cast<CallExpr>(node));
      });
  registerBuiltinHandler(
      "std.core.toUInt8",
      [this](const Expr *node) {
        auto *call = cast<CallExpr>(node);
        auto value = gen(call->expressions().front().get());
        return arith::TruncIOp::create(
            _builder, loc(call), getSourceMLIRType(call), value);
      });
  registerBuiltinHandler(
      "std.core.toUInt64",
      [this](const Expr *node) {
        auto *call = cast<CallExpr>(node);
        auto value = gen(call->expressions().front().get());
        return arith::ExtUIOp::create(
            _builder, loc(call), getSourceMLIRType(call), value);
      });
  registerBuiltinHandler(
      "std.core.toFloat32",
      [this](const Expr *node) {
        auto *call = cast<CallExpr>(node);
        auto value = gen(call->expressions().front().get());
        return arith::UIToFPOp::create(
            _builder, loc(call), getSourceMLIRType(call), value);
      });
}

auto MLIRGenImpl::registerBuiltinHandler(std::string_view name,
                                         BuiltinHandler handler) -> void {
  auto [iter, inserted] =
      _builtinHandlers.try_emplace(std::string(name), std::move(handler));
  if (!inserted)
    llvm_unreachable("duplicate builtin MLIRGen handler");
  DBG("register builtin MLIRGen handler `{0}`", iter->first);
}

auto MLIRGenImpl::lookupBuiltinHandler(std::string_view name) const
    -> const BuiltinHandler * {
  auto iter = _builtinHandlers.find(name);
  if (iter == _builtinHandlers.end())
    return nullptr;
  return &iter->second;
}

auto MLIRGenImpl::gen(const Module &node) -> MulberryResult {
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

auto MLIRGenImpl::gen(const Prototype *node, bool isExtern)
    -> func::FuncOp {
  llvm::SmallVector<mlir::Type, 3> argTypes;
  for (auto &param : node->parameters()) {
    auto *paramType = param->type();
    argTypes.push_back(getParameterMLIRType(paramType, isExtern));
  }

  auto funcName = node->id()->name();
  auto *returnType = node->type();
  llvm::SmallVector<mlir::Type, 1> resultTypes;
  if (!mulberry::isUnitType(returnType))
    resultTypes.push_back(getReturnMLIRType(returnType, isExtern));
  auto funcType = _builder.getFunctionType(argTypes, resultTypes);
  mlir::OperationState state(loc(node), func::FuncOp::getOperationName());
  func::FuncOp::build(_builder, state, funcName, funcType);
  auto func = llvm::cast<func::FuncOp>(mlir::Operation::create(state));

  setFunction(funcName, func, isExtern);

  DBG("funcName: {0}", funcName);

  return func;
}

auto MLIRGenImpl::declareFunction(const FunctionDecl *node)
    -> func::FuncOp {
  auto func = gen(node->proto().get(), node->isExtern());
  if (node->isExtern()) {
    mlir::SymbolTable::setSymbolVisibility(
        func, mlir::SymbolTable::Visibility::Private);
  }
  return func;
}

auto MLIRGenImpl::gen(const FunctionDecl *node) -> func::FuncOp {
  resetVariableScopes();
  auto funcIter = findFunction(node->proto()->id()->name());
  if (funcIter == _functionsByName.end()) {
    ERR("function `{0}` was not declared before body generation",
        node->proto()->id()->name());
    return {};
  }
  auto func = funcIter->second.operation;
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
    if (isObjectReferenceParameter(paramType, node->isExtern())) {
      auto referenceSlot = createAlloca(value.getType(), loc(node));
      createStore(value, referenceSlot, loc(node));
      setVariableObjectReference(varName, referenceSlot);
      continue;
    }
    if (isParameterValueType(paramType)) {
      setVariableValue(varName, value);
      continue;
    }
    if (mulberry::isUnitType(paramType)) {
      setUnitVariable(varName);
      continue;
    }

    auto alloca = createAlloca(getSourceMLIRType(paramType), loc(node));
    setVariableAddress(varName, alloca);
    createStore(value, alloca, loc(node));
  }

  auto body = node->body().get();
  auto location = loc(node);
  auto *returnType = node->proto()->type();
  auto hasExplicitReturn = containsReturnStat(body);
  if (hasExplicitReturn) {
    auto flagType = _builder.getI1Type();
    auto returnedFlag = createAlloca(flagType, loc(node));
    createStore(createFalseValue(loc(node)), returnedFlag, loc(node));

    mlir::Value returnSlot = nullptr;
    mlir::Type returnMLIRType = nullptr;
    if (!mulberry::isUnitType(returnType)) {
      returnMLIRType = getReturnMLIRType(returnType, node->isExtern());
      returnSlot = createAlloca(returnMLIRType, loc(node));
    }

    _functionReturnControls.push_back(
        {returnType, returnMLIRType, returnedFlag, returnSlot});

    enterVariableScope();
    genStatements(body->statements());
    leaveVariableScope(location);

    auto returnControl = _functionReturnControls.back();
    _functionReturnControls.pop_back();
    if (!mulberry::isUnitType(returnType)) {
      auto value = createLoad(returnControl.returnSlot,
                              returnControl.returnType, location);
      llvm::SmallVector<mlir::Value, 1> returnValues{value};
      func::ReturnOp::create(_builder, location, returnValues);
    } else {
      func::ReturnOp::create(_builder, location);
    }
    return func;
  }

  gen(node->body().get());
  func::ReturnOp::create(_builder, location);

  return func;
}

auto MLIRGenImpl::gen(const StructDecl *node) -> void {
  if (auto *structType = mulberry::getStructType(node->id()->type())) {
    getLayoutMLIRType(structType);
    return;
  }

  ERR("struct `{0}` has no Mulberry type", node->id()->name());
}

auto MLIRGenImpl::gen(const Expr *node) -> mlir::Value {
  mlir::Value result;
  switch (node->getKind()) {
  case Expr::Expr_Unit:
    result = gen(cast<UnitExpr>(node));
    break;
  case Expr::Expr_Lambda:
    result = gen(cast<LambdaExpr>(node));
    break;
  case Expr::Expr_Match:
    result = gen(cast<MatchExpr>(node));
    break;
  case Expr::Expr_Try:
    result = gen(cast<TryExpr>(node));
    break;
  case Expr::Expr_DecimalLiteral:
    result = gen(cast<DecimalLiteralExpr>(node));
    break;
  case Expr::Expr_FloatLiteral:
    result = gen(cast<FloatLiteralExpr>(node));
    break;
  case Expr::Expr_BoolLiteral:
    result = gen(cast<BoolLiteralExpr>(node));
    break;
  case Expr::Expr_StringLiteral:
    result = gen(cast<StringLiteralExpr>(node));
    break;
  case Expr::Expr_InterpolatedString:
    result = gen(cast<InterpolatedStringExpr>(node));
    break;
  case Expr::Expr_ObjectIdentity:
    result = gen(cast<ObjectIdentityExpr>(node));
    break;
  case Expr::Expr_CharLiteral:
    result = gen(cast<CharLiteralExpr>(node));
    break;
  case Expr::Expr_TypeInfo:
    llvm_unreachable("typeInfo expression reached MLIRGen");
  case Expr::Expr_TypeLayout:
    result = gen(cast<TypeLayoutExpr>(node));
    break;
  case Expr::Expr_HeapAlloc:
    result = gen(cast<HeapAllocExpr>(node));
    break;
  case Expr::Expr_Call:
    result = gen(cast<CallExpr>(node));
    break;
  case Expr::Expr_DataConstructor:
    result = gen(cast<DataConstructorExpr>(node));
    break;
  case Expr::Expr_StructLiteral:
    result = gen(cast<StructLiteralExpr>(node));
    break;
  case Expr::Expr_Variable:
    result = gen(cast<VariableExpr>(node));
    break;
  case Expr::Expr_Member:
    result = gen(cast<MemberExpr>(node));
    break;
  case Expr::Expr_ArrayLiteral:
    result = gen(cast<ArrayLiteralExpr>(node));
    break;
  case Expr::Expr_Index:
    result = gen(cast<IndexExpr>(node));
    break;
  case Expr::Expr_Assign:
    result = gen(cast<AssignExpr>(node));
    break;
  case Expr::Expr_Binary:
    result = gen(cast<BinaryExpr>(node));
    break;
  case Expr::Expr_Block:
    result = gen(cast<BlockExpr>(node));
    break;
  }

  if (mulberry::isUnitType(node->type()))
    return result;

  auto expectedType = getSourceMLIRType(node);
  if (!result) {
    ERR("expression `{0}` did not generate a source value",
        formatType(node->type()));
    llvm_unreachable("non-Unit expression did not generate a value");
  }
  if (result.getType() != expectedType) {
    ERR("expression `{0}` generated `{1}`, expected source type `{2}`",
        formatType(node->type()), result.getType(), expectedType);
    llvm_unreachable("expression violated the source representation contract");
  }
  return result;
}

auto MLIRGenImpl::gen(const UnitExpr *node) -> mlir::Value { return nullptr; }

auto MLIRGenImpl::gen(const BlockExpr *node) -> mlir::Value {
  enterVariableScope();
  genStatements(node->statements());
  leaveVariableScope(loc(node));
  return nullptr;
}

auto MLIRGenImpl::gen(const LambdaExpr *node) -> mlir::Value {
  auto function = findFunction(node->functionName());
  if (function == _functionsByName.end())
    llvm_unreachable("lambda has no lifted function");
  if (function->second.isExtern)
    llvm_unreachable("lambda function cannot be extern");

  DBG("materialize lambda function value `{0}`", node->functionName());
  mlir::SymbolTable::setSymbolVisibility(
      function->second.operation, mlir::SymbolTable::Visibility::Private);
  auto symbol = mlir::FlatSymbolRefAttr::get(
      _builder.getContext(), std::string(node->functionName()));
  return func::ConstantOp::create(
      _builder, loc(node), function->second.operation.getFunctionType(),
      symbol);
}

auto MLIRGenImpl::genMatchExprArmResult(const MatchExprArm *arm,
                                        const DataType *dataType,
                                        mlir::Value value) -> mlir::Value {
  auto *pattern = arm->pattern().get();
  if (pattern->constructorIndex() >= dataType->constructors().size())
    llvm_unreachable("match pattern has no semantic constructor");
  auto &constructor =
      dataType->constructors()[pattern->constructorIndex()];

  enterVariableScope();
  bindMatchPattern(pattern, constructor, value);
  enterVariableScope();
  genStatements(arm->bodyBlock()->statements());
  if (isObjectReferenceType(arm->resultExpr()->type()))
    markTensorReferencesEscaped(arm->resultExpr().get());
  auto result = gen(arm->resultExpr().get());
  leaveVariableScope(loc(arm));
  leaveVariableScope(loc(arm));
  return result;
}

auto MLIRGenImpl::genMatchExprArm(const MatchExpr *node, size_t armIndex,
                                  mlir::Value value, mlir::Value tag,
                                  mlir::Type resultType) -> mlir::Value {
  auto *dataType = mulberry::getDataType(node->value()->type());
  if (!dataType)
    llvm_unreachable("match value has no semantic data type");

  auto &arm = node->arms()[armIndex];
  if (armIndex + 1 == node->arms().size())
    return genMatchExprArmResult(arm.get(), dataType, value);

  auto *pattern = arm->pattern().get();
  auto expectedTag = createUInt64Constant(
      pattern->constructorIndex(), loc(pattern));
  auto isMatch = arith::CmpIOp::create(
      _builder, loc(pattern), arith::CmpIPredicate::eq, tag, expectedTag);

  std::vector<mlir::Type> resultTypes;
  if (resultType)
    resultTypes.push_back(resultType);
  auto ifOp = scf::IfOp::create(
      _builder, loc(pattern), resultTypes, isMatch,
      /*withElseRegion=*/true);

  _builder.setInsertionPointToStart(ifOp.thenBlock());
  auto thenResult = genMatchExprArmResult(arm.get(), dataType, value);
  std::vector<mlir::Value> thenValues;
  if (thenResult)
    thenValues.push_back(thenResult);
  scf::YieldOp::create(_builder, loc(arm.get()), thenValues);

  _builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  auto elseResult =
      genMatchExprArm(node, armIndex + 1, value, tag, resultType);
  std::vector<mlir::Value> elseValues;
  if (elseResult)
    elseValues.push_back(elseResult);
  scf::YieldOp::create(_builder, loc(arm.get()), elseValues);

  _builder.setInsertionPointAfter(ifOp);
  if (!resultType)
    return nullptr;
  return ifOp.getResult(0);
}

auto MLIRGenImpl::gen(const MatchExpr *node) -> mlir::Value {
  if (node->arms().empty())
    llvm_unreachable("match expression has no arms");

  auto value = gen(node->value().get());
  mlir::Value tag;
  if (node->arms().size() > 1)
    tag = mulberry_core::DataTagOp::create(
        _builder, loc(node), _builder.getI64Type(), value).getTag();

  auto *dataType = mulberry::getDataType(node->value()->type());
  if (!dataType)
    llvm_unreachable("match value has no semantic data type");
  DBG("generate match expression for `{0}` with {1} arms",
      formatType(dataType), node->arms().size());

  auto resultType = mulberry::isUnitType(node->type())
                        ? mlir::Type{}
                        : getSourceMLIRType(node);
  return genMatchExprArm(node, 0, value, tag, resultType);
}

auto MLIRGenImpl::gen(const TryExpr *node) -> mlir::Value {
  auto *resultType = mulberry::getDataType(node->value()->type());
  if (!resultType || resultType->arguments().size() != 2 ||
      resultType->arguments()[1].kind() != ComptimeValue::Kind::Type)
    llvm_unreachable("try expression has no semantic Result type");

  auto input = gen(node->value().get());
  std::vector<mlir::Type> valueTypes;
  if (!mulberry::isUnitType(node->type()))
    valueTypes.push_back(getSourceMLIRType(node));
  auto errorType = getStorageMLIRType(resultType->arguments()[1].type());
  DBG("generate Result propagation from `{0}`", formatType(resultType));
  auto tryOp = mulberry_core::ResultTryOp::create(
      _builder, loc(node), valueTypes, input, errorType);
  if (valueTypes.empty())
    return nullptr;
  return tryOp.getResult(0);
}

auto MLIRGenImpl::gen(const IfStat *node) -> void {
  if (node->comptimeValue()) {
    DBG("IfStat comptime value={0}", *node->comptimeValue());
    if (*node->comptimeValue())
      gen(node->thenBlock().get());
    else if (node->hasElseBlock())
      gen(node->elseBlock().get());
    return;
  }

  DBG("IfStat");
  auto cond = gen(node->conditionExpr().get());

  auto ifOp = scf::IfOp::create(_builder, loc(node), cond,
                                node->hasElseBlock());

  _builder.setInsertionPointToStart(ifOp.thenBlock());
  gen(node->thenBlock().get());

  if (node->hasElseBlock()) {
    _builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    gen(node->elseBlock().get());
  }

  _builder.setInsertionPointAfter(ifOp);
}

auto MLIRGenImpl::gen(const MatchStat *node) -> void {
  auto location = loc(node);
  auto value = gen(node->value().get());
  auto tag = mulberry_core::DataTagOp::create(
      _builder, location, _builder.getI64Type(), value).getTag();
  auto *dataType = mulberry::getDataType(node->value()->type());
  if (!dataType)
    llvm_unreachable("match value has no semantic data type");

  DBG("match data value `{0}` with {1} arms", formatType(dataType),
      node->arms().size());
  for (auto &arm : node->arms()) {
    auto *pattern = arm->pattern().get();
    if (pattern->constructorIndex() >= dataType->constructors().size())
      llvm_unreachable("match pattern has no semantic constructor");
    auto &constructor =
        dataType->constructors()[pattern->constructorIndex()];

    auto expectedTag = createUInt64Constant(
        pattern->constructorIndex(), loc(pattern));
    auto isMatch = arith::CmpIOp::create(
        _builder, loc(pattern), arith::CmpIPredicate::eq, tag, expectedTag);
    auto ifOp = scf::IfOp::create(
        _builder, loc(pattern), isMatch, /*withElseRegion=*/false);
    _builder.setInsertionPointToStart(ifOp.thenBlock());

    enterVariableScope();
    bindMatchPattern(pattern, constructor, value);
    gen(arm->bodyBlock().get());
    leaveVariableScope(loc(arm.get()));

    _builder.setInsertionPointAfter(ifOp);
  }
}

auto MLIRGenImpl::bindMatchPattern(
    const DataPattern *pattern, const DataConstructor &constructor,
    mlir::Value value) -> void {
  std::vector<mlir::Type> payloadTypes;
  for (auto *payloadType : constructor.payloadTypes()) {
    if (!mulberry::isUnitType(payloadType))
      payloadTypes.push_back(getStorageMLIRType(payloadType));
  }
  auto unpack = mulberry_core::DataUnpackOp::create(
      _builder, loc(pattern), payloadTypes,
      std::string(pattern->constructorName()),
      static_cast<uint64_t>(pattern->constructorIndex()), value);

  size_t payloadIndex = 0;
  for (size_t i = 0; i < pattern->bindings().size(); ++i) {
    auto *binding = pattern->bindings()[i].get();
    auto *payloadType = constructor.payloadTypes()[i];
    if (mulberry::isUnitType(payloadType)) {
      setUnitVariable(binding->name());
      continue;
    }
    auto payload = unpack.getPayloads()[payloadIndex++];
    if (isObjectReferenceType(payloadType)) {
      auto referenceSlot = createAlloca(payload.getType(), loc(binding));
      createStore(payload, referenceSlot, loc(binding));
      setVariableObjectReference(binding->name(), referenceSlot);
    } else if (isParameterValueType(payloadType)) {
      setVariableValue(binding->name(), payload);
    } else {
      auto address = createAlloca(payload.getType(), loc(binding));
      createStore(payload, address, loc(binding));
      setVariableAddress(binding->name(), address);
    }
  }
}

auto MLIRGenImpl::gen(const WhileStat *node) -> void {
  auto bodyBlock = node->bodyBlock().get();
  auto location = loc(node);
  auto flagType = _builder.getI1Type();
  auto falseValue = arith::ConstantIntOp::create(_builder, location, 0, 1);
  auto trueValue = arith::ConstantIntOp::create(_builder, location, 1, 1);
  auto breakFlag = createAlloca(flagType, location);
  auto continueFlag = createAlloca(flagType, location);
  createStore(falseValue, breakFlag, location);
  createStore(falseValue, continueFlag, location);

  scf::WhileOp::create(
      _builder, location, mlir::TypeRange{}, mlir::ValueRange{},
      [&](mlir::OpBuilder &builder, mlir::Location location,
          mlir::ValueRange args) {
        createStore(falseValue, continueFlag, location);
        auto cond = gen(node->conditionExpr().get());
        auto shouldBreak = createLoad(breakFlag, flagType, location);
        auto stopped = shouldBreak;
        if (!_functionReturnControls.empty()) {
          auto returned = createLoad(_functionReturnControls.back().returnedFlag,
                                     flagType, location);
          stopped = arith::OrIOp::create(builder, location, stopped, returned);
        }
        auto notStopped =
            arith::XOrIOp::create(builder, location, stopped, trueValue);
        auto loopCond =
            arith::AndIOp::create(builder, location, cond, notStopped);
        scf::ConditionOp::create(builder, location, loopCond,
                                 mlir::ValueRange{});
      },
      [&](mlir::OpBuilder &builder, mlir::Location location,
          mlir::ValueRange args) {
        _whileControls.push_back({breakFlag, continueFlag});
        gen(bodyBlock);
        _whileControls.pop_back();
        scf::YieldOp::create(builder, location);
      });
}

auto MLIRGenImpl::gen(const BreakStat *node) -> void {
  auto control = _whileControls.back();
  auto trueValue = arith::ConstantIntOp::create(_builder, loc(node), 1, 1);
  createStore(trueValue, control.breakFlag, loc(node));
}

auto MLIRGenImpl::gen(const ContinueStat *node) -> void {
  auto control = _whileControls.back();
  auto trueValue = arith::ConstantIntOp::create(_builder, loc(node), 1, 1);
  createStore(trueValue, control.continueFlag, loc(node));
}

auto MLIRGenImpl::gen(const ReturnStat *node) -> void {
  auto &control = _functionReturnControls.back();
  if (node->hasExpression()) {
    auto *expression = node->expression().get();
    if (isObjectReferenceType(expression->type()))
      markTensorReferencesEscaped(expression);
    if (mulberry::isUnitType(control.sourceReturnType)) {
      gen(expression);
    } else {
      auto value = isObjectReferenceReturn(control.sourceReturnType,
                                           /*isExtern=*/false)
                       ? genObjectReference(expression)
                       : gen(expression);
      value = castToType(value, control.returnType, loc(node));
      createStore(value, control.returnSlot, loc(node));
    }
  }
  createStore(createTrueValue(loc(node)), control.returnedFlag, loc(node));
}

auto MLIRGenImpl::gen(const ForStat *node) -> void {
  auto forLocation = loc(node);
  auto inductionType = getSourceMLIRType(node->startExpr().get());
  auto inductionPtr = createAlloca(inductionType, forLocation);

  auto startValue =
      castToType(gen(node->startExpr().get()), inductionType, forLocation);
  auto endValue =
      castToType(gen(node->endExpr().get()), inductionType, forLocation);
  auto intType = llvm::cast<mlir::IntegerType>(inductionType);
  auto oneValue = arith::ConstantIntOp::create(
      _builder, forLocation, 1, intType.getWidth());

  enterVariableScope();
  setVariableAddress(node->variableName(), inductionPtr);

  auto bodyBlock = node->bodyBlock().get();
  scf::ForOp::create(
      _builder, forLocation, startValue, endValue, oneValue,
      mlir::ValueRange{},
      [&](mlir::OpBuilder &builder, mlir::Location location,
          mlir::Value inductionValue, mlir::ValueRange args) {
        createStore(inductionValue, inductionPtr, location);
        gen(bodyBlock);
        scf::YieldOp::create(builder, location);
      },
      /*unsignedCmp=*/true);
  leaveVariableScope(forLocation);
}

auto MLIRGenImpl::gen(const CallExpr *node) -> mlir::Value {
  DBG("gen(CallExpr). functionName: {0}", node->name());
  if (node->isIndirectCall())
    return genIndirectCall(node);
  if (auto *handler = lookupBuiltinHandler(node->name())) {
    DBG("dispatch builtin MLIRGen handler `{0}`", node->name());
    return (*handler)(node);
  }
  return genNormalCall(node);
}

auto MLIRGenImpl::genIndirectCall(const CallExpr *node) -> mlir::Value {
  for (auto &argument : node->expressions())
    if (isTensorObjectType(argument->type()))
      markTensorReferencesEscaped(argument.get());

  auto callee = getVariableValue(node->name(), loc(node));
  auto calleeType = llvm::cast<mlir::FunctionType>(callee.getType());
  llvm::SmallVector<mlir::Value, 4> operands;
  for (const auto &indexedArgument : llvm::enumerate(node->expressions())) {
    auto value = genCallArgumentValue(
        indexedArgument.value().get(),
        calleeType.getInput(indexedArgument.index()), /*isExtern=*/false);
    operands.push_back(value);
  }

  auto call = func::CallIndirectOp::create(_builder, loc(node), callee,
                                           operands);
  if (mulberry::isUnitType(node->type()))
    return nullptr;
  return call.getResult(0);
}

// Compiler-generated calls already carry their source ABI representation.
// Object value-ABI adaptation belongs exclusively to source extern calls.
auto MLIRGenImpl::genDeclaredLoweredCall(std::string_view name,
                                        mlir::ValueRange args,
                                        mlir::Location location)
    -> func::CallOp {
  auto calleeOpIter = findFunction(name);
  if (calleeOpIter == _functionsByName.end()) {
    ERR("call `{0}` has no declared callee", name);
    return {};
  }
  auto& callee = calleeOpIter->second;
  if (callee.isExtern)
    llvm_unreachable("lowered call cannot target an extern value ABI");
  auto calleeType = callee.operation.getFunctionType();

  llvm::SmallVector<mlir::Value, 4> operands;
  for (const auto &indexedArg : llvm::enumerate(args)) {
    auto value = castToType(indexedArg.value(),
                            calleeType.getInput(indexedArg.index()), location);
    DBG("genDeclaredLoweredCall. value: {0}", value);
    operands.push_back(value);
  }

  auto resultTypes = calleeType.getResults();
  return func::CallOp::create(_builder, location, name, resultTypes, operands);
}

auto MLIRGenImpl::genDeclaredCall(std::string_view name,
                                  llvm::ArrayRef<const Expr *> args,
                                  mlir::Location location,
                                  std::vector<mlir::Value>*
                                      disposableTensorArguments)
    -> func::CallOp {
  auto calleeOpIter = findFunction(name);
  if (calleeOpIter == _functionsByName.end()) {
    ERR("call `{0}` has no declared callee", name);
    return {};
  }
  auto& callee = calleeOpIter->second;
  auto calleeType = callee.operation.getFunctionType();

  llvm::SmallVector<mlir::Value, 4> operands;
  for (const auto &indexedArg : llvm::enumerate(args)) {
    auto *argument = indexedArg.value();
    auto parameterType = calleeType.getInput(indexedArg.index());
    mlir::Value value;
    // Extern Tensor arguments are borrowed only until the call returns, so a
    // fresh argument expression can be disposed immediately afterward.
    if (disposableTensorArguments && callee.isExtern &&
        isFreshTensorExpression(argument)) {
      auto objectType = getLayoutMLIRType(argument);
      if (parameterType != objectType)
        llvm_unreachable("extern Tensor parameter does not use value ABI");
      auto objectReference = genObjectReference(argument);
      createTensorAssertAlive(objectReference, loc(argument));
      value = createLoad(objectReference, parameterType, loc(argument));
      disposableTensorArguments->push_back(objectReference);
    } else {
      value = genCallArgumentValue(argument, parameterType, callee.isExtern);
    }
    DBG("genDeclaredCall. value: {0}", value);
    operands.push_back(value);
  }

  auto resultTypes = calleeType.getResults();
  return func::CallOp::create(_builder, location, name, resultTypes, operands);
}

auto MLIRGenImpl::genNormalCall(const CallExpr *node) -> mlir::Value {
  if (!tensorCallDoesNotRetain(node->name()))
    for (auto &expr : node->expressions())
      if (isTensorObjectType(expr->type()))
        markTensorReferencesEscaped(expr.get());

  llvm::SmallVector<const Expr *, 4> args;
  for (auto &expr : node->expressions())
    args.push_back(expr.get());
  std::vector<mlir::Value> disposableTensorArguments;
  auto callOp = genDeclaredCall(node->name(), args, loc(node),
                                &disposableTensorArguments);
  for (auto tensor : disposableTensorArguments) {
    DBG("automatically dispose fresh Tensor argument after extern call");
    mulberry_core::TensorDisposeOp::create(_builder, loc(node), tensor);
  }

  if (mulberry::isUnitType(node->type()))
    return nullptr;

  auto result = callOp.getResult(0);
  if (!isObjectReferenceType(node->type()))
    return result;

  auto calleeOpIter = findFunction(node->name());
  if (calleeOpIter == _functionsByName.end())
    llvm_unreachable("object call has no declared callee");
  if (calleeOpIter->second.isExtern)
    return boxExternObjectResult(result, node->type(), loc(node));

  if (result.getType() != getSourceMLIRType(node->type()))
    llvm_unreachable("non-extern object call did not return a reference");
  return result;
}

auto MLIRGenImpl::gen(const StructLiteralExpr *node) -> mlir::Value {
  auto *structType = node->structType();
  if (!structType) {
    ERR("struct literal has no Mulberry struct type");
    return nullptr;
  }

  DBG("use Mulberry struct literal `{0}`",
      formatType(structType));
  for (auto &expression : node->expressions())
    if (isTensorObjectType(expression->type()))
      markTensorReferencesEscaped(expression.get());
  return genStructLiteral(node, structType);
}

auto MLIRGenImpl::gen(const VariableExpr *node) -> mlir::Value {
  if (node->comptimeValue()) {
    auto &value = *node->comptimeValue();
    DBG("materialize comptime variable `{0}`", node->name());
    switch (value.kind()) {
    case ComptimeValue::Kind::Type:
      llvm_unreachable("comptime Type reached MLIRGen");
    case ComptimeValue::Kind::Bool:
      return arith::ConstantIntOp::create(
          _builder, loc(node), value.boolValue(), 1);
    case ComptimeValue::Kind::UInt64: {
      auto intType = llvm::cast<mlir::IntegerType>(getSourceMLIRType(node));
      return arith::ConstantIntOp::create(
          _builder, loc(node), value.uint64Value(), intType.getWidth());
    }
    case ComptimeValue::Kind::String: {
      StringLiteralExpr literal(node->location(),
                                std::string(value.stringValue()));
      literal.setType(node->type());
      return gen(&literal);
    }
    }
    llvm_unreachable("unexpected comptime value");
  }

  if (mulberry::isUnitType(node->type()))
    return nullptr;

  if (node->isFunctionValue()) {
    auto function = findFunction(node->name());
    if (function == _functionsByName.end())
      llvm_unreachable("function value has no declared function");
    if (function->second.isExtern)
      llvm_unreachable("extern function value reached MLIRGen");

    DBG("materialize function value `{0}`", node->name());
    auto symbol = mlir::FlatSymbolRefAttr::get(
        _builder.getContext(), std::string(node->name()));
    return func::ConstantOp::create(
        _builder, loc(node), function->second.operation.getFunctionType(),
        symbol);
  }
  return getVariableValue(node->name(), loc(node));
}

auto MLIRGenImpl::gen(const MemberExpr *node) -> mlir::Value {
  auto *field = getStructField(node);
  if (!field) {
    ERR("struct member access has no Mulberry field information");
    return nullptr;
  }

  auto fieldPtr = genLValue(node);
  return loadSourceValueFromStorage(fieldPtr, node->type(), loc(node));
}

auto MLIRGenImpl::gen(const DecimalLiteralExpr *node) -> mlir::Value {
  mlir::Type type = getSourceMLIRType(node);
  auto intType = llvm::cast<mlir::IntegerType>(type);
  return arith::ConstantIntOp::create(_builder, loc(node), node->value(),
                                      intType.getWidth());
}

auto MLIRGenImpl::gen(const FloatLiteralExpr *node) -> mlir::Value {
  mlir::Type type = getSourceMLIRType(node);
  return arith::ConstantFloatOp::create(
      _builder, loc(node), llvm::cast<mlir::FloatType>(type), node->value());
}

auto MLIRGenImpl::gen(const BoolLiteralExpr *node) -> mlir::Value {
  return arith::ConstantIntOp::create(_builder, loc(node), node->value(), 1);
}

auto MLIRGenImpl::gen(const CharLiteralExpr *node) -> mlir::Value {
  return arith::ConstantIntOp::create(_builder, loc(node), node->value(), 8);
}

auto MLIRGenImpl::gen(const StringLiteralExpr *node) -> mlir::Value {
  auto storageType = llvm::cast<mulberry_core::RecordType>(
      getLayoutMLIRType(node));
  return genStringLiteral(node, storageType);
}

auto MLIRGenImpl::gen(const InterpolatedStringExpr *node) -> mlir::Value {
  auto &segments = node->segments();
  if (segments.empty())
    llvm_unreachable("interpolated String has no segments");

  auto result = gen(segments.front().get());
  if (segments.size() == 1)
    return result;

  for (size_t index = 1; index < segments.size(); ++index) {
    auto next = gen(segments[index].get());
    auto call = genDeclaredLoweredCall(
        "std.string.concat", mlir::ValueRange{result, next}, loc(node));
    result = call.getResult(0);
  }
  return result;
}

auto MLIRGenImpl::gen(const ObjectIdentityExpr *node) -> mlir::Value {
  constexpr std::string_view functionName =
      "mulberry_string_object_identity";
  auto calleeIter = findFunction(functionName);
  if (calleeIter == _functionsByName.end()) {
    ERR("object identity call `{0}` has no declared callee", functionName);
    return nullptr;
  }

  auto &callee = calleeIter->second;
  if (!callee.isExtern)
    llvm_unreachable("object identity helper must use the runtime ABI");

  auto calleeType = callee.operation.getFunctionType();
  if (calleeType.getNumInputs() != 2 || calleeType.getNumResults() != 1)
    llvm_unreachable("object identity helper has an invalid signature");

  StringLiteralExpr typeName(node->location(), std::string(node->typeName()));
  typeName.setType(node->type());
  auto typeNameValue =
      genCallArgumentValue(&typeName, calleeType.getInput(0), true);

  auto objectReference = genObjectReference(node->value().get());
  auto objectPointer = castToType(objectReference, calleeType.getInput(1),
                                  loc(node));
  if (!objectPointer)
    llvm_unreachable("object identity reference cannot use runtime ABI");

  DBG("format object identity `{0}` through `{1}`", node->typeName(),
      functionName);
  auto call = func::CallOp::create(
      _builder, loc(node), functionName, calleeType.getResults(),
      mlir::ValueRange{typeNameValue, objectPointer});
  return boxExternObjectResult(call.getResult(0), node->type(), loc(node));
}

auto MLIRGenImpl::genStringLiteral(
    const StringLiteralExpr *node,
    mulberry_core::RecordType storageType) -> mlir::Value {
  auto location = loc(node);

  // String is a source object reference: the header lives on the GC heap, and
  // the bytes live in a separate NUL-terminated heap buffer for runtime/FFI.
  auto bytes = node->value();
  auto byteCount = createUInt64Constant(bytes.size() + 1, location);
  auto dataBuffer = mulberry_core::HeapAllocOp::create(
                        _builder, location, getPtrType(_builder.getI8Type()),
                        _builder.getI8Type(), byteCount)
                        .getResult();

  for (size_t index = 0; index < bytes.size(); ++index) {
    auto byteIndex = arith::ConstantIndexOp::create(
        _builder, location, static_cast<int64_t>(index));
    auto bytePtr = mulberry_core::PtrIndexOp::create(
        _builder, location, getPtrType(_builder.getI8Type()), dataBuffer,
        byteIndex);
    auto byteValue = arith::ConstantIntOp::create(
        _builder, location,
        static_cast<int64_t>(static_cast<unsigned char>(bytes[index])), 8);
    createStore(byteValue, bytePtr, location);
  }

  auto nulIndex = arith::ConstantIndexOp::create(
      _builder, location, static_cast<int64_t>(bytes.size()));
  auto nulPtr = mulberry_core::PtrIndexOp::create(
      _builder, location, getPtrType(_builder.getI8Type()), dataBuffer,
      nulIndex);
  auto nulValue = arith::ConstantIntOp::create(_builder, location, 0, 8);
  createStore(nulValue, nulPtr, location);

  auto storage = createHeapObject(storageType, location);
  auto lengthValue = arith::ConstantIntOp::create(
      _builder, location, static_cast<int64_t>(bytes.size()), 64);
  storeRecordFieldValue(storage, storageType, "length", lengthValue, location);
  storeRecordFieldValue(storage, storageType, "data", dataBuffer, location);

  return storage;
}

auto MLIRGenImpl::gen(const TypeLayoutExpr *node) -> mlir::Value {
  return arith::ConstantIntOp::create(_builder, loc(node), node->value(), 64);
}

auto MLIRGenImpl::gen(const HeapAllocExpr *node) -> mlir::Value {
  auto allocatedType = getStorageMLIRType(node->allocatedType());
  auto resultType =
      llvm::cast<mulberry_core::PtrType>(getSourceMLIRType(node));
  auto count = node->count()
                   ? gen(node->count().get())
                   : createUInt64Constant(1, loc(node));
  return mulberry_core::HeapAllocOp::create(_builder, loc(node), resultType,
                                            allocatedType, count);
}

auto MLIRGenImpl::genLValue(const Expr *node) -> mlir::Value {
  if (auto varExpr = llvm::dyn_cast<VariableExpr>(node)) {
    DBG("varExpr->name(): {0}", varExpr->name());
    auto parentAddress = getVariableAddress(varExpr->name(), loc(node));
    return parentAddress;
  }

  if (auto *memberExpr = llvm::dyn_cast<MemberExpr>(node)) {
    mlir::Value basePtr = genRecordPtrForMember(memberExpr);

    if (auto *field = getStructField(memberExpr)) {
      return createStructFieldPtr(basePtr, *field, loc(node));
    }

    ERR("struct member access has no Mulberry field information");
    return nullptr;
  }

  if (auto *indexExpr = llvm::dyn_cast<IndexExpr>(node)) {
    if (indexExpr->indexKind() == IndexExpr::IndexKind::Array)
      return genArrayElementPtr(indexExpr);

    if (indexExpr->indexKind() == IndexExpr::IndexKind::StdlibTensor)
      return genStdlibTensorElementPtr(indexExpr);

    auto source = gen(indexExpr->base().get());
    if (llvm::isa<mulberry_core::PtrType>(source.getType()))
      return genPtrIndex(indexExpr, source);
  }

  ERR("unknown EXPR");
  llvm_unreachable("unexpected lvalue expression");

  return nullptr;
}

auto MLIRGenImpl::genRecordPtrForMember(const MemberExpr *memberExpr)
    -> mlir::Value {
  auto *base = memberExpr->base().get();
  if (auto *ptrType = mulberry::getPtrType(base->type())) {
    if (mulberry::getStructType(ptrType->pointeeType()))
      return gen(base);
  }

  if (isObjectReferenceType(base->type())) {
    auto reference = genObjectReference(base);
    if (isTensorObjectType(base->type()))
      createTensorAssertAlive(reference, loc(memberExpr));
    return reference;
  }

  return genLValue(base);
}

auto MLIRGenImpl::getStructField(const MemberExpr *memberExpr) const
    -> const StructField * {
  auto *base = memberExpr->base().get();
  auto *baseType = base->type();
  auto *ptrType = mulberry::getPtrType(baseType);
  auto *structType = ptrType ? mulberry::getStructType(ptrType->pointeeType())
                             : mulberry::getStructType(baseType);
  if (!structType)
    return nullptr;

  auto index = memberExpr->fieldIndex();
  auto &fields = structType->fields();
  if (index >= fields.size()) {
    DBG("Mulberry struct field index `{0}` out of bounds for `{1}`", index,
        formatType(structType));
    return nullptr;
  }

  auto *field = &fields[index];
  if (!field->type()) {
    DBG("Mulberry struct field `{0}` has no Mulberry type",
        field->name());
    return nullptr;
  }

  DBG("use Mulberry struct field `{0}` from `{1}`", field->name(),
      formatType(structType));
  return field;
}

auto MLIRGenImpl::gen(const BinaryExpr *node) -> mlir::Value {
  using Operator = BinaryExpr::Operator;
  auto op = node->opEnum();

  auto *resultStructType = mulberry::getStructType(node->type());
  if (op == Operator::Add && resultStructType &&
      resultStructType->name() == "std.string.String") {
    const Expr *arguments[] = {node->lhs().get(), node->rhs().get()};
    return genDeclaredCall("std.string.concat", arguments, loc(node))
        .getResult(0);
  }

  auto lhs =
      castToType(gen(node->lhs().get()),
                 getSourceMLIRType(node->lhs().get()), loc(node->lhs().get()));
  auto rhs =
      castToType(gen(node->rhs().get()),
                 getSourceMLIRType(node->rhs().get()), loc(node->rhs().get()));
  if (llvm::isa<mlir::FloatType>(lhs.getType())) {
    switch (op) {
    case Operator::Add:
      return arith::AddFOp::create(_builder, loc(node), lhs, rhs);
    case Operator::Diff:
      return arith::SubFOp::create(_builder, loc(node), lhs, rhs);
    case Operator::Mul:
      return arith::MulFOp::create(_builder, loc(node), lhs, rhs);
    case Operator::Div:
      return arith::DivFOp::create(_builder, loc(node), lhs, rhs);
    case Operator::Rem:
      return arith::RemFOp::create(_builder, loc(node), lhs, rhs);
    case Operator::EQ:
      return arith::CmpFOp::create(
          _builder, loc(node), arith::CmpFPredicate::OEQ, lhs, rhs);
    case Operator::NEQ:
      return arith::CmpFOp::create(
          _builder, loc(node), arith::CmpFPredicate::ONE, lhs, rhs);
    case Operator::LT:
      return arith::CmpFOp::create(
          _builder, loc(node), arith::CmpFPredicate::OLT, lhs, rhs);
    case Operator::LE:
      return arith::CmpFOp::create(
          _builder, loc(node), arith::CmpFPredicate::OLE, lhs, rhs);
    case Operator::GT:
      return arith::CmpFOp::create(
          _builder, loc(node), arith::CmpFPredicate::OGT, lhs, rhs);
    case Operator::GE:
      return arith::CmpFOp::create(
          _builder, loc(node), arith::CmpFPredicate::OGE, lhs, rhs);
    default:
      break;
    }
  }

  switch (op) {
  case Operator::Add:
    return arith::AddIOp::create(_builder, loc(node), lhs, rhs);
  case Operator::Diff:
    return arith::SubIOp::create(_builder, loc(node), lhs, rhs);
  case Operator::Mul:
    return arith::MulIOp::create(_builder, loc(node), lhs, rhs);
  case Operator::Div:
    return arith::DivUIOp::create(_builder, loc(node), lhs, rhs);
  case Operator::Rem:
    return arith::RemUIOp::create(_builder, loc(node), lhs, rhs);
  case Operator::And:
    return arith::AndIOp::create(_builder, loc(node), lhs, rhs);
  case Operator::Or:
    return arith::OrIOp::create(_builder, loc(node), lhs, rhs);
  case Operator::EQ:
    return arith::CmpIOp::create(
        _builder, loc(node), arith::CmpIPredicate::eq, lhs, rhs);
  case Operator::NEQ:
    return arith::CmpIOp::create(
        _builder, loc(node), arith::CmpIPredicate::ne, lhs, rhs);
  case Operator::LT:
    return arith::CmpIOp::create(
        _builder, loc(node), arith::CmpIPredicate::ult, lhs, rhs);
  case Operator::LE:
    return arith::CmpIOp::create(
        _builder, loc(node), arith::CmpIPredicate::ule, lhs, rhs);
  case Operator::GT:
    return arith::CmpIOp::create(
        _builder, loc(node), arith::CmpIPredicate::ugt, lhs, rhs);
  case Operator::GE:
    return arith::CmpIOp::create(
        _builder, loc(node), arith::CmpIPredicate::uge, lhs, rhs);
  }

  llvm_unreachable("Unexpected statement");
}

auto MLIRGenImpl::gen(const AssignExpr *node) -> mlir::Value {
  if (isTensorObjectType(node->lhs()->type())) {
    markTensorReferencesEscaped(node->lhs().get());
    markTensorReferencesEscaped(node->rhs().get());
  }
  llvm::TypeSwitch<const Expr *>(node->lhs().get())
      .Case<VariableExpr>([&](const auto *var) {
        auto name = var->name();
        if (isObjectReferenceType(var->type())) {
          auto rhsReference = genObjectReference(node->rhs().get());
          auto *binding = getVariableBinding(name);
          if (binding && binding->isObjectReference()) {
            createStore(rhsReference, binding->mlirValue, loc(node));
            return;
          }

          auto referenceSlot = createAlloca(rhsReference.getType(), loc(node));
          createStore(rhsReference, referenceSlot, loc(node));
          setVariableObjectReference(name, referenceSlot);
          return;
        }

        auto rhs = gen(node->rhs().get());
        auto address = getVariableAddress(name, loc(node));
        if (!mulberry::isUnitType(node->lhs()->type())) {
          rhs =
              castToType(rhs, getSourceMLIRType(node->lhs().get()), loc(node));
          createStore(rhs, address, loc(node));
        }
      })
      .Case<MemberExpr>([&](const auto *member) {
        mlir::Value lhsPtr = genLValue(member);
        auto rhs = genValueForStorage(node->rhs().get(), member->type(),
                                      loc(node));
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
  case Stat::Stat_If:
    return gen(cast<IfStat>(node));
  case Stat::Stat_Match:
    return gen(cast<MatchStat>(node));
  case Stat::Stat_While:
    return gen(cast<WhileStat>(node));
  case Stat::Stat_For:
    return gen(cast<ForStat>(node));
  case Stat::Stat_Break:
    return gen(cast<BreakStat>(node));
  case Stat::Stat_Continue:
    return gen(cast<ContinueStat>(node));
  case Stat::Stat_Return:
    return gen(cast<ReturnStat>(node));
  }
}

auto MLIRGenImpl::genStatements(const VectorUniquePtr<Stat> &statements)
    -> void {
  for (auto &statement : statements) {
    if (_whileControls.empty() && _functionReturnControls.empty())
      gen(statement.get());
    else
      genGuardedStatement(statement.get());
  }
}

auto MLIRGenImpl::createTrueValue(mlir::Location location) -> mlir::Value {
  return arith::ConstantIntOp::create(_builder, location, 1, 1);
}

auto MLIRGenImpl::createFalseValue(mlir::Location location) -> mlir::Value {
  return arith::ConstantIntOp::create(_builder, location, 0, 1);
}

void MLIRGenImpl::disposeCurrentTensorLocals(mlir::Location location) {
  for (auto &[name, local] : _tensorLocalScopes.currentScope()) {
    if (local.escaped)
      continue;

    if (!local.initializedFlag) {
      disposeTensorLocal(name, local, location);
      continue;
    }

    auto initialized = createLoad(local.initializedFlag,
                                  _builder.getI1Type(), location);
    auto ifOp = scf::IfOp::create(
        _builder, location, initialized, /*withElseRegion=*/false);
    mlir::OpBuilder::InsertionGuard guard(_builder);
    _builder.setInsertionPointToStart(ifOp.thenBlock());
    disposeTensorLocal(name, local, location);
  }
}

void MLIRGenImpl::disposeTensorLocal(std::string_view name,
                                     const TensorLocal& local,
                                     mlir::Location location) {
  auto slotType = llvm::cast<mulberry_core::PtrType>(
      local.referenceSlot.getType());
  auto tensor = createLoad(local.referenceSlot, slotType.getPointeeType(),
                           location);
  DBG("automatically dispose local Tensor `{0}`", name);
  mulberry_core::TensorDisposeOp::create(_builder, location, tensor);
}

auto MLIRGenImpl::isFreshTensorExpression(const Expr *expr) const -> bool {
  if (!isTensorObjectType(expr->type()))
    return false;

  if (auto *matchExpr = llvm::dyn_cast<MatchExpr>(expr)) {
    if (matchExpr->arms().empty())
      return false;
    for (auto &arm : matchExpr->arms())
      if (!isFreshTensorExpression(arm->resultExpr().get()))
        return false;
    return true;
  }

  auto *call = llvm::dyn_cast<CallExpr>(expr);
  if (!call)
    return false;

  auto name = call->name();
  // Tensor is the only source object with early disposal today. Until the
  // language has a Disposable effect, distinguish owning stdlib constructors
  // from views such as reshape() and sliceFirst() after generic mangling.
  if (startsWith(name, "std_tensor_from__") ||
      startsWith(name, "std_tensor_zeros"))
    return true;

  // Extern Tensor ABI borrows arguments and transfers ownership of Tensor
  // results. Package names do not participate in the lifetime decision.
  auto function = _functionsByName.find(name);
  return function != _functionsByName.end() && function->second.isExtern;
}

auto MLIRGenImpl::tensorCallDoesNotRetain(std::string_view name) const
    -> bool {
  auto function = _functionsByName.find(name);
  if (function != _functionsByName.end() && function->second.isExtern)
    return true;

  constexpr std::string_view observers[] = {
      "std_tensor_Tensor_ndim__", "std_tensor_Tensor_numel__",
      "std_tensor_Tensor_shape__", "std_tensor_Tensor_dim__",
      "std_tensor_Tensor_stride__", "std_tensor_Tensor_dispose__",
  };
  for (auto observer : observers)
    if (startsWith(name, observer))
      return true;
  return false;
}

void MLIRGenImpl::markTensorReferencesEscaped(const Expr *expr) {
  // This analysis is deliberately path-insensitive. Any possible alias or
  // retained reference leaves the payload under GC management.
  if (!expr)
    return;

  if (auto *matchExpr = llvm::dyn_cast<MatchExpr>(expr)) {
    markTensorReferencesEscaped(matchExpr->value().get());
    for (auto &arm : matchExpr->arms())
      markTensorReferencesEscaped(arm->resultExpr().get());
    return;
  }

  if (auto *tryExpr = llvm::dyn_cast<TryExpr>(expr)) {
    markTensorReferencesEscaped(tryExpr->value().get());
    return;
  }

  if (auto *variable = llvm::dyn_cast<VariableExpr>(expr)) {
    if (isTensorObjectType(variable->type())) {
      if (auto *local = getTensorLocal(variable->name())) {
        DBG("Tensor local `{0}` escapes automatic disposal", variable->name());
        local->escaped = true;
      }
    }
    return;
  }

  if (auto *member = llvm::dyn_cast<MemberExpr>(expr)) {
    markTensorReferencesEscaped(member->base().get());
    return;
  }
  if (auto *call = llvm::dyn_cast<CallExpr>(expr)) {
    if (call->hasReceiver())
      markTensorReferencesEscaped(call->receiver().get());
    for (auto &argument : call->expressions())
      markTensorReferencesEscaped(argument.get());
    return;
  }
  if (auto *literal = llvm::dyn_cast<StructLiteralExpr>(expr)) {
    for (auto &value : literal->expressions())
      markTensorReferencesEscaped(value.get());
    return;
  }
  if (auto *constructor = llvm::dyn_cast<DataConstructorExpr>(expr)) {
    for (auto &value : constructor->expressions())
      markTensorReferencesEscaped(value.get());
    return;
  }
  if (auto *array = llvm::dyn_cast<ArrayLiteralExpr>(expr)) {
    for (auto &value : array->getElements())
      markTensorReferencesEscaped(value.get());
    return;
  }
  if (auto *index = llvm::dyn_cast<IndexExpr>(expr)) {
    markTensorReferencesEscaped(index->base().get());
    for (auto &value : index->indices())
      markTensorReferencesEscaped(value.get());
    return;
  }
  if (auto *assign = llvm::dyn_cast<AssignExpr>(expr)) {
    markTensorReferencesEscaped(assign->lhs().get());
    markTensorReferencesEscaped(assign->rhs().get());
    return;
  }
  if (auto *binary = llvm::dyn_cast<BinaryExpr>(expr)) {
    markTensorReferencesEscaped(binary->lhs().get());
    markTensorReferencesEscaped(binary->rhs().get());
    return;
  }
  if (auto *interpolated = llvm::dyn_cast<InterpolatedStringExpr>(expr)) {
    for (auto &segment : interpolated->segments())
      markTensorReferencesEscaped(segment.get());
    return;
  }
  if (auto *identity = llvm::dyn_cast<ObjectIdentityExpr>(expr)) {
    markTensorReferencesEscaped(identity->value().get());
    return;
  }
  if (auto *allocation = llvm::dyn_cast<HeapAllocExpr>(expr))
    markTensorReferencesEscaped(allocation->count().get());
}

auto MLIRGenImpl::currentFunctionReturnAllowed(mlir::Location location)
    -> mlir::Value {
  auto trueValue = createTrueValue(location);
  if (_functionReturnControls.empty())
    return trueValue;

  auto flagType = _builder.getI1Type();
  auto returned = createLoad(_functionReturnControls.back().returnedFlag,
                             flagType, location);
  return arith::XOrIOp::create(_builder, location, returned, trueValue);
}

auto MLIRGenImpl::currentExecutionAllowed() -> mlir::Value {
  auto location = _builder.getUnknownLoc();
  auto trueValue = createTrueValue(location);
  auto allowed = currentFunctionReturnAllowed(location);

  auto flagType = _builder.getI1Type();
  if (!_whileControls.empty()) {
    auto control = _whileControls.back();
    auto shouldBreak = createLoad(control.breakFlag, flagType, location);
    auto shouldContinue = createLoad(control.continueFlag, flagType, location);
    auto stopped = arith::OrIOp::create(
        _builder, location, shouldBreak, shouldContinue);
    auto loopAllowed =
        arith::XOrIOp::create(_builder, location, stopped, trueValue);
    allowed = arith::AndIOp::create(_builder, location, allowed, loopAllowed);
  }

  return allowed;
}

auto MLIRGenImpl::genGuardedStatement(const Stat *node) -> void {
  if (auto *variable = llvm::dyn_cast<VariableStat>(node)) {
    if (variable->comptimeValue())
      return;
    declareLocalVariableSlot(variable);
  }

  auto condition = currentExecutionAllowed();
  auto ifOp = scf::IfOp::create(_builder, loc(node), condition,
                                /*withElseRegion=*/false);
  _builder.setInsertionPointToStart(ifOp.thenBlock());
  gen(node);
  _builder.setInsertionPointAfter(ifOp);
}

auto MLIRGenImpl::declareLocalVariableSlot(const VariableStat *node) -> void {
  if (node->comptimeValue())
    return;

  auto *varType = node->type();
  if (mulberry::isUnitType(varType)) {
    setUnitVariable(node->variable()->name());
    return;
  }

  if (isObjectReferenceType(varType)) {
    auto referenceSlot = createAlloca(getSourceMLIRType(varType), loc(node));
    setVariableObjectReference(node->variable()->name(), referenceSlot);
    if (isTensorObjectType(varType) &&
        isFreshTensorExpression(node->init().get())) {
      auto initializedFlag = createAlloca(_builder.getI1Type(), loc(node));
      createStore(createFalseValue(loc(node)), initializedFlag, loc(node));
      registerTensorLocal(node->variable()->name(), referenceSlot,
                          initializedFlag);
    }
    return;
  }

  auto mlirType = getSourceMLIRType(varType);
  auto alloca = createAlloca(mlirType, loc(node));
  setVariableAddress(node->variable()->name(), alloca);
}

auto MLIRGenImpl::getPtrType(mlir::Type pointeeType) const
    -> mulberry_core::PtrType {
  return mulberry_core::PtrType::get(_builder.getContext(), pointeeType);
}

auto MLIRGenImpl::createIndexConstant(int64_t value,
                                      mlir::Location location) -> mlir::Value {
  return arith::ConstantIndexOp::create(_builder, location, value);
}

auto MLIRGenImpl::createUInt64Constant(int64_t value,
                                       mlir::Location location) -> mlir::Value {
  return arith::ConstantIntOp::create(_builder, location, value, 64);
}

auto MLIRGenImpl::createAlloca(mlir::Type mlirType,
                               mlir::Location location) -> mlir::Value {
  auto alloca = mulberry_core::AllocaOp::create(
      _builder, location, getPtrType(mlirType), mlirType);
  auto *parentBlock = alloca.getOperation()->getBlock();
  alloca.getOperation()->moveBefore(&parentBlock->front());
  return alloca;
}

auto MLIRGenImpl::createHeapObject(mlir::Type mlirType,
                                   mlir::Location location) -> mlir::Value {
  return mulberry_core::HeapAllocOp::create(
             _builder, location, getPtrType(mlirType), mlirType,
             createUInt64Constant(1, location))
      .getResult();
}

auto MLIRGenImpl::createLoad(mlir::Value ptr, mlir::Type type,
                             mlir::Location location) -> mlir::Value {
  return mulberry_core::LoadOp::create(_builder, location, type, ptr);
}

void MLIRGenImpl::createStore(mlir::Value value, mlir::Value ptr,
                              mlir::Location location) {
  mulberry_core::StoreOp::create(_builder, location, value, ptr);
}

auto MLIRGenImpl::createRecordFieldPtr(
    mlir::Value recordPtr, mulberry_core::RecordType recordType,
    std::string_view fieldName, mlir::Location location) -> mlir::Value {
  auto name = std::string(fieldName);
  auto fieldType = recordType.getFieldType(name);
  return mulberry_core::RecordGetFieldOp::create(
      _builder, location, getPtrType(fieldType), recordPtr, name);
}

auto MLIRGenImpl::loadRecordFieldValue(
    mlir::Value recordPtr, mulberry_core::RecordType recordType,
    std::string_view fieldName, mlir::Location location) -> mlir::Value {
  auto name = std::string(fieldName);
  auto fieldType = recordType.getFieldType(name);
  auto fieldPtr = createRecordFieldPtr(recordPtr, recordType, name, location);
  return createLoad(fieldPtr, fieldType, location);
}

void MLIRGenImpl::storeRecordFieldValue(
    mlir::Value recordPtr, mulberry_core::RecordType recordType,
    std::string_view fieldName, mlir::Value value, mlir::Location location) {
  auto name = std::string(fieldName);
  auto fieldType = recordType.getFieldType(name);
  auto fieldPtr = createRecordFieldPtr(recordPtr, recordType, name, location);
  auto storedValue = castToType(value, fieldType, location);
  createStore(storedValue, fieldPtr, location);
}

auto MLIRGenImpl::createStructFieldPtr(mlir::Value recordPtr,
                                       const StructField& field,
                                       mlir::Location location)
    -> mlir::Value {
  auto fieldType = getStorageMLIRType(field.type());
  auto fieldPtrType = getPtrType(fieldType);
  return mulberry_core::RecordGetFieldOp::create(
      _builder, location, fieldPtrType, recordPtr,
      std::string(field.name()));
}

auto MLIRGenImpl::gen(const DataConstructorExpr *node) -> mlir::Value {
  auto *dataType = mulberry::getDataType(node->type());
  if (!dataType || node->constructorIndex() >= dataType->constructors().size())
    llvm_unreachable("data constructor has no semantic variant");

  auto &constructor = dataType->constructors()[node->constructorIndex()];
  if (constructor.payloadTypes().size() != node->expressions().size())
    llvm_unreachable("data constructor payload count changed after Sema");

  std::vector<mlir::Value> payloads;
  for (size_t i = 0; i < node->expressions().size(); ++i) {
    auto &expression = node->expressions()[i];
    auto *payloadType = constructor.payloadTypes()[i];
    // Unit affects constructor typing and arity, but has no storage value.
    if (mulberry::isUnitType(payloadType)) {
      gen(expression.get());
      continue;
    }
    if (isTensorObjectType(expression->type()))
      markTensorReferencesEscaped(expression.get());
    payloads.push_back(genValueForStorage(
        expression.get(), payloadType, loc(expression.get())));
  }

  auto resultType = llvm::cast<mulberry_core::PtrType>(
      getSourceMLIRType(node));
  DBG("construct data variant `{0}` tag {1} as {2}", node->name(),
      node->constructorIndex(), resultType);
  return mulberry_core::DataConstructOp::create(
      _builder, loc(node), resultType, std::string(node->name()),
      static_cast<uint64_t>(node->constructorIndex()), payloads);
}

auto MLIRGenImpl::genStructLiteral(const StructLiteralExpr *structLiteral,
                                   const StructType *structType)
    -> mlir::Value {
  DBG("struct literal type: {0}", formatType(structType));

  auto recordType =
      llvm::dyn_cast<mulberry_core::RecordType>(
          getLayoutMLIRType(structType));
  if (!recordType) {
    ERR("NOT a RecordType. structType: {0}",
        formatType(structType));
    return nullptr;
  }

  auto &fields = structType->fields();
  if (fields.size() != structLiteral->expressions().size()) {
    ERR("Mulberry struct literal field count mismatch for `{0}`",
        formatType(structType));
    return nullptr;
  }

  for (auto &field : fields) {
    if (!field.type()) {
      ERR("Mulberry struct literal field `{0}` has no Mulberry type",
          field.name());
      return nullptr;
    }
  }

  auto object = createHeapObject(recordType, loc(structLiteral));
  DBG("structType: {0}, object: {1}", formatType(structType), object);

  unsigned index = 0;
  for (auto &expr : *structLiteral) {
    auto &field = fields[index];
    DBG("Mulberry struct literal field index: {0}, field name: {1}, "
        "expr type: {2}",
        field.index(), field.name(),
        formatType(expr->type()));
    auto fieldPtr = createStructFieldPtr(object, field, loc(structLiteral));
    auto value = genValueForStorage(expr.get(), field.type(), loc(expr.get()));
    createStore(value, fieldPtr, loc(expr.get()));

    index++;
  }

  return object;
}

auto MLIRGenImpl::getLayoutMLIRType(const Type *type) const -> mlir::Type {
  if (!type)
    return {};

  DBG("convert Mulberry type `{0}` to layout MLIR type", formatType(type));
  return _typeConverter.convertLayout(type);
}

auto MLIRGenImpl::getLayoutMLIRType(const Expr *expr) const -> mlir::Type {
  return getLayoutMLIRType(expr->type());
}

auto MLIRGenImpl::getSourceMLIRType(const Type *type) const -> mlir::Type {
  return _typeConverter.convertSource(type);
}

auto MLIRGenImpl::getSourceMLIRType(const Expr *expr) const -> mlir::Type {
  return getSourceMLIRType(expr->type());
}

auto MLIRGenImpl::getStorageMLIRType(const Type *type) const -> mlir::Type {
  return _typeConverter.convertStorage(type);
}

auto MLIRGenImpl::getParameterMLIRType(const Type *type,
                                       bool isExtern) const -> mlir::Type {
  return isExtern ? getLayoutMLIRType(type) : getSourceMLIRType(type);
}

auto MLIRGenImpl::getReturnMLIRType(const Type *type,
                                    bool isExtern) const -> mlir::Type {
  return isExtern ? getLayoutMLIRType(type) : getSourceMLIRType(type);
}

auto MLIRGenImpl::castToType(mlir::Value value, mlir::Type type,
                             mlir::Location location) -> mlir::Value {
  if (!value || value.getType() == type)
    return value;

  if (llvm::isa<mulberry_core::PtrType>(value.getType()) &&
      llvm::isa<mulberry_core::PtrType>(type))
    return mulberry_core::PtrCastOp::create(_builder, location, type, value);

  auto sourceIntType = llvm::dyn_cast<mlir::IntegerType>(value.getType());
  auto targetIntType = llvm::dyn_cast<mlir::IntegerType>(type);
  if (sourceIntType && targetIntType) {
    // Mulberry only has unsigned integer scalars today, so integer casts use
    // zero-extension and truncation instead of signed variants.
    if (sourceIntType.getWidth() < targetIntType.getWidth()) {
      DBG("castToType zero-extend integer {0} -> {1}", value.getType(), type);
      return arith::ExtUIOp::create(_builder, location, type, value);
    }
    DBG("castToType truncate integer {0} -> {1}", value.getType(), type);
    return arith::TruncIOp::create(_builder, location, type, value);
  }

  return nullptr;
}

auto MLIRGenImpl::loadObjectReferenceFromStorage(mlir::Value storagePtr,
                                                 const Type *type,
                                                 mlir::Location location)
    -> mlir::Value {
  return createLoad(storagePtr, getSourceMLIRType(type), location);
}

auto MLIRGenImpl::loadSourceValueFromStorage(mlir::Value storagePtr,
                                             const Type *type,
                                             mlir::Location location)
    -> mlir::Value {
  if (isObjectReferenceType(type))
    return loadObjectReferenceFromStorage(storagePtr, type, location);
  return createLoad(storagePtr, getStorageMLIRType(type), location);
}

auto MLIRGenImpl::genValueForStorage(const Expr *expr, const Type *type,
                                     mlir::Location location) -> mlir::Value {
  if (isObjectReferenceType(type))
    return castToType(genObjectReference(expr), getStorageMLIRType(type),
                      location);
  return castToType(gen(expr), getStorageMLIRType(type), location);
}

auto MLIRGenImpl::genIndexValue(const Expr *node) -> mlir::Value {
  if (auto *decimal = llvm::dyn_cast<DecimalLiteralExpr>(node))
    return arith::ConstantIndexOp::create(_builder, loc(node),
                                          decimal->value());

  auto value = gen(node);
  if (value.getType().isIndex())
    return value;

  if (!llvm::isa<mlir::IntegerType>(value.getType()))
    value = castToType(value, _builder.getI64Type(), loc(node));
  return arith::IndexCastOp::create(_builder, loc(node),
                                    _builder.getIndexType(), value);
}

auto MLIRGenImpl::genPtrIndex(const IndexExpr *expr, mlir::Value ptr)
    -> mlir::Value {
  auto ptrType = llvm::cast<mulberry_core::PtrType>(ptr.getType());
  auto index = genIndexValue(expr->indices().front().get());
  return mulberry_core::PtrIndexOp::create(_builder, loc(expr), ptrType, ptr,
                                           index);
}

auto MLIRGenImpl::genArrayElementPtr(const IndexExpr *expr) -> mlir::Value {
  auto recordType = llvm::cast<mulberry_core::RecordType>(
      getLayoutMLIRType(expr->base().get()));
  auto arrayPtr = genAddressableValue(expr->base().get(), recordType);
  auto dataPtr = loadRecordFieldValue(arrayPtr, recordType, "data", loc(expr));
  return mulberry_core::PtrIndexOp::create(
      _builder, loc(expr), dataPtr.getType(), dataPtr,
      genIndexValue(expr->indices().front().get()));
}

auto MLIRGenImpl::genStdlibTensorElementPtr(const IndexExpr *expr)
    -> mlir::Value {
  auto recordType = llvm::cast<mulberry_core::RecordType>(
      getLayoutMLIRType(expr->base().get()));
  auto tensorPtr = genAddressableValue(expr->base().get(), recordType);
  createTensorAssertAlive(tensorPtr, loc(expr));

  auto dataPtr = loadRecordFieldValue(tensorPtr, recordType, "data", loc(expr));

  auto stridesReference =
      loadRecordFieldValue(tensorPtr, recordType, "strides", loc(expr));
  auto stridesReferenceType =
      llvm::cast<mulberry_core::PtrType>(stridesReference.getType());
  auto stridesRecordType = llvm::cast<mulberry_core::RecordType>(
      stridesReferenceType.getPointeeType());
  auto stridesData =
      loadRecordFieldValue(stridesReference, stridesRecordType, "data",
                           loc(expr));
  auto stridesDataFieldType = stridesData.getType();

  auto zero = arith::ConstantIndexOp::create(_builder, loc(expr), 0);
  mlir::Value offset = zero;
  for (size_t i = 0; i < expr->indices().size(); ++i) {
    auto dimensionIndex =
        arith::ConstantIndexOp::create(_builder, loc(expr), i);
    auto stridePtr = mulberry_core::PtrIndexOp::create(
        _builder, loc(expr), stridesDataFieldType, stridesData,
        dimensionIndex);
    auto stride = createLoad(stridePtr, _builder.getI64Type(), loc(expr));
    auto strideIndex = arith::IndexCastOp::create(
        _builder, loc(expr), _builder.getIndexType(), stride);
    auto dimensionOffset = arith::MulIOp::create(
        _builder, loc(expr), genIndexValue(expr->indices()[i].get()),
        strideIndex);
    offset = arith::AddIOp::create(_builder, loc(expr), offset,
                                   dimensionOffset);
  }

  return mulberry_core::PtrIndexOp::create(
      _builder, loc(expr), dataPtr.getType(), dataPtr, offset);
}

auto MLIRGenImpl::genStdlibListGet(const IndexExpr *expr) -> mlir::Value {
  auto list = genAddressableValue(expr->base().get(),
                                  getLayoutMLIRType(expr->base().get()));
  auto index = castToType(gen(expr->indices().front().get()),
                          _builder.getI64Type(), loc(expr));
  auto call = genDeclaredLoweredCall(
      expr->getFunctionName(), mlir::ValueRange{list, index}, loc(expr));
  return call.getResult(0);
}

auto MLIRGenImpl::genAddressableValue(const Expr *expr,
                                      mlir::Type valueType) -> mlir::Value {
  if (isObjectReferenceType(expr->type()))
    return castToType(genObjectReference(expr), getPtrType(valueType),
                      loc(expr));

  bool shouldSpill = true;
  if (auto *variable = llvm::dyn_cast<VariableExpr>(expr)) {
    auto *binding = getVariableBinding(variable->name());
    if (binding && binding->isAddress())
      return binding->mlirValue;
  } else if (expr->isLvalue()) {
    shouldSpill = false;
    if (auto *index = llvm::dyn_cast<IndexExpr>(expr))
      // List elements are settable through List.set, but List.get returns a
      // value, so passing one by reference needs a temporary slot.
      shouldSpill = index->indexKind() == IndexExpr::IndexKind::StdlibList;
    if (!shouldSpill)
      return genLValue(expr);
  }

  auto ptr = createAlloca(valueType, loc(expr));
  auto value = gen(expr);
  if (auto ptrType =
          llvm::dyn_cast<mulberry_core::PtrType>(value.getType())) {
    if (ptrType.getPointeeType() == valueType)
      return value;
  }
  createStore(castToType(value, valueType, loc(expr)), ptr, loc(expr));
  return ptr;
}

auto MLIRGenImpl::genObjectReference(const Expr *expr) -> mlir::Value {
  if (!isObjectReferenceType(expr->type()))
    llvm_unreachable("non-object expression requested as an object reference");
  return gen(expr);
}

auto MLIRGenImpl::loadObjectHeaderForExternArgument(
    const Expr *expr, mlir::Type parameterType) -> mlir::Value {
  auto objectType = getLayoutMLIRType(expr);
  if (parameterType != objectType)
    llvm_unreachable("extern object parameter does not use value ABI");

  auto objectReference = genObjectReference(expr);
  if (isTensorObjectType(expr->type()))
    createTensorAssertAlive(objectReference, loc(expr));
  DBG("load extern object argument header: {0} -> {1}",
      objectReference.getType(), parameterType);
  return createLoad(objectReference, parameterType, loc(expr));
}

auto MLIRGenImpl::boxExternObjectResult(mlir::Value value, const Type *type,
                                        mlir::Location location)
    -> mlir::Value {
  auto objectType = getLayoutMLIRType(type);
  if (value.getType() != objectType)
    llvm_unreachable("extern object result does not use value ABI");

  auto objectReference = createHeapObject(objectType, location);
  DBG("box extern object result: {0} -> {1}", value.getType(),
      objectReference.getType());
  createStore(value, objectReference, location);
  return objectReference;
}

auto MLIRGenImpl::genCallArgumentValue(const Expr *expr,
                                       mlir::Type parameterType,
                                       bool isExtern) -> mlir::Value {
  if (isObjectReferenceType(expr->type())) {
    if (isExtern)
      return loadObjectHeaderForExternArgument(expr, parameterType);
    return castToType(genObjectReference(expr), parameterType, loc(expr));
  }

  auto exprType = getLayoutMLIRType(expr);
  if (!isExtern) {
    auto ptrType =
        llvm::dyn_cast<mulberry_core::PtrType>(parameterType);
    if (!ptrType)
      return castToType(gen(expr), parameterType, loc(expr));

    auto pointeeType = ptrType.getPointeeType();
    if (exprType == pointeeType) {
      return genAddressableValue(expr, pointeeType);
    }
  }

  return castToType(gen(expr), parameterType, loc(expr));
}

auto MLIRGenImpl::gen(const ArrayLiteralExpr *expr) -> mlir::Value {
  for (auto &element : expr->getElements())
    if (isTensorObjectType(element->type()))
      markTensorReferencesEscaped(element.get());
  auto arrayType =
      llvm::cast<mulberry_core::RecordType>(getLayoutMLIRType(expr));
  return genArrayLiteral(expr, arrayType);
}

auto MLIRGenImpl::genArrayLiteral(
    const ArrayLiteralExpr *expr,
    mulberry_core::RecordType arrayType) -> mlir::Value {
  auto location = loc(expr);
  auto elementCount = static_cast<int64_t>(expr->getElements().size());
  auto dataFieldType = arrayType.getFieldType("data");
  auto dataPtrType = llvm::cast<mulberry_core::PtrType>(dataFieldType);
  auto data = mulberry_core::HeapAllocOp::create(
                  _builder, location, dataPtrType,
                  dataPtrType.getPointeeType(),
                  createUInt64Constant(elementCount, location))
                  .getResult();
  auto *sourceArrayType = llvm::cast<ArrayType>(expr->type());
  storeArrayElements(expr, sourceArrayType, data, dataPtrType.getPointeeType());

  // Source Arrays are objects. Put the header itself on the GC heap so
  // binding, assignment, call, and return copy the reference, not the header.
  auto arrayPtr = createHeapObject(arrayType, location);
  storeRecordFieldValue(arrayPtr, arrayType, "length",
                        createUInt64Constant(elementCount, location),
                        location);
  storeRecordFieldValue(arrayPtr, arrayType, "data", data, location);
  return arrayPtr;
}

auto MLIRGenImpl::genTensorDispose(const CallExpr *expr) -> mlir::Value {
  auto tensor = genObjectReference(expr->expressions().front().get());
  mulberry_core::TensorDisposeOp::create(_builder, loc(expr), tensor);
  return nullptr;
}

auto MLIRGenImpl::genTensorStorageAlloc(const CallExpr *expr) -> mlir::Value {
  auto location = loc(expr);
  auto storagePtrType =
      llvm::cast<mulberry_core::PtrType>(getSourceMLIRType(expr));
  auto storageType =
      llvm::cast<mulberry_core::RecordType>(storagePtrType.getPointeeType());
  auto dataPtrType = llvm::cast<mulberry_core::PtrType>(
      storageType.getFieldType("data"));
  auto elementType = dataPtrType.getPointeeType();
  auto payloadType = mulberry_core::TensorType::get(
      _builder.getContext(), {mlir::ShapedType::kDynamic}, elementType);
  auto count = gen(expr->expressions().front().get());
  auto allocation = mulberry_core::TensorStorageAllocOp::create(
      _builder, location, mlir::TypeRange{storagePtrType, payloadType},
      elementType, count);
  return allocation.getStorage();
}

void MLIRGenImpl::createTensorAssertAlive(mlir::Value tensor,
                                          mlir::Location location) {
  mulberry_core::TensorAssertAliveOp::create(_builder, location, tensor);
}

void MLIRGenImpl::storeArrayElements(const ArrayLiteralExpr *expr,
                                     const ArrayType *arrayType,
                                     mlir::Value dataPtr,
                                     mlir::Type elementType) {
  auto ptrType = llvm::cast<mulberry_core::PtrType>(dataPtr.getType());
  auto *sourceElementType = arrayType->elementType();
  for (size_t index = 0; index < expr->getElements().size(); ++index) {
    auto *element = expr->getElements()[index].get();
    auto elementPtr = mulberry_core::PtrIndexOp::create(
        _builder, loc(element), ptrType, dataPtr,
        createIndexConstant(index, loc(element)));
    auto value = genValueForStorage(element, sourceElementType, loc(element));
    value = castToType(value, elementType, loc(element));
    createStore(value, elementPtr, loc(element));
  }
}

mlir::Value MLIRGenImpl::gen(const IndexExpr *expr) {
  if (expr->indexKind() == IndexExpr::IndexKind::Array) {
    auto ptr = genArrayElementPtr(expr);
    return loadSourceValueFromStorage(ptr, expr->type(), loc(expr));
  }

  if (expr->indexKind() == IndexExpr::IndexKind::StdlibTensor) {
    auto ptr = genStdlibTensorElementPtr(expr);
    return loadSourceValueFromStorage(ptr, expr->type(), loc(expr));
  }

  if (expr->indexKind() == IndexExpr::IndexKind::StdlibList)
    return genStdlibListGet(expr);

  auto source = gen(expr->base().get());
  if (llvm::isa<mulberry_core::PtrType>(source.getType())) {
    auto ptr = genPtrIndex(expr, source);
    return loadSourceValueFromStorage(ptr, expr->type(), loc(expr));
  }

  llvm_unreachable("index expression was not classified by Sema");
}

void MLIRGenImpl::genAssignment(const IndexExpr *lhs, const Expr *rhs) {
  if (lhs->indexKind() == IndexExpr::IndexKind::Array) {
    auto ptr = genArrayElementPtr(lhs);
    auto rhsValue = genValueForStorage(rhs, lhs->type(), loc(lhs));
    createStore(rhsValue, ptr, loc(lhs));
    return;
  }

  if (lhs->indexKind() == IndexExpr::IndexKind::StdlibTensor) {
    auto ptr = genStdlibTensorElementPtr(lhs);
    auto rhsValue = genValueForStorage(rhs, lhs->type(), loc(lhs));
    createStore(rhsValue, ptr, loc(lhs));
    return;
  }

  if (lhs->indexKind() == IndexExpr::IndexKind::StdlibList) {
    auto list = genAddressableValue(lhs->base().get(),
                                    getLayoutMLIRType(lhs->base().get()));
    auto index = castToType(gen(lhs->indices().front().get()),
                            _builder.getI64Type(), loc(lhs));
    auto rhsValue = genValueForStorage(rhs, lhs->type(), loc(lhs));
    genDeclaredLoweredCall(lhs->setFunctionName(),
                           mlir::ValueRange{list, index, rhsValue}, loc(lhs));
    return;
  }

  mlir::Value source = gen(lhs->base().get());
  if (llvm::isa<mulberry_core::PtrType>(source.getType())) {
    auto ptr = genPtrIndex(lhs, source);
    auto rhsValue = genValueForStorage(rhs, lhs->type(), loc(lhs));
    createStore(rhsValue, ptr, loc(lhs));
    return;
  }

  llvm_unreachable("index assignment was not classified by Sema");
}

auto MLIRGenImpl::gen(const VariableStat *node) -> void {
  if (node->comptimeValue()) {
    DBG("skip comptime variable declaration `{0}`",
        node->variable()->name());
    return;
  }

  auto *varType = node->type();
  auto *ptrType = mulberry::getPtrType(varType);
  auto varName = node->variable()->name();
  auto *predeclaredBinding = getCurrentVariableBinding(varName);

  if (ptrType) {
    DBG("use Mulberry variable ptr type `{0}`", formatType(ptrType));
    auto targetType = getSourceMLIRType(varType);
    auto value = gen(node->init().get());
    value = castToType(value, targetType, loc(node));

    if (predeclaredBinding && predeclaredBinding->isAddress()) {
      createStore(value, predeclaredBinding->mlirValue, loc(node));
      return;
    }

    if (node->isConstBinding()) {
      setVariableValue(varName, value);
      return;
    }

    auto alloca = createAlloca(targetType, loc(node));
    setVariableAddress(varName, alloca);
    createStore(value, alloca, loc(node));
    return;
  }

  if (isObjectReferenceType(varType)) {
    DBG("use Mulberry object reference `{0}`", formatType(varType));
    auto freshTensor = isFreshTensorExpression(node->init().get());
    if (isTensorObjectType(varType) && !freshTensor)
      markTensorReferencesEscaped(node->init().get());
    auto objectReference = genObjectReference(node->init().get());
    if (predeclaredBinding && predeclaredBinding->isObjectReference()) {
      createStore(objectReference, predeclaredBinding->mlirValue, loc(node));
      if (auto *local = getTensorLocal(varName))
        createStore(createTrueValue(loc(node)), local->initializedFlag,
                    loc(node));
      return;
    }

    auto referenceSlot = createAlloca(objectReference.getType(), loc(node));
    createStore(objectReference, referenceSlot, loc(node));
    setVariableObjectReference(varName, referenceSlot);
    if (freshTensor)
      registerTensorLocal(varName, referenceSlot, nullptr);
    return;
  }

  if (mulberry::isUnitType(varType)) {
    setUnitVariable(varName);
    gen(node->init().get());
    return;
  }

  auto mlirType = getSourceMLIRType(varType);
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

namespace mulberry {

auto mlirGen(const llvm::SourceMgr &sourceManager, mlir::MLIRContext &context,
             const Module &moduleAST, mlir::OwningOpRef<mlir::ModuleOp> &module)
    -> MulberryResult {
  auto generator = MLIRGenImpl(sourceManager, context);
  auto result = generator.gen(moduleAST);
  module = generator.module;
  return result;
}

} // end namespace mulberry
