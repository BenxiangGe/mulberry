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
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/SourceMgr.h"

#include <functional>
#include <map>
#include <optional>
#include <set>
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
  llvm::StringRef _fileNameIdentifier;
  int _globalTensorCounter = 0;
  MLIRTypeConverter _typeConverter{_builder};

  struct FunctionSymbol {
    enum class ABI {
      CIR,
      Func,
    };

    ABI abi = ABI::CIR;
    std::vector<const Type *> parameterTypes;
  };

  NameMap<FunctionSymbol> _functionsByName;
  FunctionSymbol::ABI _currentFunctionABI = FunctionSymbol::ABI::CIR;

  struct TensorDimSource {
    mlir::Value tensor;
    int64_t dim;
  };

  // Declarations
  auto gen(const Decl *node, mlir::Operation *&op) -> CherryResult;
  auto gen(const Prototype *node, FunctionSymbol::ABI abi) -> mlir::Operation *;
  auto gen(const FunctionDecl *node) -> mlir::Operation *;
  auto gen(const StructDecl *node) -> CherryResult;

  // Expressions
  auto gen(const Expr *node) -> mlir::Value;
  auto genBlock(const BlockExpr *node) -> llvm::FailureOr<mlir::Value>;
  auto gen(const UnitExpr *node) -> mlir::Value;
  auto gen(const BlockExpr *node) -> mlir::Value;
  auto gen(const IfExpr *node) -> mlir::Value;
  auto gen(const WhileExpr *node) -> mlir::Value;
  auto gen(const ForExpr *node) -> mlir::Value;
  auto genTensorCarriedFor(const ForExpr *node,
                           const std::vector<std::string> &carriedNames)
      -> mlir::Value;
  auto genPrint(const CallExpr *node) -> mlir::Value;
  auto genMatmul(const CallExpr *node) -> mlir::Value;
  auto genMatadd(const CallExpr *node) -> mlir::Value;
  auto genTranspose(const CallExpr *node) -> mlir::Value;
  auto genElementwiseNN(const CallExpr *node) -> mlir::Value;
  auto genArgmax(const CallExpr *node) -> mlir::Value;
  auto genSize(const CallExpr *node) -> mlir::Value;
  auto gen(const CallExpr *node) -> mlir::Value;
  auto gen(const ListLiteralExpr *node) -> mlir::Value;
  auto gen(const StructLiteralExpr *node) -> mlir::Value;
  auto gen(const VariableExpr *node) -> mlir::Value;
  auto gen(const MemberExpr *node) -> mlir::Value;
  auto gen(const DecimalLiteralExpr *node) -> mlir::Value;
  auto gen(const FloatLiteralExpr *node) -> mlir::Value;
  auto gen(const BoolLiteralExpr *node) -> mlir::Value;
  auto gen(const AssignExpr *node) -> mlir::Value;
  auto gen(const BinaryExpr *node) -> mlir::Value;

  // Statements
  auto gen(const Stat *node) -> CherryResult;
  auto gen(const VariableStat *node) -> CherryResult;
  auto gen(const ExprStat *node) -> CherryResult;

  auto getAlignOne() -> mlir::IntegerAttr;

  template <typename T>
  void setSymbol(NameMap<T> &symbols, std::string_view name, T value) {
    symbols[std::string(name)] = value;
  }

  void resetVariableScopes() {
    _variableScopes.reset();
    enterVariableScope();
  }

  void enterVariableScope() {
    _variableScopes.enterScope();
  }

  void leaveVariableScope() {
    _variableScopes.leaveScope();
  }

  void setVariable(std::string_view name, mlir::Value value) {
    if (_variableScopes.empty())
      enterVariableScope();
    setSymbol(_variableScopes.currentScope(), name, value);
  }

  void assignVariable(std::string_view name, mlir::Value value) {
    if (!_variableScopes.assign(name, value))
      setVariable(name, value);
  }

  auto getVariable(std::string_view name) -> mlir::Value {
    if (auto *variable = _variableScopes.lookup(name))
      return *variable;
    return {};
  }

  void setFunction(std::string_view name, FunctionSymbol func) {
    setSymbol(_functionsByName, name, func);
  }

  auto makeFunctionSymbol(const Prototype *node, FunctionSymbol::ABI abi)
      -> FunctionSymbol {
    std::vector<const Type *> parameterTypes;
    for (const auto& param : node->parameters())
      parameterTypes.push_back(param->type());
    return FunctionSymbol{abi, std::move(parameterTypes)};
  }

  auto findFunction(std::string_view name) {
    return _functionsByName.find(name);
  }

  auto genStructLiteral(const StructLiteralExpr *structLiteral,
                        const StructType *structType,
                        mlir::Value targetPtr) -> mlir::Value;
  auto genListLiteral(const ListLiteralExpr *listLiteral,
                      const ListType *listType) -> mlir::Value;
  auto genTensorListLiteral(const ListLiteralExpr *listLiteral,
                            const ListType *listType)
      -> mlir::Value;
  auto getStaticIndex(const Expr *expr) -> std::optional<uint64_t>;
  void collectAssignedVariables(
      const Expr *expr, std::set<std::string> &names,
      const std::function<bool(const Type *)> &shouldCollect);
  void collectAssignedVariables(
      const Stat *stat, std::set<std::string> &names,
      const std::function<bool(const Type *)> &shouldCollect);
  void collectAssignedTensorVariables(const Expr *expr,
                                      std::set<std::string> &names);
  void collectAssignedTensorVariables(const Stat *stat,
                                      std::set<std::string> &names);
  auto getTensorCarriedVariables(const ForExpr *node) -> std::vector<std::string>;
  auto getFuncABICarriedVariables(const ForExpr *node)
      -> std::vector<std::string>;
  auto genTensorListIndex(const IndexExpr *expr, mlir::Value tensorList)
      -> mlir::Value;
  auto isMulberryList(mlir::Value value) -> bool;
  auto getMulberryListElementType(mlir::Value value) -> mlir::Type;
  auto isTensorList(const ListType *type) const -> bool;
  auto containsTensorValue(const Type *type) const -> bool;
  auto containsTensorValue(const BlockExpr *block) const -> bool;
  auto containsTensorValue(const Expr *expr) const -> bool;
  auto containsTensorValue(const Stat *stat) const -> bool;
  auto useFuncABI(const Prototype *node) const -> bool;
  auto useFuncABI(const FunctionDecl *node) const -> bool;
  auto isFuncABI() const -> bool;
  auto isFuncABIScalarType(const Type *type) const -> bool;
  auto isFuncABIValueType(const Type *type) const -> bool;
  auto isFuncABIScalarExpr(const Expr *expr) const -> bool;
  auto isSupportedFuncABIParameterType(const Type *type) const -> bool;
  auto isSupportedFuncABIReturnType(const Type *type) const -> bool;
  auto canMaterializeListStorage(const ListType *type) const -> bool;
  auto getListStorageElementType(const ListType *listType) -> mlir::Type;
  auto createListStorage(mlir::Location location, mlir::Type listStorageType,
                         mlir::Type elementType, mlir::Value dataPtr,
                         size_t length) -> mlir::Value;
  void storeListElement(mlir::Location location, mlir::Value dataPtr,
                        mlir::Type elementType, size_t index,
                        mlir::Value value);
  auto getListStoragePointer(const IndexExpr *expr) -> mlir::Value;
  auto genListElementPointer(const IndexExpr *expr) -> mlir::Value;
  auto gen(const TensorLiteralExpr *expr) -> mlir::Value;
  auto genGlobalTensorLiteral(const TensorLiteralExpr *expr,
                              std::string_view varName) -> mlir::Value;
  void storeTensorElements(const TensorLiteralExpr *expr, mlir::Value memref,
                           mlir::Type elementType,
                           llvm::SmallVectorImpl<mlir::Value> &indices);
  mlir::Value gen(const IndexExpr *expr, bool isLValue = false);
  void genAssignment(const IndexExpr *lhs, const Expr *rhs);

  auto genLValue(const Expr *node) -> mlir::Value;
  auto genRValue(const Expr *node) -> mlir::Value;
  auto getStructField(const MemberExpr *memberExpr) const
      -> const StructField *;
  auto genIndexValue(const Expr *node) -> mlir::Value;
  auto genCIRIndexValue(const Expr *node) -> mlir::Value;
  auto genTensorDim(mlir::Location location, TensorDimSource source)
      -> mlir::Value;
  auto createTensorAlloc(mlir::Location location, mlir::MemRefType memRefType,
                         const std::vector<TensorDimSource>& resultDims)
      -> mlir::Value;
  auto createSameShapeTensorAlloc(mlir::Location location,
                                  mlir::MemRefType memRefType,
                                  mlir::Value source)
      -> mlir::Value;
  auto genMemRefElementValue(const Expr *node, mlir::Type elementType)
      -> mlir::Value;
  auto genMemRefLoadValue(const IndexExpr *expr) -> mlir::Value;
  auto genFuncABIIf(const IfExpr *node) -> mlir::Value;
  auto genFuncABIFor(const ForExpr *node) -> mlir::Value;
  auto genFuncABIBinary(const BinaryExpr *node) -> mlir::Value;
  auto genFuncABIBoolToUInt64(mlir::Value operand, mlir::Location location)
      -> mlir::Value;
  auto genFuncABISize(const CallExpr *node) -> mlir::Value;
  auto castToType(mlir::Value value, mlir::Type type, mlir::Location location)
      -> mlir::Value;
  auto genFuncABIArgument(const Expr *expr, const Type *paramType)
      -> mlir::Value;
  auto genFuncABIScalarLiteral(const DecimalLiteralExpr *node) -> mlir::Value;
  auto genFuncABIScalarLiteral(const FloatLiteralExpr *node) -> mlir::Value;
  auto genFuncABIScalarLiteral(const BoolLiteralExpr *node) -> mlir::Value;
  auto getMLIRType(const Type *type) const -> mlir::Type;
  auto getMLIRType(const Expr *expr) const -> mlir::Type;
  auto getFuncABIType(const Type *type) const -> mlir::Type;
  auto getFuncABIType(const Expr *expr) const -> mlir::Type;
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
  auto getFunctionReturnType(const Type *returnType) const -> mlir::Type;

  // Utility
  auto loc(const Node *node) -> mlir::Location {
    auto [line, col] = _sourceManager.getLineAndColumn(node->location());
    return mlir::FileLineColLoc::get(
        _builder.getStringAttr(_fileNameIdentifier), line, col);
  }

  auto emitError(const Node *node, const llvm::Twine& message) -> CherryResult {
    _sourceManager.PrintMessage(node->location(),
                                llvm::SourceMgr::DiagKind::DK_Error,
                                message);
    return failure();
  }

  auto createEntryBlockAlloca(mlir::Type mlirType, mlir::Location loc)
      -> mlir::Value {
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
    mlir::Operation *op = nullptr;
    if (gen(decl.get(), op))
      return failure();
    if (op)
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

auto MLIRGenImpl::gen(const Decl *node, mlir::Operation *&op) -> CherryResult {
  switch (node->getKind()) {
  case Decl::Decl_Function: {
    op = gen(cast<FunctionDecl>(node));
    return op ? success() : failure();
  }
  case Decl::Decl_Struct: {
    op = nullptr;
    return gen(cast<StructDecl>(node));
  }
  }
}

auto MLIRGenImpl::gen(const Prototype *node, FunctionSymbol::ABI abi)
    -> mlir::Operation * {
  auto usesFuncABI = abi == FunctionSymbol::ABI::Func;
  llvm::SmallVector<mlir::Type, 3> argTypes;
  for (auto &param : node->parameters()) {
    auto *paramType = param->type();
    if (usesFuncABI && !isSupportedFuncABIParameterType(paramType)) {
      emitError(param.get(),
                "Tensor function ABI only supports Tensor and builtin scalar "
                "parameters for now");
      return nullptr;
    }
    if (auto *listType = cherry::getListType(paramType)) {
      if (containsTensorValue(listType)) {
        emitError(param.get(),
                  "function parameters containing Tensor require list/tensor "
                  "descriptor ABI");
        return nullptr;
      }
    }
    if (!cherry::isTensorType(paramType) && containsTensorValue(paramType)) {
      emitError(param.get(),
                "function parameters containing Tensor require tensor "
                "descriptor ABI");
      return nullptr;
    }
    argTypes.push_back(usesFuncABI ? getFuncABIType(paramType)
                                   : getMLIRType(paramType));
  }

  auto funcName = node->id()->name();
  auto *returnType = node->type();
  if (usesFuncABI && !isSupportedFuncABIReturnType(returnType)) {
    emitError(node,
              "Tensor function ABI only supports Tensor and builtin scalar "
              "returns for now");
    return nullptr;
  }
  if (auto *listType = cherry::getListType(returnType)) {
    if (containsTensorValue(listType)) {
      emitError(node,
                "function returns containing Tensor require list/tensor "
                "descriptor ABI");
      return nullptr;
    }
  }
  if (!cherry::isTensorType(returnType) && containsTensorValue(returnType)) {
    emitError(node,
              "function returns containing Tensor require tensor descriptor "
              "ABI");
    return nullptr;
  }

  if (usesFuncABI) {
    mlir::TypeRange resultTypes;
    llvm::SmallVector<mlir::Type, 1> nonUnitResultTypes;
    if (!cherry::isUnitType(returnType)) {
      nonUnitResultTypes.push_back(getFuncABIType(returnType));
      resultTypes = nonUnitResultTypes;
    }

    auto funcType =
        mlir::FunctionType::get(_builder.getContext(), argTypes, resultTypes);
    auto func = mlir::func::FuncOp::create(loc(node), funcName, funcType);

    auto *entryBlock = func.addEntryBlock();
    _builder.setInsertionPointToStart(entryBlock);
    for (const auto &varValue :
         llvm::zip(node->parameters(), entryBlock->getArguments())) {
      auto &var = std::get<0>(varValue);
      auto varName = var->variable()->name();
      auto value = std::get<1>(varValue);
      auto *paramType = var->type();

      if (cherry::isUnitType(paramType)) {
        setVariable(varName, nullptr);
        continue;
      }

      if (isFuncABIValueType(paramType)) {
        setVariable(varName, value);
        continue;
      }

      auto alloca =
          createEntryBlockAlloca(getMLIRType(paramType), loc(node));
      setVariable(varName, alloca);
      cir::StoreOp::create(_builder, loc(node), value, alloca,
                           /*isVolatile=*/false,
                           /*alignment=*/mlir::IntegerAttr{},
                           /*sync_scope=*/cir::SyncScopeKindAttr(),
                           /*mem-order=*/cir::MemOrderAttr());
    }

    setFunction(funcName, makeFunctionSymbol(node, abi));

    DBG("funcName: {0}", funcName);

    return func;
  }

  auto funcType = cir::FuncType::get(argTypes,
                                     getFunctionReturnType(returnType));
  mlir::OperationState state(loc(node), cir::FuncOp::getOperationName());
  cir::FuncOp::build(_builder, state, funcName, funcType);
  auto func = llvm::cast<cir::FuncOp>(mlir::Operation::create(state));

  auto &entryBlock = *func.addEntryBlock();
  _builder.setInsertionPointToStart(&entryBlock);
  for (const auto &varValue :
       llvm::zip(node->parameters(), entryBlock.getArguments())) {
    auto &var = std::get<0>(varValue);
    auto varName = var->variable()->name();
    auto value = std::get<1>(varValue);
    auto *paramType = var->type();
    if (cherry::isUnitType(paramType)) {
      setVariable(varName, nullptr);
      continue;
    }

    auto alloca =
        createEntryBlockAlloca(getMLIRType(paramType), loc(node));
    setVariable(varName, alloca);
    cir::StoreOp::create(_builder, loc(node), value, alloca,
                         /*isVolatile=*/false,
                         /*alignment=*/mlir::IntegerAttr{},
                         /*sync_scope=*/cir::SyncScopeKindAttr(),
                         /*mem-order=*/cir::MemOrderAttr());
  }

  setFunction(funcName, makeFunctionSymbol(node, abi));

  DBG("funcName: {0}", funcName);

  return func;
}

auto MLIRGenImpl::gen(const FunctionDecl *node) -> mlir::Operation * {
  resetVariableScopes();
  auto abi =
      useFuncABI(node) ? FunctionSymbol::ABI::Func : FunctionSymbol::ABI::CIR;
  auto func = gen(node->proto().get(), abi);
  if (!func)
    return nullptr;

  auto previousFunctionABI = _currentFunctionABI;
  _currentFunctionABI = llvm::isa<mlir::func::FuncOp>(func)
                            ? FunctionSymbol::ABI::Func
                            : FunctionSymbol::ABI::CIR;

  auto result = genBlock(node->body().get());
  if (failed(result)) {
    _currentFunctionABI = previousFunctionABI;
    return nullptr;
  }
  auto value = *result;

  auto location = loc(node->body()->expression().get());
  if (value) {
    auto *returnType = node->proto()->type();
    if (llvm::isa<mlir::func::FuncOp>(func)) {
      value = castToType(value, getFuncABIType(returnType), location);
      mlir::func::ReturnOp::create(_builder, location, value);
    } else {
      value = castToType(value, getMLIRType(returnType), location);
      llvm::SmallVector<mlir::Value, 1> returnValues{value};
      cir::ReturnOp::create(_builder, location, returnValues);
    }
  } else {
    if (llvm::isa<mlir::func::FuncOp>(func))
      mlir::func::ReturnOp::create(_builder, location);
    else
      cir::ReturnOp::create(_builder, location);
  }

  _currentFunctionABI = previousFunctionABI;
  return func;
}

auto MLIRGenImpl::gen(const StructDecl *node) -> CherryResult {
  if (auto *structType = cherry::getStructType(node->id()->type())) {
    if (containsTensorValue(structType))
      return emitError(node,
                       "struct fields containing Tensor require struct tensor "
                       "storage ABI");

    getMLIRType(structType);
    return success();
  }

  ERR("struct `{0}` has no Cherry type", node->id()->name());
  return failure();
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
  case Expr::Expr_ListLiteral:
    return gen(cast<ListLiteralExpr>(node));
  case Expr::Expr_StructLiteral:
    return gen(cast<StructLiteralExpr>(node));
  case Expr::Expr_Variable:
    return gen(cast<VariableExpr>(node));
  case Expr::Expr_Member:
    return gen(cast<MemberExpr>(node));
  case Expr::Expr_TensorLiteral:
    return gen(cast<TensorLiteralExpr>(node));
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

auto MLIRGenImpl::genBlock(const BlockExpr *node)
    -> llvm::FailureOr<mlir::Value> {
  enterVariableScope();
  for (auto &expr : *node) {
    if (gen(expr.get())) {
      leaveVariableScope();
      return failure();
    }
  }
  auto value = gen(node->expression().get());
  leaveVariableScope();
  return value;
}

auto MLIRGenImpl::gen(const BlockExpr *node) -> mlir::Value {
  auto result = genBlock(node);
  if (failed(result))
    return nullptr;
  return *result;
}

auto MLIRGenImpl::gen(const IfExpr *node) -> mlir::Value {
  DBG("IfExpr Cherry type: {0}, then type: {1}, else type: {2}",
      formatType(node->type()),
      formatType(node->thenBlock()->type()),
      formatType(node->elseBlock()->type()));
  if (isFuncABI() &&
      (isFuncABIScalarExpr(node) || cherry::isUnitType(node->type())))
    return genFuncABIIf(node);

  auto cond = gen(node->conditionExpr().get());

  if (cherry::isUnitType(node->type())) {
    cir::IfOp::create(
        _builder, loc(node), cond,
        /*withElseRegion=*/true,

        /*thenBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          gen(node->thenBlock().get());
          cir::YieldOp::create(b, loc);
        },

        /*elseBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          gen(node->elseBlock().get());
          cir::YieldOp::create(b, loc);
        });
    return nullptr;
  }

  auto resultType = getMLIRType(node);
  auto ptrType = cir::PointerType::get(_builder.getContext(), resultType);
  cir::AllocaOp ifYieldResult =
      cir::AllocaOp::create(_builder, loc(node), ptrType, resultType,
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
  auto conditionExprBuilder = [&](mlir::OpBuilder &builder,
                                  mlir::Location location) {
    auto cond = gen(node->conditionExpr().get());
    cir::ConditionOp::create(builder, loc(node->conditionExpr().get()), cond);
  };

  auto bodyBlock = node->bodyBlock().get();
  auto bodyExprBuilder = [&](mlir::OpBuilder &builder,
                             mlir::Location location) {
    gen(bodyBlock);
    cir::YieldOp::create(builder, loc(bodyBlock->expression().get()));
  };

  cir::WhileOp::create(_builder, loc(node), conditionExprBuilder,
                       bodyExprBuilder);

  return nullptr;
}

auto MLIRGenImpl::genTensorCarriedFor(
    const ForExpr *node, const std::vector<std::string> &carriedNames)
    -> mlir::Value {
  auto forLocation = loc(node);
  auto lowerBound = genIndexValue(node->startExpr().get());
  auto upperBound = genIndexValue(node->endExpr().get());
  auto step = mlir::arith::ConstantIndexOp::create(_builder, forLocation, 1);

  std::vector<mlir::Value> initArgs;
  for (const auto &name : carriedNames) {
    auto value = getVariable(name);
    if (!value || !llvm::isa<mlir::MemRefType>(value.getType())) {
      emitError(node, "tensor loop-carried variable has no memref value");
      return nullptr;
    }
    initArgs.push_back(value);
  }

  // CIR for has no loop-carried results, but tensor variables are MLIR memrefs.
  // Use scf.for only for tensor-carrying loops, so each iteration can yield the
  // next tensor value through iter_args. Normal scalar loops stay on cir.for.
  auto loop = mlir::scf::ForOp::create(
      _builder, forLocation, lowerBound, upperBound, step, initArgs,
      [&](mlir::OpBuilder &builder, mlir::Location location,
          mlir::Value inductionVar, mlir::ValueRange iterArgs) {
        mlir::OpBuilder::InsertionGuard guard(_builder);
        _builder.setInsertionPointToStart(builder.getInsertionBlock());

        enterVariableScope();
        setVariable(node->variableName(), inductionVar);
        for (const auto &value : llvm::enumerate(iterArgs))
          setVariable(carriedNames[value.index()], value.value());

        gen(node->bodyBlock().get());

        std::vector<mlir::Value> yieldedValues;
        for (const auto &name : carriedNames)
          yieldedValues.push_back(getVariable(name));
        mlir::scf::YieldOp::create(_builder, location, yieldedValues);
        leaveVariableScope();
      });

  for (const auto &result : llvm::enumerate(loop.getResults()))
    assignVariable(carriedNames[result.index()], result.value());
  return nullptr;
}

auto MLIRGenImpl::gen(const ForExpr *node) -> mlir::Value {
  if (isFuncABI())
    return genFuncABIFor(node);

  auto carriedNames = getTensorCarriedVariables(node);
  if (!carriedNames.empty())
    return genTensorCarriedFor(node, carriedNames);

  auto forLocation = loc(node);
  auto inductionType = getMLIRType(node->startExpr().get());
  auto inductionPtr = createEntryBlockAlloca(inductionType, forLocation);

  auto startValue =
      castToType(gen(node->startExpr().get()), inductionType, forLocation);
  auto endValue =
      castToType(gen(node->endExpr().get()), inductionType, forLocation);
  auto oneValue =
      cir::ConstantOp::create(_builder, forLocation,
                              cir::IntAttr::get(inductionType, 1));

  cir::StoreOp::create(_builder, forLocation, startValue, inductionPtr,
                       /*isVolatile=*/false,
                       /*alignment=*/mlir::IntegerAttr{},
                       /*sync_scope=*/cir::SyncScopeKindAttr(),
                       /*mem-order=*/cir::MemOrderAttr());

  enterVariableScope();
  setVariable(node->variableName(), inductionPtr);

  auto conditionExprBuilder = [&](mlir::OpBuilder &builder,
                                  mlir::Location location) {
    auto currentValue =
        cir::LoadOp::create(builder, location, inductionPtr);
    auto cond = cir::CmpOp::create(builder, location, cir::CmpOpKind::lt,
                                   currentValue, endValue);
    cir::ConditionOp::create(builder, location, cond);
  };

  auto bodyBlock = node->bodyBlock().get();
  auto bodyExprBuilder = [&](mlir::OpBuilder &builder,
                             mlir::Location location) {
    gen(bodyBlock);
    cir::YieldOp::create(builder, loc(bodyBlock->expression().get()));
  };

  auto stepExprBuilder = [&](mlir::OpBuilder &builder,
                             mlir::Location location) {
    auto currentValue =
        cir::LoadOp::create(builder, location, inductionPtr);
    auto nextValue = cir::BinOp::create(builder, location, cir::BinOpKind::Add,
                                        currentValue, oneValue);
    cir::StoreOp::create(builder, location, nextValue, inductionPtr,
                         /*isVolatile=*/false,
                         /*alignment=*/mlir::IntegerAttr{},
                         /*sync_scope=*/cir::SyncScopeKindAttr(),
                         /*mem-order=*/cir::MemOrderAttr());
    cir::YieldOp::create(builder, location);
  };

  cir::ForOp::create(_builder, forLocation, conditionExprBuilder,
                     bodyExprBuilder, stepExprBuilder);
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
    return isFuncABI() ? genFuncABISize(node) : genSize(node);

  llvm::SmallVector<mlir::Value, 4> operands;
  auto calleeOpIter = findFunction(node->name());
  auto hasCalleeSymbol = calleeOpIter != _functionsByName.end();
  auto calleeSymbol = hasCalleeSymbol ? calleeOpIter->second : FunctionSymbol{};
  auto calleeUsesFuncABI =
      hasCalleeSymbol && calleeSymbol.abi == FunctionSymbol::ABI::Func;
  for (const auto &exprValue : llvm::enumerate(*node)) {
    auto &expr = exprValue.value();
    mlir::Value value;
    if (calleeUsesFuncABI &&
        exprValue.index() < calleeSymbol.parameterTypes.size()) {
      value = genFuncABIArgument(
          expr.get(), calleeSymbol.parameterTypes[exprValue.index()]);
    } else {
      value = gen(expr.get());
    }
    if (!value && !cherry::isUnitType(expr->type()))
      return nullptr;
    DBG("gen(CallExpr). value: {0}", value);
    operands.push_back(value);
  }

  if (name == builtins::boolToUInt64)
    return isFuncABI() ? genFuncABIBoolToUInt64(operands.front(), loc(node))
                       : CastOp::create(_builder, loc(node), operands.front());

  if (!hasCalleeSymbol) {
    // TODO: placeholder for functions implemented after the caller
    ERR("callee {0} DOESN'T exist.", node->name());
    return nullptr;
  }

  auto callee = mlir::SymbolRefAttr::get(_builder.getContext(), name);
  auto isUnitCall = cherry::isUnitType(node->type());
  if (calleeSymbol.abi == FunctionSymbol::ABI::Func) {
    auto callResultType = isUnitCall ? mlir::Type{} : getFuncABIType(node);
    mlir::TypeRange callResultTypes;
    llvm::SmallVector<mlir::Type, 1> nonUnitResultTypes;
    if (!isUnitCall) {
      nonUnitResultTypes.push_back(callResultType);
      callResultTypes = nonUnitResultTypes;
    }

    auto callOp = mlir::func::CallOp::create(
        _builder, loc(node), callee, callResultTypes, operands);
    return isUnitCall ? nullptr : callOp.getResult(0);
  }

  auto callResultType = isUnitCall ? mlir::Type{} : getMLIRType(node);
  auto callOp = cir::CallOp::create(_builder, loc(node), callee, callResultType,
                                    operands);

  return isUnitCall ? nullptr : callOp.getResult();
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
  return cir::LoadOp::create(_builder, loc(node), ptr);
}

auto MLIRGenImpl::gen(const ListLiteralExpr *node) -> mlir::Value {
  auto *listType = cherry::getListType(node->type());
  if (!listType) {
    ERR("list literal has no Cherry list type");
    return nullptr;
  }

  return genListLiteral(node, listType);
}

auto MLIRGenImpl::genPrint(const CallExpr *node) -> mlir::Value {
  auto &expressions = node->expressions();
  auto *expr = expressions.front().get();
  auto operand = llvm::isa<IndexExpr>(expr)
                     ? genMemRefLoadValue(llvm::cast<IndexExpr>(expr))
                     : gen(expr);
  operand = castToType(operand, _builder.getI64Type(), loc(expr));
  return PrintOp::create(_builder, loc(node), operand);
}

auto MLIRGenImpl::genTensorDim(mlir::Location location,
                               TensorDimSource source) -> mlir::Value {
  return mlir::memref::DimOp::create(_builder, location, source.tensor,
                                     source.dim);
}

auto MLIRGenImpl::createTensorAlloc(
    mlir::Location location, mlir::MemRefType memRefType,
    const std::vector<TensorDimSource>& resultDims) -> mlir::Value {
  if (memRefType.getRank() != static_cast<int64_t>(resultDims.size())) {
    ERR("Tensor alloc rank mismatch. type: {0}, dims: {1}",
        memRefType, resultDims.size());
    return nullptr;
  }

  // memref.alloc takes one size operand for each `?` dimension, in result
  // shape order. Static dimensions are encoded in the memref type itself.
  std::vector<mlir::Value> dynamicSizes;
  for (int64_t dim = 0; dim < memRefType.getRank(); ++dim)
    if (memRefType.isDynamicDim(dim))
      dynamicSizes.push_back(genTensorDim(location, resultDims[dim]));

  return mlir::memref::AllocOp::create(_builder, location, memRefType,
                                       dynamicSizes);
}

auto MLIRGenImpl::createSameShapeTensorAlloc(
    mlir::Location location, mlir::MemRefType memRefType,
    mlir::Value source) -> mlir::Value {
  std::vector<TensorDimSource> resultDims;
  for (int64_t dim = 0; dim < memRefType.getRank(); ++dim)
    resultDims.push_back({source, dim});
  return createTensorAlloc(location, memRefType, resultDims);
}

auto MLIRGenImpl::genMatmul(const CallExpr *node) -> mlir::Value {
  auto location = loc(node);
  auto &expressions = node->expressions();
  auto lhs = gen(expressions[0].get());
  auto rhs = gen(expressions[1].get());
  auto outType = getMemRefType(node);
  auto out = createTensorAlloc(location, outType, {{lhs, 0}, {rhs, 1}});
  if (!out)
    return nullptr;
  mlir::cherry_nn::MatmulOp::create(_builder, location, lhs, rhs, out);
  return out;
}

auto MLIRGenImpl::genMatadd(const CallExpr *node) -> mlir::Value {
  auto location = loc(node);
  auto &expressions = node->expressions();
  auto lhs = gen(expressions[0].get());
  auto rhs = gen(expressions[1].get());
  auto outType = getMemRefType(node);
  auto out = createSameShapeTensorAlloc(location, outType, lhs);
  if (!out)
    return nullptr;
  mlir::cherry_nn::MataddOp::create(_builder, location, lhs, rhs, out);
  return out;
}

auto MLIRGenImpl::genTranspose(const CallExpr *node) -> mlir::Value {
  auto location = loc(node);
  auto &expressions = node->expressions();
  auto input = gen(expressions[0].get());
  auto outType = getMemRefType(node);
  auto out = createTensorAlloc(location, outType, {{input, 1}, {input, 0}});
  if (!out)
    return nullptr;
  mlir::cherry_nn::TransposeOp::create(_builder, location, input, out);
  return out;
}

auto MLIRGenImpl::genElementwiseNN(const CallExpr *node) -> mlir::Value {
  auto location = loc(node);
  auto &expressions = node->expressions();
  auto input = gen(expressions[0].get());
  auto outType = getMemRefType(node);
  auto out = createSameShapeTensorAlloc(location, outType, input);
  if (!out)
    return nullptr;

  if (node->name() == nn::exp) {
    mlir::cherry_nn::ExpOp::create(_builder, location, input, out);
    return out;
  }

  mlir::cherry_nn::SigmoidOp::create(_builder, location, input, out);
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
  auto *argument = expressions.front().get();
  if (cherry::isListType(argument->type())) {
    if (auto *variable = llvm::dyn_cast<VariableExpr>(argument)) {
      auto list = getVariable(variable->name());
      if (isMulberryList(list))
        return mlir::mulberry::ListSizeOp::create(_builder, loc(node),
                                                  _builder.getI64Type(), list);
    }

    auto listStoragePtr = genLValue(argument);
    auto lengthType = getMLIRType(node);
    auto lengthPtrType =
        cir::PointerType::get(_builder.getContext(), lengthType);
    mlir::Value lengthPtr = cir::GetMemberOp::create(
        _builder, loc(node), lengthPtrType, listStoragePtr, "length",
        ListStorageLayout::lengthIndex);
    DBG("size() loads Cherry list length");
    return cir::LoadOp::create(_builder, loc(node), lengthPtr);
  }

  auto *tensorType = cherry::getTensorType(argument->type());
  if (!tensorType) {
    ERR("size() argument has no Cherry tensor or list type");
    return nullptr;
  }
  auto size = tensorType->shape().front();
  DBG("size() static tensor size: {0}", size);
  auto type = getMLIRType(node);
  auto attr = cir::IntAttr::get(type, size);
  return cir::ConstantOp::create(_builder, loc(node), attr);
}

auto MLIRGenImpl::gen(const VariableExpr *node) -> mlir::Value {
  auto address = getVariable(node->name());
  if (cherry::isTensorType(node->type()))
    return address;
  if (address && !llvm::isa<cir::PointerType>(address.getType()))
    return address;
  return cir::LoadOp::create(_builder, loc(node), address);
}

auto MLIRGenImpl::gen(const MemberExpr *node) -> mlir::Value {
  auto ptr = genLValue(node);
  return cir::LoadOp::create(_builder, loc(node), ptr);
}

auto MLIRGenImpl::gen(const DecimalLiteralExpr *node) -> mlir::Value {
  if (isFuncABI() && isFuncABIValueType(node->type()))
    return genFuncABIScalarLiteral(node);

  mlir::Type type = getMLIRType(node);
  cir::IntAttr attr = cir::IntAttr::get(type, node->value());
  DBG("type: {0}, attr: {1}", type, attr);
  return cir::ConstantOp::create(_builder, loc(node), attr);
}

auto MLIRGenImpl::gen(const FloatLiteralExpr *node) -> mlir::Value {
  if (isFuncABI() && isFuncABIValueType(node->type()))
    return genFuncABIScalarLiteral(node);

  mlir::Type type = getMLIRType(node);
  cir::FPAttr attr = cir::FPAttr::get(type, node->value());
  DBG("type: {0}, attr: {1}", type, attr);
  return cir::ConstantOp::create(_builder, loc(node), attr);
}

auto MLIRGenImpl::gen(const BoolLiteralExpr *node) -> mlir::Value {
  if (isFuncABI() && isFuncABIValueType(node->type()))
    return genFuncABIScalarLiteral(node);

  cir::BoolAttr attr = cir::BoolAttr::get(_builder.getContext(), node->value());
  DBG("attr: {0}", attr);
  return cir::ConstantOp::create(_builder, loc(node), attr);
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
      auto fieldTy = getMLIRType(field->type());
      DBG("Cherry fieldTy: {0}", fieldTy);
      cir::PointerType fieldPtrTy =
          cir::PointerType::get(_builder.getContext(), fieldTy);
      DBG("Cherry fieldPtrTy: {0}", fieldPtrTy);

      mlir::Value addr = cir::GetMemberOp::create(
          _builder, loc(node), fieldPtrTy, basePtr, field->name(),
          field->index());
      return addr;
    }

    ERR("struct member access has no Cherry field information");
    return nullptr;
  }

  ERR("unknown EXPR");
  exit(-1);

  return nullptr;
}

auto MLIRGenImpl::getListStoragePointer(const IndexExpr *expr)
    -> mlir::Value {
  auto listStoragePtr = getVariable(expr->getVarName());
  if (!listStoragePtr) {
    ERR("List index has no storage pointer: {0}", expr->getVarName());
    return nullptr;
  }

  if (llvm::isa<cir::PointerType>(listStoragePtr.getType()))
    return listStoragePtr;

  ERR("List index lowering expected storage pointer: {0}",
      expr->getVarName());
  return nullptr;
}

auto MLIRGenImpl::genListElementPointer(const IndexExpr *expr) -> mlir::Value {
  auto elementType = getMLIRType(expr->type());
  if (!cir::isSized(elementType)) {
    ERR("List index element lowering is not implemented yet: {0}",
        formatType(expr->type()));
    return nullptr;
  }

  auto listStoragePtr = getListStoragePointer(expr);
  if (!listStoragePtr)
    return nullptr;

  auto dataFieldType = cir::PointerType::get(elementType);
  auto dataFieldPtrType =
      cir::PointerType::get(_builder.getContext(), dataFieldType);
  mlir::Value dataPtrField = cir::GetMemberOp::create(
      _builder, loc(expr), dataFieldPtrType, listStoragePtr, "data",
      ListStorageLayout::dataPtrIndex);
  auto dataPtr = cir::LoadOp::create(_builder, loc(expr), dataPtrField);
  auto index = genCIRIndexValue(expr->getIndices().front().get());
  return cir::PtrStrideOp::create(_builder, loc(expr), dataFieldType,
                                  dataPtr, index);
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
    mlir::Value val = cir::LoadOp::create(_builder, loc(memberExpr), ptr);
    return val;
  }

  return gen(node);
}

auto MLIRGenImpl::gen(const BinaryExpr *node) -> mlir::Value {
  if (isFuncABI() && isFuncABIScalarExpr(node))
    return genFuncABIBinary(node);

  using Operator = BinaryExpr::Operator;
  auto genBoolCmp = [&](Operator op, mlir::Value lhs,
                        mlir::Value rhs) -> mlir::Value {
    auto neq =
        cir::BinOp::create(_builder, loc(node), cir::BinOpKind::Xor, lhs, rhs);
    if (op == Operator::NEQ)
      return neq;

    return cir::UnaryOp::create(_builder, loc(node), neq.getType(),
                                cir::UnaryOpKind::Not, neq);
  };

  auto op = node->opEnum();

  auto lhs =
      castToType(gen(node->lhs().get()),
                 getMLIRType(node->lhs().get()), loc(node->lhs().get()));
  auto rhs =
      castToType(gen(node->rhs().get()),
                 getMLIRType(node->rhs().get()), loc(node->rhs().get()));
  switch (op) {
  case Operator::Add:
    return cir::BinOp::create(_builder, loc(node), cir::BinOpKind::Add, lhs,
                              rhs);
  case Operator::Diff:
    return cir::BinOp::create(_builder, loc(node), cir::BinOpKind::Sub, lhs,
                              rhs);
  case Operator::Mul:
    return cir::BinOp::create(_builder, loc(node), cir::BinOpKind::Mul, lhs,
                              rhs);
  case Operator::Div:
    return cir::BinOp::create(_builder, loc(node), cir::BinOpKind::Div, lhs,
                              rhs);
  case Operator::Rem:
    return cir::BinOp::create(_builder, loc(node), cir::BinOpKind::Rem, lhs,
                              rhs);
  case Operator::And:
    return cir::BinOp::create(_builder, loc(node), cir::BinOpKind::And, lhs,
                              rhs);
  case Operator::Or:
    return cir::BinOp::create(_builder, loc(node), cir::BinOpKind::Or, lhs,
                              rhs);
  case Operator::EQ:
    if (llvm::isa<cir::BoolType>(lhs.getType()))
      return genBoolCmp(op, lhs, rhs);

    return cir::CmpOp::create(_builder, loc(node), cir::CmpOpKind::eq, lhs,
                              rhs);
  case Operator::NEQ:
    if (llvm::isa<cir::BoolType>(lhs.getType()))
      return genBoolCmp(op, lhs, rhs);

    return cir::CmpOp::create(_builder, loc(node), cir::CmpOpKind::ne, lhs,
                              rhs);
  case Operator::LT:
    return cir::CmpOp::create(_builder, loc(node), cir::CmpOpKind::lt, lhs,
                              rhs);
  case Operator::LE:
    return cir::CmpOp::create(_builder, loc(node), cir::CmpOpKind::le, lhs,
                              rhs);
  case Operator::GT:
    return cir::CmpOp::create(_builder, loc(node), cir::CmpOpKind::gt, lhs,
                              rhs);
  case Operator::GE:
    return cir::CmpOp::create(_builder, loc(node), cir::CmpOpKind::ge, lhs,
                              rhs);
  }

  llvm_unreachable("Unexpected statement");
}

auto MLIRGenImpl::gen(const AssignExpr *node) -> mlir::Value {
  llvm::TypeSwitch<const Expr *>(node->lhs().get())
      .Case<VariableExpr>([&](const auto *var) {
        auto name = var->name();
        auto rhs = gen(node->rhs().get());
        if (cherry::isTensorType(var->type())) {
          rhs = castToType(rhs, getMemRefType(var), loc(node));
          assignVariable(name, rhs);
          return;
        }

        if (isFuncABI() && isFuncABIValueType(var->type())) {
          rhs = castToType(rhs, getFuncABIType(var->type()), loc(node));
          assignVariable(name, rhs);
          return;
        }

        auto address = getVariable(name);
        if (!cherry::isUnitType(node->lhs()->type())) {
          rhs =
              castToType(rhs, getMLIRType(node->lhs().get()), loc(node));
          cir::StoreOp::create(_builder, loc(node), rhs, address,
                               /*isVolatile=*/false,
                               /*alignment=*/mlir::IntegerAttr{},
                               /*sync_scope=*/cir::SyncScopeKindAttr(),
                               /*mem-order=*/cir::MemOrderAttr());
        }
      })
      .Case<MemberExpr>([&](const auto *member) {
        mlir::Value lhsPtr = genLValue(member);
        auto rhs = castToType(gen(node->rhs().get()),
                              getMLIRType(member), loc(node));

        cir::StoreOp::create(_builder, loc(node), rhs, lhsPtr,
                             /*isVolatile=*/false,
                             /*alignment=*/mlir::IntegerAttr{},
                             /*sync_scope=*/cir::SyncScopeKindAttr(),
                             /*mem-order=*/cir::MemOrderAttr());
      })
      .Case<IndexExpr>([&](const auto *indexExpr) {
        genAssignment(indexExpr, node->rhs().get());
      })
      .Default(
          [&](const Expr *) { llvm_unreachable("Unexpected expression"); });
  return nullptr;
}

auto MLIRGenImpl::gen(const Stat *node) -> CherryResult {
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

auto MLIRGenImpl::genStructLiteral(const StructLiteralExpr *structLiteral,
                                   const StructType *structType,
                                   mlir::Value targetPtr) -> mlir::Value {
  DBG("structType: {0}, structLiteral->name(): {1}",
      formatType(structType),
      structLiteral->name());

  auto recordType =
      llvm::dyn_cast<cir::RecordType>(getMLIRType(structType));
  if (!recordType) {
    ERR("NOT a RecordType. structType: {0}",
        formatType(structType));
    return nullptr;
  }
  if (recordType.getKind() != cir::RecordType::RecordKind::Struct) {
    ERR("record type kind: {0}", recordType.getKind());
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

  mlir::Type recordPtrTy = cir::PointerType::get(recordType);

  mlir::Value varPtrOp;
  if (!targetPtr) {
    auto alloca = cir::AllocaOp::create(_builder, loc(structLiteral),
                                        recordPtrTy, recordType,
                                        structType->name(), getAlignOne());
    DBG("structType: {0}, alloca: {1}", formatType(structType),
        alloca);

    auto *parentBlock = alloca->getBlock();
    alloca->moveBefore(&parentBlock->front());

    varPtrOp = alloca;
  } else {
    varPtrOp = llvm::dyn_cast<cir::GetMemberOp>(targetPtr.getDefiningOp());
  }

  unsigned index = 0;
  for (auto &expr : *structLiteral) {
    auto &field = fields[index];
    mlir::Type fieldTy = getMLIRType(field.type());
    cir::PointerType fieldPtrTy =
        cir::PointerType::get(_builder.getContext(), fieldTy);

    DBG("Cherry struct literal field index: {0}, field name: {1}, "
        "expr type: {2}",
        field.index(), field.name(),
        formatType(expr->type()));
    auto memberPtr = cir::GetMemberOp::create(
        _builder, loc(structLiteral), fieldPtrTy, varPtrOp, field.name(),
        field.index());

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
      mlir::Value val = gen(expr.get());
      val = castToType(val, fieldTy, loc(expr.get()));
      cir::StoreOp::create(_builder, loc(structLiteral), val, memberPtr,
                           /*isVolatile=*/false,
                           /*alignment=*/mlir::IntegerAttr{},
                           /*sync_scope=*/cir::SyncScopeKindAttr(),
                           /*mem-order=*/cir::MemOrderAttr());
    }

    index++;
  }

  return varPtrOp;
}

auto MLIRGenImpl::getListStorageElementType(const ListType *listType)
    -> mlir::Type {
  return _typeConverter.convertListStorageElement(listType->elementType());
}

auto MLIRGenImpl::createListStorage(mlir::Location location,
                                    mlir::Type listStorageType,
                                    mlir::Type elementType,
                                    mlir::Value dataPtr, size_t length)
    -> mlir::Value {
  auto listStoragePtrType = cir::PointerType::get(listStorageType);
  auto listStorage = cir::AllocaOp::create(_builder, location,
                                           listStoragePtrType,
                                           listStorageType, "list",
                                           getAlignOne());
  auto *parentBlock = listStorage->getBlock();
  listStorage->moveBefore(&parentBlock->front());
  mlir::Value listStoragePtr = listStorage.getAddr();

  auto indexType =
      cir::IntType::get(_builder.getContext(), 64, /*isSigned=*/false);
  auto lengthPtrType =
      cir::PointerType::get(_builder.getContext(), indexType);
  mlir::Value lengthPtr = cir::GetMemberOp::create(
      _builder, location, lengthPtrType, listStoragePtr, "length",
      ListStorageLayout::lengthIndex);
  auto lengthValue =
      cir::ConstantOp::create(_builder, location,
                              cir::IntAttr::get(indexType, length));
  cir::StoreOp::create(_builder, location, lengthValue, lengthPtr,
                       /*isVolatile=*/false,
                       /*alignment=*/mlir::IntegerAttr{},
                       /*sync_scope=*/cir::SyncScopeKindAttr(),
                       /*mem-order=*/cir::MemOrderAttr());

  auto dataFieldType = cir::PointerType::get(elementType);
  auto dataFieldPtrType =
      cir::PointerType::get(_builder.getContext(), dataFieldType);
  mlir::Value dataPtrField = cir::GetMemberOp::create(
      _builder, location, dataFieldPtrType, listStoragePtr, "data",
      ListStorageLayout::dataPtrIndex);
  cir::StoreOp::create(_builder, location, dataPtr, dataPtrField,
                       /*isVolatile=*/false,
                       /*alignment=*/mlir::IntegerAttr{},
                       /*sync_scope=*/cir::SyncScopeKindAttr(),
                       /*mem-order=*/cir::MemOrderAttr());

  return cir::LoadOp::create(_builder, location, listStoragePtr);
}

void MLIRGenImpl::storeListElement(mlir::Location location,
                                   mlir::Value dataPtr,
                                   mlir::Type elementType, size_t index,
                                   mlir::Value value) {
  auto indexType =
      cir::IntType::get(_builder.getContext(), 64, /*isSigned=*/false);
  auto indexValue =
      cir::ConstantOp::create(_builder, location,
                              cir::IntAttr::get(indexType, index));
  auto elementPtrType = cir::PointerType::get(elementType);
  auto elementPtr =
      cir::PtrStrideOp::create(_builder, location, elementPtrType,
                               dataPtr, indexValue);
  cir::StoreOp::create(_builder, location, value, elementPtr,
                       /*isVolatile=*/false,
                       /*alignment=*/mlir::IntegerAttr{},
                       /*sync_scope=*/cir::SyncScopeKindAttr(),
                       /*mem-order=*/cir::MemOrderAttr());
}

auto MLIRGenImpl::genTensorListLiteral(const ListLiteralExpr *listLiteral,
                                       const ListType *listType)
    -> mlir::Value {
  auto *elementType = cherry::getTensorType(listType->elementType());
  if (!elementType)
    return {};

  std::vector<mlir::Value> values;
  for (const auto &element : listLiteral->elements()) {
    auto value = castToType(gen(element.get()), getMemRefType(elementType),
                            loc(element.get()));
    if (!value)
      return {};
    auto memRefType = llvm::dyn_cast<mlir::MemRefType>(value.getType());
    if (!memRefType) {
      emitError(element.get(), "List<Tensor> element lowering expected memref");
      return {};
    }

    // Keep List<Tensor> as a list of tensor descriptors in Mulberry IR.
    // Lowering may temporarily unpack these descriptors back to memrefs, but
    // the list layer should not directly traffic in raw MLIR memref values.
    auto descriptorType =
        mlir::mulberry::TensorDescriptorType::get(_builder.getContext(),
                                                  memRefType);
    auto descriptor = mlir::mulberry::TensorPackOp::create(
        _builder, loc(element.get()), descriptorType, value);
    values.push_back(descriptor);
  }

  auto mlirElementType =
      mlir::mulberry::TensorDescriptorType::get(_builder.getContext(),
                                                getMemRefType(elementType));
  auto listMLIRType =
      mlir::mulberry::ListType::get(_builder.getContext(), mlirElementType);
  return mlir::mulberry::ListCreateOp::create(_builder, loc(listLiteral),
                                              listMLIRType, values);
}

auto MLIRGenImpl::getStaticIndex(const Expr *expr) -> std::optional<uint64_t> {
  if (auto *decimal = llvm::dyn_cast<DecimalLiteralExpr>(expr))
    return decimal->value();
  return std::nullopt;
}

void MLIRGenImpl::collectAssignedVariables(
    const Expr *expr, std::set<std::string> &names,
    const std::function<bool(const Type *)> &shouldCollect) {
  if (auto *assign = llvm::dyn_cast<AssignExpr>(expr)) {
    if (auto *var = llvm::dyn_cast<VariableExpr>(assign->lhs().get()))
      if (shouldCollect(var->type()))
        names.insert(std::string(var->name()));
    collectAssignedVariables(assign->rhs().get(), names, shouldCollect);
    return;
  }

  if (auto *call = llvm::dyn_cast<CallExpr>(expr)) {
    for (const auto &arg : call->expressions())
      collectAssignedVariables(arg.get(), names, shouldCollect);
    return;
  }

  if (auto *block = llvm::dyn_cast<BlockExpr>(expr)) {
    for (const auto &stat : *block)
      collectAssignedVariables(stat.get(), names, shouldCollect);
    collectAssignedVariables(block->expression().get(), names, shouldCollect);
    return;
  }

  if (auto *ifExpr = llvm::dyn_cast<IfExpr>(expr)) {
    collectAssignedVariables(ifExpr->conditionExpr().get(), names,
                             shouldCollect);
    collectAssignedVariables(ifExpr->thenBlock().get(), names, shouldCollect);
    collectAssignedVariables(ifExpr->elseBlock().get(), names, shouldCollect);
    return;
  }

  if (auto *whileExpr = llvm::dyn_cast<WhileExpr>(expr)) {
    collectAssignedVariables(whileExpr->conditionExpr().get(), names,
                             shouldCollect);
    collectAssignedVariables(whileExpr->bodyBlock().get(), names,
                             shouldCollect);
    return;
  }

  if (auto *forExpr = llvm::dyn_cast<ForExpr>(expr)) {
    collectAssignedVariables(forExpr->startExpr().get(), names, shouldCollect);
    collectAssignedVariables(forExpr->endExpr().get(), names, shouldCollect);
    collectAssignedVariables(forExpr->bodyBlock().get(), names, shouldCollect);
    return;
  }

  if (auto *binary = llvm::dyn_cast<BinaryExpr>(expr)) {
    collectAssignedVariables(binary->lhs().get(), names, shouldCollect);
    collectAssignedVariables(binary->rhs().get(), names, shouldCollect);
    return;
  }

  if (auto *member = llvm::dyn_cast<MemberExpr>(expr)) {
    collectAssignedVariables(member->base().get(), names, shouldCollect);
    return;
  }

  if (auto *index = llvm::dyn_cast<IndexExpr>(expr)) {
    for (const auto &indexExpr : index->getIndices())
      collectAssignedVariables(indexExpr.get(), names, shouldCollect);
    return;
  }

  if (auto *list = llvm::dyn_cast<ListLiteralExpr>(expr)) {
    for (const auto &element : list->elements())
      collectAssignedVariables(element.get(), names, shouldCollect);
    return;
  }

  if (auto *tensor = llvm::dyn_cast<TensorLiteralExpr>(expr)) {
    for (const auto &element : tensor->getElements())
      collectAssignedVariables(element.get(), names, shouldCollect);
    return;
  }

  if (auto *structLiteral = llvm::dyn_cast<StructLiteralExpr>(expr)) {
    for (const auto &field : *structLiteral)
      collectAssignedVariables(field.get(), names, shouldCollect);
  }
}

void MLIRGenImpl::collectAssignedVariables(
    const Stat *stat, std::set<std::string> &names,
    const std::function<bool(const Type *)> &shouldCollect) {
  if (auto *exprStat = llvm::dyn_cast<ExprStat>(stat))
    collectAssignedVariables(exprStat->expression().get(), names,
                             shouldCollect);
  if (auto *varStat = llvm::dyn_cast<VariableStat>(stat))
    collectAssignedVariables(varStat->init().get(), names, shouldCollect);
}

void MLIRGenImpl::collectAssignedTensorVariables(
    const Expr *expr, std::set<std::string> &names) {
  collectAssignedVariables(expr, names,
                           [](const Type *type) {
                             return cherry::isTensorType(type);
                           });
}

void MLIRGenImpl::collectAssignedTensorVariables(
    const Stat *stat, std::set<std::string> &names) {
  collectAssignedVariables(stat, names,
                           [](const Type *type) {
                             return cherry::isTensorType(type);
                           });
}

auto MLIRGenImpl::getTensorCarriedVariables(const ForExpr *node)
    -> std::vector<std::string> {
  std::set<std::string> names;
  for (const auto &stat : *node->bodyBlock())
    collectAssignedTensorVariables(stat.get(), names);
  collectAssignedTensorVariables(node->bodyBlock()->expression().get(), names);

  std::vector<std::string> carriedNames;
  for (const auto &name : names)
    if (getVariable(name))
      carriedNames.push_back(name);
  return carriedNames;
}

auto MLIRGenImpl::getFuncABICarriedVariables(const ForExpr *node)
    -> std::vector<std::string> {
  std::set<std::string> names;
  auto shouldCollect = [&](const Type *type) {
    return isFuncABIValueType(type);
  };
  for (const auto &stat : *node->bodyBlock())
    collectAssignedVariables(stat.get(), names, shouldCollect);
  collectAssignedVariables(node->bodyBlock()->expression().get(), names,
                           shouldCollect);

  std::vector<std::string> carriedNames;
  for (const auto &name : names)
    if (getVariable(name))
      carriedNames.push_back(name);
  return carriedNames;
}

auto MLIRGenImpl::genTensorListIndex(const IndexExpr *expr,
                                     mlir::Value tensorList)
    -> mlir::Value {
  auto resultType = getMemRefType(expr);
  auto &indices = expr->getIndices();
  auto indexValue = genIndexValue(indices.front().get());
  auto listElementType = getMulberryListElementType(tensorList);
  if (!listElementType) {
    emitError(expr, "List<Tensor> index lowering expected mulberry.list");
    return nullptr;
  }

  auto element = mlir::mulberry::ListGetOp::create(
      _builder, loc(expr), listElementType, tensorList, indexValue);
  auto tensor = mlir::mulberry::TensorUnpackOp::create(
      _builder, loc(expr), resultType, element);
  return castToType(tensor, resultType, loc(expr));
}

auto MLIRGenImpl::genListLiteral(const ListLiteralExpr *listLiteral,
                                 const ListType *listType) -> mlir::Value {
  auto &elements = listLiteral->elements();
  if (!canMaterializeListStorage(listType)) {
    if (isTensorList(listType))
      return genTensorListLiteral(listLiteral, listType);

    // TODO: lower nested List<Tensor> after mulberry.list supports nested list
    // values. Generic CIR list storage still cannot hold tensor descriptors.
    emitError(listLiteral, "nested List<Tensor> lowering is not implemented");
    return nullptr;
  }

  auto elementType = getListStorageElementType(listType);
  if (!elementType)
    return nullptr;
  if (!cir::isSized(elementType)) {
    emitError(listLiteral, "unsupported list element type lowering");
    return nullptr;
  }

  auto elementStorageType = cir::ArrayType::get(elementType, elements.size());
  auto elementStoragePtrType = cir::PointerType::get(elementStorageType);
  auto elementStorage = cir::AllocaOp::create(
      _builder, loc(listLiteral), elementStoragePtrType, elementStorageType,
      "list_elements", getAlignOne());

  auto *parentBlock = elementStorage->getBlock();
  elementStorage->moveBefore(&parentBlock->front());
  mlir::Value elementStoragePtr = elementStorage.getAddr();

  auto elementPtrType = cir::PointerType::get(elementType);
  auto dataPtr = cir::CastOp::create(_builder, loc(listLiteral),
                                     elementPtrType,
                                     cir::CastKind::array_to_ptrdecay,
                                     elementStoragePtr);

  for (size_t i = 0; i < elements.size(); ++i) {
    auto value = castToType(gen(elements[i].get()), elementType,
                            loc(elements[i].get()));
    storeListElement(loc(elements[i].get()), dataPtr, elementType, i, value);
  }

  auto listStorageType =
      llvm::cast<cir::RecordType>(getMLIRType(listType));
  return createListStorage(loc(listLiteral), listStorageType, elementType,
                           dataPtr, elements.size());
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

auto MLIRGenImpl::getFuncABIType(const Type *type) const -> mlir::Type {
  // func.func can directly carry MLIR memrefs, so keep its ABI in the MLIR
  // scalar world instead of exposing CIR scalar types at that boundary.
  if (cherry::isUInt64Type(type))
    return mlir::IntegerType::get(_builder.getContext(), 64);
  if (cherry::isFloat32Type(type))
    return mlir::Float32Type::get(_builder.getContext());
  if (cherry::isBoolType(type))
    return mlir::IntegerType::get(_builder.getContext(), 1);
  if (cherry::isUnitType(type))
    return {};

  return getMLIRType(type);
}

auto MLIRGenImpl::getFuncABIType(const Expr *expr) const -> mlir::Type {
  return getFuncABIType(expr->type());
}

auto MLIRGenImpl::isMulberryList(mlir::Value value) -> bool {
  return value &&
         llvm::isa<mlir::mulberry::ListType>(value.getType());
}

auto MLIRGenImpl::getMulberryListElementType(mlir::Value value)
    -> mlir::Type {
  if (!isMulberryList(value))
    return {};
  return llvm::cast<mlir::mulberry::ListType>(value.getType())
      .getElementType();
}

auto MLIRGenImpl::isTensorList(const ListType *type) const -> bool {
  return type && cherry::isTensorType(type->elementType());
}

auto MLIRGenImpl::containsTensorValue(const Type *type) const -> bool {
  if (cherry::isTensorType(type))
    return true;

  if (auto *listType = cherry::getListType(type)) {
    auto *elementType = listType->elementType();
    return containsTensorValue(elementType);
  }

  if (auto *structType = cherry::getStructType(type)) {
    for (const auto& field : structType->fields())
      if (containsTensorValue(field.type()))
        return true;
  }

  return false;
}

auto MLIRGenImpl::containsTensorValue(const BlockExpr *block) const -> bool {
  if (!block)
    return false;

  if (containsTensorValue(block->type()))
    return true;

  for (const auto& stat : block->statements())
    if (containsTensorValue(stat.get()))
      return true;

  return containsTensorValue(block->expression().get());
}

auto MLIRGenImpl::containsTensorValue(const Expr *expr) const -> bool {
  if (!expr)
    return false;

  if (containsTensorValue(expr->type()))
    return true;

  switch (expr->getKind()) {
  case Expr::Expr_Unit:
  case Expr::Expr_DecimalLiteral:
  case Expr::Expr_FloatLiteral:
  case Expr::Expr_BoolLiteral:
  case Expr::Expr_Variable:
    return false;
  case Expr::Expr_Call: {
    auto *call = llvm::cast<CallExpr>(expr);
    for (const auto& argument : call->expressions())
      if (containsTensorValue(argument.get()))
        return true;
    return false;
  }
  case Expr::Expr_StructLiteral: {
    auto *literal = llvm::cast<StructLiteralExpr>(expr);
    for (const auto& field : literal->expressions())
      if (containsTensorValue(field.get()))
        return true;
    return false;
  }
  case Expr::Expr_ListLiteral: {
    auto *literal = llvm::cast<ListLiteralExpr>(expr);
    for (const auto& element : literal->elements())
      if (containsTensorValue(element.get()))
        return true;
    return false;
  }
  case Expr::Expr_TensorLiteral:
    return true;
  case Expr::Expr_Index: {
    auto *index = llvm::cast<IndexExpr>(expr);
    for (const auto& idx : index->getIndices())
      if (containsTensorValue(idx.get()))
        return true;
    return false;
  }
  case Expr::Expr_Member:
    return containsTensorValue(llvm::cast<MemberExpr>(expr)->base().get());
  case Expr::Expr_Assign: {
    auto *assign = llvm::cast<AssignExpr>(expr);
    return containsTensorValue(assign->lhs().get()) ||
           containsTensorValue(assign->rhs().get());
  }
  case Expr::Expr_Binary: {
    auto *binary = llvm::cast<BinaryExpr>(expr);
    return containsTensorValue(binary->lhs().get()) ||
           containsTensorValue(binary->rhs().get());
  }
  case Expr::Expr_Block: {
    return containsTensorValue(llvm::cast<BlockExpr>(expr));
  }
  case Expr::Expr_If: {
    auto *ifExpr = llvm::cast<IfExpr>(expr);
    return containsTensorValue(ifExpr->conditionExpr().get()) ||
           containsTensorValue(ifExpr->thenBlock().get()) ||
           containsTensorValue(ifExpr->elseBlock().get());
  }
  case Expr::Expr_While: {
    auto *whileExpr = llvm::cast<WhileExpr>(expr);
    return containsTensorValue(whileExpr->conditionExpr().get()) ||
           containsTensorValue(whileExpr->bodyBlock().get());
  }
  case Expr::Expr_For: {
    auto *forExpr = llvm::cast<ForExpr>(expr);
    return containsTensorValue(forExpr->startExpr().get()) ||
           containsTensorValue(forExpr->endExpr().get()) ||
           containsTensorValue(forExpr->bodyBlock().get());
  }
  }
  llvm_unreachable("Unexpected expression kind");
}

auto MLIRGenImpl::containsTensorValue(const Stat *stat) const -> bool {
  if (auto *variable = llvm::dyn_cast<VariableStat>(stat))
    return containsTensorValue(variable->type()) ||
           containsTensorValue(variable->init().get());

  auto *exprStat = llvm::cast<ExprStat>(stat);
  return containsTensorValue(exprStat->expression().get());
}

auto MLIRGenImpl::useFuncABI(const Prototype *node) const -> bool {
  if (cherry::isTensorType(node->type()))
    return true;

  for (const auto& param : node->parameters())
    if (cherry::isTensorType(param->type()))
      return true;

  return false;
}

auto MLIRGenImpl::useFuncABI(const FunctionDecl *node) const -> bool {
  if (useFuncABI(node->proto().get()))
    return true;

  // Tensor operations use MLIR memrefs and standard MLIR scalars. Keeping a
  // Tensor-using function in CIR forces mixed CIR/func scalar bridge values at
  // func.call sites, which does not lower cleanly before ClangIR lowering.
  auto usesTensorBody = containsTensorValue(node->body().get());
  DBG("function `{0}` uses Tensor body: {1}", node->proto()->id()->name(),
      usesTensorBody);
  return usesTensorBody;
}

auto MLIRGenImpl::isFuncABI() const -> bool {
  return _currentFunctionABI == FunctionSymbol::ABI::Func;
}

auto MLIRGenImpl::isFuncABIScalarType(const Type *type) const -> bool {
  return cherry::isUInt64Type(type) || cherry::isFloat32Type(type) ||
         cherry::isBoolType(type);
}

auto MLIRGenImpl::isFuncABIValueType(const Type *type) const -> bool {
  return cherry::isTensorType(type) || isFuncABIScalarType(type) ||
         cherry::isUnitType(type);
}

auto MLIRGenImpl::isFuncABIScalarExpr(const Expr *expr) const -> bool {
  return expr && isFuncABIScalarType(expr->type());
}

auto MLIRGenImpl::isSupportedFuncABIParameterType(const Type *type) const
    -> bool {
  return isFuncABIValueType(type);
}

auto MLIRGenImpl::isSupportedFuncABIReturnType(const Type *type) const
    -> bool {
  return isFuncABIValueType(type);
}

auto MLIRGenImpl::canMaterializeListStorage(const ListType *type) const
    -> bool {
  // Generic CIR list storage is {length, dataPtr}. It works for sized CIR
  // element values. Tensor values lower to raw MLIR memrefs, so List elements
  // containing tensors need descriptor/reference storage instead.
  return type && !containsTensorValue(type->elementType());
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

auto MLIRGenImpl::getFunctionReturnType(const Type *returnType) const
    -> mlir::Type {
  if (cherry::isUnitType(returnType))
    return cir::VoidType::get(_builder.getContext());
  return getMLIRType(returnType);
}

auto MLIRGenImpl::genFuncABIIf(const IfExpr *node) -> mlir::Value {
  auto condition = castToType(gen(node->conditionExpr().get()),
                              getFuncABIType(node->conditionExpr().get()),
                              loc(node->conditionExpr().get()));
  auto isUnit = cherry::isUnitType(node->type());
  auto resultType = getFuncABIType(node);
  mlir::TypeRange resultTypes;
  llvm::SmallVector<mlir::Type, 1> nonUnitResultTypes;
  if (!isUnit) {
    nonUnitResultTypes.push_back(resultType);
    resultTypes = nonUnitResultTypes;
  }

  // func.func bodies live in the standard MLIR scalar world. Use scf.if's SSA
  // result instead of CIR's alloca/store/load pattern.
  auto ifOp = mlir::scf::IfOp::create(
      _builder, loc(node), resultTypes, condition, /*withElseRegion=*/true);

  {
    mlir::OpBuilder::InsertionGuard guard(_builder);
    _builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    if (isUnit) {
      gen(node->thenBlock().get());
    } else {
      auto thenValue =
          castToType(gen(node->thenBlock().get()), resultType, loc(node));
      mlir::scf::YieldOp::create(_builder, loc(node->thenBlock().get()),
                                 thenValue);
    }
  }

  {
    mlir::OpBuilder::InsertionGuard guard(_builder);
    _builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    if (isUnit) {
      gen(node->elseBlock().get());
    } else {
      auto elseValue =
          castToType(gen(node->elseBlock().get()), resultType, loc(node));
      mlir::scf::YieldOp::create(_builder, loc(node->elseBlock().get()),
                                 elseValue);
    }
  }

  return isUnit ? nullptr : ifOp.getResult(0);
}

auto MLIRGenImpl::genFuncABIFor(const ForExpr *node) -> mlir::Value {
  auto forLocation = loc(node);
  auto lowerBound = genIndexValue(node->startExpr().get());
  auto upperBound = genIndexValue(node->endExpr().get());
  auto step = mlir::arith::ConstantIndexOp::create(_builder, forLocation, 1);

  auto carriedNames = getFuncABICarriedVariables(node);
  std::vector<mlir::Value> initArgs;
  for (const auto &name : carriedNames) {
    auto value = getVariable(name);
    if (!value) {
      emitError(node, "Func ABI loop-carried variable has no SSA value");
      return nullptr;
    }
    initArgs.push_back(value);
  }

  // Func ABI code is pure SSA. Any assignment to an outer variable inside
  // scf.for must flow through iter_args; otherwise the loop body would leak a
  // region-local SSA value to users after the loop.
  auto loop = mlir::scf::ForOp::create(
      _builder, forLocation, lowerBound, upperBound, step, initArgs,
      [&](mlir::OpBuilder &builder, mlir::Location location,
          mlir::Value inductionVar, mlir::ValueRange iterArgs) {
        mlir::OpBuilder::InsertionGuard guard(_builder);
        _builder.setInsertionPointToStart(builder.getInsertionBlock());

        enterVariableScope();
        setVariable(node->variableName(), inductionVar);
        for (const auto &value : llvm::enumerate(iterArgs))
          setVariable(carriedNames[value.index()], value.value());

        gen(node->bodyBlock().get());

        std::vector<mlir::Value> yieldedValues;
        for (const auto &name : carriedNames)
          yieldedValues.push_back(getVariable(name));
        mlir::scf::YieldOp::create(_builder, location, yieldedValues);
        leaveVariableScope();
      });

  for (const auto &result : llvm::enumerate(loop.getResults()))
    assignVariable(carriedNames[result.index()], result.value());
  return nullptr;
}

auto MLIRGenImpl::genFuncABIBinary(const BinaryExpr *node) -> mlir::Value {
  using Operator = BinaryExpr::Operator;
  auto op = node->opEnum();
  auto lhsType = getFuncABIType(node->lhs().get());
  auto rhsType = getFuncABIType(node->rhs().get());
  auto lhs = castToType(gen(node->lhs().get()), lhsType,
                        loc(node->lhs().get()));
  auto rhs = castToType(gen(node->rhs().get()), rhsType,
                        loc(node->rhs().get()));
  auto location = loc(node);
  auto isFloat = llvm::isa<mlir::FloatType>(lhs.getType());

  switch (op) {
  case Operator::Add:
    if (isFloat)
      return mlir::arith::AddFOp::create(_builder, location, lhs, rhs);
    return mlir::arith::AddIOp::create(_builder, location, lhs, rhs);
  case Operator::Diff:
    if (isFloat)
      return mlir::arith::SubFOp::create(_builder, location, lhs, rhs);
    return mlir::arith::SubIOp::create(_builder, location, lhs, rhs);
  case Operator::Mul:
    if (isFloat)
      return mlir::arith::MulFOp::create(_builder, location, lhs, rhs);
    return mlir::arith::MulIOp::create(_builder, location, lhs, rhs);
  case Operator::Div:
    if (isFloat)
      return mlir::arith::DivFOp::create(_builder, location, lhs, rhs);
    return mlir::arith::DivUIOp::create(_builder, location, lhs, rhs);
  case Operator::Rem:
    return mlir::arith::RemUIOp::create(_builder, location, lhs, rhs);
  case Operator::And:
    return mlir::arith::AndIOp::create(_builder, location, lhs, rhs);
  case Operator::Or:
    return mlir::arith::OrIOp::create(_builder, location, lhs, rhs);
  case Operator::EQ:
    if (isFloat)
      return mlir::arith::CmpFOp::create(
          _builder, location, mlir::arith::CmpFPredicate::OEQ, lhs, rhs);
    return mlir::arith::CmpIOp::create(
        _builder, location, mlir::arith::CmpIPredicate::eq, lhs, rhs);
  case Operator::NEQ:
    if (isFloat)
      return mlir::arith::CmpFOp::create(
          _builder, location, mlir::arith::CmpFPredicate::ONE, lhs, rhs);
    return mlir::arith::CmpIOp::create(
        _builder, location, mlir::arith::CmpIPredicate::ne, lhs, rhs);
  case Operator::LT:
    if (isFloat)
      return mlir::arith::CmpFOp::create(
          _builder, location, mlir::arith::CmpFPredicate::OLT, lhs, rhs);
    return mlir::arith::CmpIOp::create(
        _builder, location, mlir::arith::CmpIPredicate::ult, lhs, rhs);
  case Operator::LE:
    if (isFloat)
      return mlir::arith::CmpFOp::create(
          _builder, location, mlir::arith::CmpFPredicate::OLE, lhs, rhs);
    return mlir::arith::CmpIOp::create(
        _builder, location, mlir::arith::CmpIPredicate::ule, lhs, rhs);
  case Operator::GT:
    if (isFloat)
      return mlir::arith::CmpFOp::create(
          _builder, location, mlir::arith::CmpFPredicate::OGT, lhs, rhs);
    return mlir::arith::CmpIOp::create(
        _builder, location, mlir::arith::CmpIPredicate::ugt, lhs, rhs);
  case Operator::GE:
    if (isFloat)
      return mlir::arith::CmpFOp::create(
          _builder, location, mlir::arith::CmpFPredicate::OGE, lhs, rhs);
    return mlir::arith::CmpIOp::create(
        _builder, location, mlir::arith::CmpIPredicate::uge, lhs, rhs);
  }

  llvm_unreachable("Unexpected BinaryExpr operator");
}

auto MLIRGenImpl::genFuncABIBoolToUInt64(mlir::Value operand,
                                         mlir::Location location)
    -> mlir::Value {
  return mlir::arith::ExtUIOp::create(_builder, location, _builder.getI64Type(),
                                      operand);
}

auto MLIRGenImpl::genFuncABISize(const CallExpr *node) -> mlir::Value {
  auto &expressions = node->expressions();
  auto *argument = expressions.front().get();
  auto *tensorType = cherry::getTensorType(argument->type());
  if (!tensorType)
    return genSize(node);

  auto size = tensorType->shape().front();
  DBG("size() static tensor size in func ABI: {0}", size);
  return mlir::arith::ConstantIntOp::create(_builder, loc(node), size, 64);
}

auto MLIRGenImpl::genFuncABIArgument(const Expr *expr, const Type *paramType)
    -> mlir::Value {
  if (cherry::isTensorType(paramType))
    return castToType(gen(expr), getFuncABIType(paramType), loc(expr));

  if (!isFuncABIScalarType(paramType))
    return castToType(gen(expr), getFuncABIType(paramType), loc(expr));

  if (auto *decimal = llvm::dyn_cast<DecimalLiteralExpr>(expr))
    return genFuncABIScalarLiteral(decimal);
  if (auto *floating = llvm::dyn_cast<FloatLiteralExpr>(expr))
    return genFuncABIScalarLiteral(floating);
  if (auto *boolean = llvm::dyn_cast<BoolLiteralExpr>(expr))
    return genFuncABIScalarLiteral(boolean);

  auto value = gen(expr);
  auto targetType = getFuncABIType(paramType);
  auto converted = castToType(value, targetType, loc(expr));
  if (!converted)
    emitError(expr, "cannot convert scalar argument to Tensor function ABI");
  return converted;
}

auto MLIRGenImpl::genFuncABIScalarLiteral(const DecimalLiteralExpr *node)
    -> mlir::Value {
  auto intType = llvm::cast<mlir::IntegerType>(getFuncABIType(node));
  return mlir::arith::ConstantIntOp::create(
      _builder, loc(node), node->value(), intType.getWidth());
}

auto MLIRGenImpl::genFuncABIScalarLiteral(const FloatLiteralExpr *node)
    -> mlir::Value {
  return mlir::arith::ConstantFloatOp::create(
      _builder, loc(node), llvm::cast<mlir::FloatType>(getFuncABIType(node)),
      node->value());
}

auto MLIRGenImpl::genFuncABIScalarLiteral(const BoolLiteralExpr *node)
    -> mlir::Value {
  auto boolType = llvm::cast<mlir::IntegerType>(getFuncABIType(node));
  return mlir::arith::ConstantIntOp::create(
      _builder, loc(node), node->value(), boolType.getWidth());
}

auto MLIRGenImpl::castToType(mlir::Value value, mlir::Type type,
                             mlir::Location location) -> mlir::Value {
  if (!value || value.getType() == type)
    return value;
  auto valueIntType = llvm::dyn_cast<mlir::IntegerType>(value.getType());
  auto targetIntType = llvm::dyn_cast<mlir::IntegerType>(type);
  auto valueFloatType = llvm::dyn_cast<mlir::FloatType>(value.getType());
  auto targetFloatType = llvm::dyn_cast<mlir::FloatType>(type);
  if ((llvm::isa<mlir::IntegerType>(value.getType()) &&
       llvm::isa<cir::IntType>(type)) ||
      (llvm::isa<cir::IntType>(value.getType()) &&
       llvm::isa<mlir::IntegerType>(type))) {
    return BridgeCastOp::create(_builder, location, type, value);
  }

  if ((valueIntType && valueIntType.getWidth() == 1 &&
       llvm::isa<cir::BoolType>(type)) ||
      (llvm::isa<cir::BoolType>(value.getType()) &&
       targetIntType && targetIntType.getWidth() == 1) ||
      (valueFloatType && valueFloatType.isF32() &&
       llvm::isa<cir::SingleType>(type)) ||
      (llvm::isa<cir::SingleType>(value.getType()) &&
       targetFloatType && targetFloatType.isF32())) {
    // Func ABI functions use standard MLIR scalar types, while CIR functions
    // still use CIR scalar types. The bridge op is erased after both sides are
    // lowered to the same LLVM scalar type.
    return BridgeCastOp::create(_builder, location, type, value);
  }

  if (llvm::isa<mlir::MemRefType>(value.getType()) &&
      llvm::isa<mlir::MemRefType>(type)) {
    // Assignment compatibility may erase static tensor shape information.
    // MLIR represents that explicitly with memref.cast.
    return mlir::memref::CastOp::create(_builder, location, type, value);
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

auto MLIRGenImpl::genCIRIndexValue(const Expr *node) -> mlir::Value {
  auto indexType =
      cir::IntType::get(_builder.getContext(), 64, /*isSigned=*/false);
  if (auto *decimal = llvm::dyn_cast<DecimalLiteralExpr>(node)) {
    return cir::ConstantOp::create(_builder, loc(node),
                                   cir::IntAttr::get(indexType,
                                                     decimal->value()));
  }

  return castToType(gen(node), indexType, loc(node));
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

  if (auto *indexExpr = llvm::dyn_cast<IndexExpr>(node))
    return genMemRefLoadValue(indexExpr);

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

mlir::Value MLIRGenImpl::genMemRefLoadValue(const IndexExpr *expr) {
  mlir::Value memref = getVariable(expr->getVarName());
  if (!memref || !llvm::isa<mlir::MemRefType>(memref.getType())) {
    ERR("Tensor index lowering expected memref: {0}", expr->getVarName());
    return nullptr;
  }

  llvm::SmallVector<mlir::Value, 4> mlirIndices;
  for (auto &idxExpr : expr->getIndices())
    mlirIndices.push_back(genIndexValue(idxExpr.get()));

  return mlir::memref::LoadOp::create(_builder, loc(expr), memref,
                                      mlirIndices);
}

mlir::Value MLIRGenImpl::gen(const IndexExpr *expr, bool isLValue) {
  if (isLValue)
    return nullptr;

  auto variable = getVariable(expr->getVarName());
  if (isMulberryList(variable)) {
    return genTensorListIndex(expr, variable);
  }

  if (variable && llvm::isa<cir::PointerType>(variable.getType())) {
    auto elementPtr = genListElementPointer(expr);
    if (!elementPtr)
      return nullptr;
    return cir::LoadOp::create(_builder, loc(expr), elementPtr);
  }

  auto loaded = genMemRefLoadValue(expr);
  if (isFuncABI() && isFuncABIValueType(expr->type()))
    return castToType(loaded, getFuncABIType(expr), loc(expr));
  return castToType(loaded, getMLIRType(expr), loc(expr));
}

void MLIRGenImpl::genAssignment(const IndexExpr *lhs, const Expr *rhs) {
  mlir::Value memref = getVariable(lhs->getVarName());
  if (memref && llvm::isa<cir::PointerType>(memref.getType())) {
    auto elementPtr = genListElementPointer(lhs);
    if (!elementPtr)
      return;

    auto rhsValue = castToType(gen(rhs), getMLIRType(lhs), loc(rhs));
    cir::StoreOp::create(_builder, loc(lhs), rhsValue, elementPtr,
                         /*isVolatile=*/false,
                         /*alignment=*/mlir::IntegerAttr{},
                         /*sync_scope=*/cir::SyncScopeKindAttr(),
                         /*mem-order=*/cir::MemOrderAttr());
    return;
  }

  if (!memref || !llvm::isa<mlir::MemRefType>(memref.getType())) {
    ERR("Tensor index assignment lowering expected memref: {0}",
        lhs->getVarName());
    return;
  }

  auto memRefType = llvm::cast<mlir::MemRefType>(memref.getType());

  llvm::SmallVector<mlir::Value, 4> mlirIndices;
  for (auto &idxExpr : lhs->getIndices())
    mlirIndices.push_back(genIndexValue(idxExpr.get()));

  auto rhsValue = genMemRefElementValue(rhs, memRefType.getElementType());
  mlir::memref::StoreOp::create(_builder, loc(lhs), rhsValue, memref,
                                mlirIndices);
}

auto MLIRGenImpl::gen(const VariableStat *node) -> CherryResult {
  auto *varType = node->type();
  auto *tensorType = cherry::getTensorType(varType);
  auto *listType = cherry::getListType(varType);
  auto *structType = cherry::getStructType(varType);
  auto varName = node->variable()->name();

  if (tensorType) {
    DBG("use Cherry variable tensor type `{0}`",
        formatType(tensorType));

    if (node->isConst()) {
      if (auto *tensorLiteral =
              llvm::dyn_cast<TensorLiteralExpr>(node->init().get())) {
        if (shouldPromoteConstTensorData(tensorLiteral)) {
          auto value = genGlobalTensorLiteral(tensorLiteral, varName);
          setVariable(varName, castToType(value, getMLIRType(varType),
                                          loc(node)));
          return success();
        }
      }
    }

    auto value = gen(node->init().get());
    setVariable(varName, castToType(value, getMLIRType(varType), loc(node)));
    return success();
  }

  if (listType) {
    DBG("use Cherry variable list type `{0}`", formatType(listType));
    if (!canMaterializeListStorage(listType)) {
      if (!isTensorList(listType))
        return emitError(node,
                         "List values containing Tensor require list/tensor "
                         "descriptor storage");

      auto *listLiteral = llvm::dyn_cast<ListLiteralExpr>(node->init().get());
      if (!listLiteral)
        return emitError(node,
                         "List<Tensor> lowering only supports list literals");

      auto value = genTensorListLiteral(listLiteral, listType);
      if (!value)
        return failure();
      setVariable(varName, value);
      return success();
    }

    auto mlirType = getMLIRType(varType);
    auto alloca = createEntryBlockAlloca(mlirType, loc(node));
    setVariable(varName, alloca);

    auto initValue = gen(node->init().get());
    if (!initValue)
      return failure();

    cir::StoreOp::create(_builder, loc(node), initValue, alloca,
                         /*isVolatile=*/false,
                         /*alignment=*/mlir::IntegerAttr{},
                         /*sync_scope=*/cir::SyncScopeKindAttr(),
                         /*mem-order=*/cir::MemOrderAttr());
    return success();
  }

  if (structType) {
    auto *structLiteral = llvm::dyn_cast<StructLiteralExpr>(node->init().get());
    if (structLiteral) {
      DBG("use Cherry variable struct literal `{0}`",
          formatType(structType));
      setVariable(varName, genStructLiteral(structLiteral, structType, nullptr));
      return success();
    }

    DBG("use Cherry variable struct value `{0}`",
        formatType(structType));
    auto mlirType = getMLIRType(varType);
    auto alloca = createEntryBlockAlloca(mlirType, loc(node));
    setVariable(varName, alloca);

    auto initValue = gen(node->init().get());
    cir::StoreOp::create(_builder, loc(node), initValue, alloca,
                         /*isVolatile=*/false,
                         /*alignment=*/mlir::IntegerAttr{},
                         /*sync_scope=*/cir::SyncScopeKindAttr(),
                         /*mem-order=*/cir::MemOrderAttr());
    return success();
  }

  if (cherry::isUnitType(varType)) {
    setVariable(varName, nullptr);
    gen(node->init().get());
    return success();
  }

  if (isFuncABI() && isFuncABIValueType(varType)) {
    auto value = gen(node->init().get());
    setVariable(varName, castToType(value, getFuncABIType(varType), loc(node)));
    return success();
  }

  auto mlirType = getMLIRType(varType);
  auto alloca = createEntryBlockAlloca(mlirType, loc(node));
  setVariable(varName, alloca);

  auto initValue = gen(node->init().get());

  initValue = castToType(initValue, mlirType, loc(node));
  cir::StoreOp::create(_builder, loc(node), initValue, alloca,
                       /*isVolatile=*/false,
                       /*alignment=*/mlir::IntegerAttr{},
                       /*sync_scope=*/cir::SyncScopeKindAttr(),
                       /*mem-order=*/cir::MemOrderAttr());
  return success();
}

auto MLIRGenImpl::gen(const ExprStat *node) -> CherryResult {
  gen(node->expression().get());
  return success();
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
