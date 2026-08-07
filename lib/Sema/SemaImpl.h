//===--- SemaImpl.h - Mulberry semantic implementation context -*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_SEMA_IMPL_H
#define MULBERRY_SEMA_IMPL_H

#include "SemaData.h"
#include "Symbols.h"
#include "mulberry/AST/AST.h"
#include "llvm/Support/SourceMgr.h"
#include "mulberry/Sema/DiagnosticsSema.h"
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mulberry {

class ComptimeFrame;
struct ComptimeExecutionState;

class SemaImpl {
  friend class ComptimeSema;
  friend class DeclarationSema;
  friend class ExpressionSema;
  friend class SemaSession;
  friend class StatementSema;
  friend class TraitSema;

public:
  SemaImpl(const llvm::SourceMgr &sourceManager, bool persistent = false)
      : _sourceManager{sourceManager}, _persistent{persistent} {
    if (_persistent)
      _symbols.resetVariables();
    addBuiltins();
    registerBuiltinHandlers();
  }

  SemaImpl(const llvm::SourceMgr &sourceManager,
           const std::map<std::string, std::string> &importAliases,
           bool persistent = false)
      : _sourceManager{sourceManager}, _persistent{persistent},
        _importAliases{importAliases} {
    if (_persistent)
      _symbols.resetVariables();
    addBuiltins();
    registerBuiltinHandlers();
  }

  auto sema(Module &node, bool checkReplResult = true)
      -> llvm::LogicalResult;
  auto completionNames() const -> std::vector<std::string> {
    return _symbols.completionNames();
  }
  auto completionMembers(std::string_view receiver) const
      -> std::vector<std::string>;

private:
  struct State {
    Symbols symbols;
    std::map<std::string, const StructType *> genericStructTypes;
    std::map<std::string, DataType *> dataTypes;
    std::map<std::string, const FunctionDecl *> functionDecls;
    std::map<std::string, const FunctionSymbol *> instantiatedFunctionSymbols;
    std::map<std::string, std::string> functionPackages;
    std::map<std::string, std::string> genericFunctionPackages;
    std::map<std::string, std::string> instantiatedFunctionPackages;
    size_t instantiatedFunctions = 0;
    size_t lambdaFunctions = 0;
    std::string currentPackageName;
    int whileDepth = 0;
    int noncapturingLambdaDepth = 0;
    uint64_t lambdaCounter = 0;
    uint64_t nextReplSlot = 0;
  };

  auto saveState() const -> State;
  auto restoreState(State state) -> void;

  using BuiltinHandler =
      std::function<llvm::LogicalResult(Expr *, const Type *)>;

  const llvm::SourceMgr &_sourceManager;
  bool _persistent = false;
  TypeContext _typeContext;
  Symbols _symbols;
  std::map<std::string, const FunctionDecl *> _functionDecls;
  std::map<std::string, const StructType *> _genericStructTypes;
  std::map<std::string, DataType *> _dataTypes;
  std::map<std::string, const FunctionSymbol *> _instantiatedFunctionSymbols;
  std::map<std::string, std::string> _functionPackages;
  std::map<std::string, std::string> _genericFunctionPackages;
  std::map<std::string, std::string> _instantiatedFunctionPackages;
  std::map<std::string, BuiltinHandler, std::less<>> _builtinHandlers;
  VectorUniquePtr<FunctionDecl> _instantiatedFunctions;
  VectorUniquePtr<FunctionDecl> _lambdaFunctions;
  const std::map<std::string, std::string> &_importAliases =
      emptyImportAliases();
  std::string _currentPackageName;
  const Type *_currentFunctionReturnType = nullptr;
  ComptimeFrame *_activeComptimeFrame = nullptr;
  ComptimeExecutionState *_activeComptimeExecutionState = nullptr;
  int _whileDepth = 0;
  int _noncapturingLambdaDepth = 0;
  uint64_t _lambdaCounter = 0;
  uint64_t _nextReplSlot = 0;

  enum class UnitPolicy {
    Allow,
    Reject,
  };

  // Semantic Analysis

  auto semaModule(Module &node, bool checkReplResult) -> llvm::LogicalResult;

  // Declarations
  auto sema(Decl *node) -> llvm::LogicalResult;
  // Expressions
  auto sema(Expr *node) -> llvm::LogicalResult;
  auto sema(Expr *node, const Type *type) -> llvm::LogicalResult;
  auto semaExpected(std::unique_ptr<Expr> &node, const Type *type)
      -> llvm::LogicalResult;
  auto sema(BlockExpr *node) -> llvm::LogicalResult;
  // Compiler builtins
  auto registerBuiltinHandlers() -> void;
  auto registerBuiltinHandler(std::string_view name, BuiltinHandler handler)
      -> void;
  auto lookupBuiltinHandler(std::string_view name) const
      -> const BuiltinHandler *;
  auto semaToUInt8(CallExpr *node, const Type *expectedType)
      -> llvm::LogicalResult;
  auto semaToUInt64(CallExpr *node, const Type *expectedType)
      -> llvm::LogicalResult;
  auto semaToFloat32(CallExpr *node, const Type *expectedType)
      -> llvm::LogicalResult;
  auto semaToFloat64(CallExpr *node, const Type *expectedType)
      -> llvm::LogicalResult;

  // Statements
  auto sema(Stat *node) -> llvm::LogicalResult;

  // Errors
  auto emitError(const Node *node, const llvm::Twine &msg) -> llvm::LogicalResult ;

  auto emitError(llvm::SMLoc loc, const llvm::Twine &msg) -> llvm::LogicalResult ;

  auto emitNote(llvm::SMLoc loc, const llvm::Twine &msg) -> void ;

  auto isInternalSourceLocation(llvm::SMLoc location) const -> bool ;

  auto checkInternalFeature(llvm::SMLoc location) -> llvm::LogicalResult ;

  auto lookupVariable(std::string_view name) const
      -> const VariableSymbol * ;

  auto addBuiltins() -> void ;

  auto lookupFunction(std::string_view name) -> const FunctionSymbol * ;

  auto lookupFunctionDecl(std::string_view name) -> const FunctionDecl * ;

  auto registerFunctionDecl(std::string_view name,
                            const FunctionDecl *decl) -> void ;

  auto lookupGenericFunction(std::string_view name)
      -> const GenericFunctionSymbol * ;

  static auto emptyImportAliases()
      -> const std::map<std::string, std::string> & ;

  auto canonicalizeImportedName(std::string_view name) const -> std::string ;

  auto qualifyCurrentPackageName(std::string_view name) const -> std::string ;

  auto lookupType(std::string_view name) -> const Type * ;

  auto comptimeTypeAliasName(std::string_view name) -> std::string ;

  auto lookupComptimeTypeAlias(std::string_view name)
      -> const ComptimeTypeAliasSymbol * ;

  auto dataDeclName(std::string_view name) -> std::string ;

  auto lookupDataConstructor(std::string_view name,
                             std::string &resolvedName)
      -> const DataConstructorSymbol * ;

  auto resolveType(const NamedTypeNode *typeNode) -> const Type * ;

  auto resolveType(const UnitTypeNode *typeNode) -> const Type * ;

  auto declareVariable(std::string_view name, const Type *type,
                       bool isConstBinding = false,
                       bool canMutateObject = true,
                       std::optional<ComptimeValue> comptimeValue = std::nullopt,
                       bool isComptimeOnly = false)
      -> llvm::LogicalResult ;

  auto beginFunctionScope() -> void {
    if (_persistent)
      _symbols.enterVariableScope();
    else
      _symbols.resetVariables();
  }

  auto endFunctionScope() -> void { _symbols.leaveVariableScope(); }

  class FunctionScope {
  public:
    explicit FunctionScope(SemaImpl &sema) : _sema(sema) {
      _sema.beginFunctionScope();
    }

    ~FunctionScope() { _sema.endFunctionScope(); }

  private:
    SemaImpl &_sema;
  };

  class VariableScope {
  public:
    explicit VariableScope(Symbols &symbols) : _symbols(symbols) {
      _symbols.enterVariableScope();
    }

    ~VariableScope() { _symbols.leaveVariableScope(); }

  private:
    Symbols &_symbols;
  };

  class ComptimeFrameScope {
  public:
    ComptimeFrameScope(SemaImpl &sema, ComptimeFrame *frame)
        : _sema(sema), _oldFrame(sema._activeComptimeFrame) {
      _sema._activeComptimeFrame = frame;
    }

    ~ComptimeFrameScope() { _sema._activeComptimeFrame = _oldFrame; }

  private:
    SemaImpl &_sema;
    ComptimeFrame *_oldFrame = nullptr;
  };

  class ComptimeExecutionScope {
  public:
    ComptimeExecutionScope(SemaImpl &sema, ComptimeExecutionState *state)
        : _sema(sema), _oldState(sema._activeComptimeExecutionState) {
      _sema._activeComptimeExecutionState = state;
    }

    ~ComptimeExecutionScope() {
      _sema._activeComptimeExecutionState = _oldState;
    }

  private:
    SemaImpl &_sema;
    ComptimeExecutionState *_oldState = nullptr;
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

  class WhileScope {
  public:
    explicit WhileScope(int &whileDepth) : _whileDepth(whileDepth) {
      ++_whileDepth;
    }

    ~WhileScope() { --_whileDepth; }

  private:
    int &_whileDepth;
  };

  class IsolatedWhileScope {
  public:
    explicit IsolatedWhileScope(int &whileDepth)
        : _whileDepth(whileDepth), _oldWhileDepth(whileDepth) {
      _whileDepth = 0;
    }

    ~IsolatedWhileScope() { _whileDepth = _oldWhileDepth; }

  private:
    int &_whileDepth;
    int _oldWhileDepth;
  };

  class NoncapturingLambdaScope {
  public:
    explicit NoncapturingLambdaScope(int &depth) : _depth(depth) { ++_depth; }

    ~NoncapturingLambdaScope() { --_depth; }

  private:
    int &_depth;
  };

  class FunctionReturnTypeScope {
  public:
    FunctionReturnTypeScope(const Type *&currentFunctionReturnType,
                            const Type *returnType)
        : _currentFunctionReturnType(currentFunctionReturnType),
          _oldFunctionReturnType(currentFunctionReturnType) {
      _currentFunctionReturnType = returnType;
    }

    ~FunctionReturnTypeScope() {
      _currentFunctionReturnType = _oldFunctionReturnType;
    }

  private:
    const Type *&_currentFunctionReturnType;
    const Type *_oldFunctionReturnType;
  };

  auto declareFunction(std::string_view name, const FunctionType *type,
                       bool isExtern,
                       std::string_view packageName = {})
      -> llvm::LogicalResult;

  auto declareGenericFunction(std::string_view name,
                              const FunctionDecl *decl,
                              std::string_view packageName = {})
      -> llvm::LogicalResult;

  auto declareType(std::string_view name, const Type *type) -> llvm::LogicalResult ;

  auto declareStructType(const StructType *type) -> llvm::LogicalResult ;

  auto resolveType(const ArrayTypeNode *typeNode) -> const Type * ;

  auto resolveType(const PtrTypeNode *typeNode) -> const Type * ;

  auto resolveType(const FunctionTypeNode *typeNode) -> const Type * ;

  auto resolveStructFields(const StructTypeNode *typeNode)
      -> std::optional<std::vector<StructField>> ;

  auto resolveType(const StructTypeNode *typeNode, std::string_view name)
      -> const StructType * ;

  auto resolveType(const StructTypeNode *typeNode, std::string_view name,
                   ComptimeAliasOrigin origin) -> const StructType * ;

  auto resolveType(const ComputedTypeNode *typeNode) -> const Type * ;

  auto completeDataType(const DataDecl *decl, DataType *dataType,
                        const std::vector<TypeSubstitution> &substitutions)
      -> llvm::LogicalResult ;

  auto instantiateDataType(
      const DataDecl *decl, const std::vector<ComptimeArgument> &arguments,
      const std::vector<TypeSubstitution> &substitutions) -> DataType * ;

  auto resolveType(const GenericTypeNode *typeNode) -> const Type * ;

  auto resolveType(const TypeNode *typeNode) -> const Type * ;

  auto checkType(const TypeNode *typeNode, UnitPolicy unitPolicy)
      -> const Type * ;

  auto setBuiltinType(Expr *expr, BuiltinTypeKind kind) -> void ;

};

} // namespace mulberry

#endif // MULBERRY_SEMA_IMPL_H
