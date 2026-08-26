//===--- Sema.cpp - Mulberry Semantic Analysis ------------------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "mulberry/Sema/Sema.h"
#include "SemaImpl.h"
#include "SemaDeclarations.h"
#include "SemaExpressions.h"
#include "SemaStatements.h"
#include "SemaSupport.h"
#include "SemaTraits.h"
#include "mulberry/AST/AST.h"
#include "mulberry/Basic/Builtins.h"
#include "llvm/Support/Debug.h"
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#undef DEBUG_TYPE
#define DEBUG_TYPE "Sema"

namespace mulberry {
using llvm::cast;

auto SemaImpl::emitError(const Node *node, const llvm::Twine &msg) -> llvm::LogicalResult {
    _sourceManager.PrintMessage(node->location(),
                                llvm::SourceMgr::DiagKind::DK_Error, msg);
    return failure();
  }

auto SemaImpl::emitError(llvm::SMLoc loc, const llvm::Twine &msg) -> llvm::LogicalResult {
    _sourceManager.PrintMessage(loc, llvm::SourceMgr::DiagKind::DK_Error, msg);
    return failure();
  }

auto SemaImpl::emitNote(llvm::SMLoc loc, const llvm::Twine &msg) -> void {
  if (loc.isValid())
    _sourceManager.PrintMessage(loc, llvm::SourceMgr::DiagKind::DK_Note, msg);
}

auto SemaImpl::isInternalSourceLocation(llvm::SMLoc location) const -> bool {
    if (!location.isValid())
      return false;

    // Imported stdlib declarations keep their own source buffer locations, so
    // this allows internal storage code while rejecting the user's file.
    auto bufferId = _sourceManager.FindBufferContainingLoc(location);
    if (bufferId == 0)
      return false;

    auto path = std::string(
        _sourceManager.getMemoryBuffer(bufferId)->getBufferIdentifier());
    return path.rfind("stdlib/", 0) == 0 ||
           path.find("/stdlib/") != std::string::npos;
  }

auto SemaImpl::checkInternalFeature(llvm::SMLoc location) -> llvm::LogicalResult {
    if (isInternalSourceLocation(location))
      return success();
    return emitError(location, diag::internal_only);
  }

auto SemaImpl::lookupVariable(std::string_view name) const
    -> const VariableSymbol * {
  return _symbols.lookupVariable(name);
}

auto SemaImpl::completionMembers(std::string_view receiver) const
    -> std::vector<std::string> {
  std::vector<std::string> result;
  std::set<std::string, std::less<>> names;

  auto addName = [&](std::string_view name) -> void {
    if (!name.empty() && names.insert(std::string(name)).second)
      result.push_back(std::string(name));
  };

  auto appendStructMethods = [&](const StructType *structType) -> void {
    std::vector<std::string> owners;
    if (auto *origin = structType->origin())
      owners.push_back(std::string(origin->aliasName()));
    owners.push_back(std::string(structType->name()));

    for (const auto &owner : owners) {
      auto prefix = methodFunctionName(owner, "");
      for (const auto &[name, symbol] : _symbols.functions()) {
        (void)symbol;
        if (name.rfind(prefix, 0) != 0)
          continue;
        auto methodName = std::string_view(name).substr(prefix.size());
        if (methodName.find('.') == std::string_view::npos)
          addName(methodName);
      }
      for (const auto &[name, symbol] : _symbols.genericFunctions()) {
        (void)symbol;
        if (name.rfind(prefix, 0) != 0)
          continue;
        auto methodName = std::string_view(name).substr(prefix.size());
        if (methodName.find('.') == std::string_view::npos)
          addName(methodName);
      }
    }
  };

  auto typeForReceiver = [&](std::string_view name) -> const Type * {
    size_t dot = name.find('.');
    auto firstName = name.substr(0, dot);
    auto *variable = _symbols.lookupVariable(firstName);
    if (!variable)
      return nullptr;

    auto *type = variable->type;
    while (dot != std::string_view::npos) {
      auto *ptrType = getPtrType(type);
      auto *structType = getStructType(ptrType ? ptrType->pointeeType()
                                                : type);
      if (!structType)
        return nullptr;
      auto nextDot = name.find('.', dot + 1);
      auto fieldName = name.substr(dot + 1, nextDot - dot - 1);
      auto *field = structType->field(fieldName);
      if (!field)
        return nullptr;
      type = field->type();
      dot = nextDot;
    }
    return type;
  };

  if (auto *type = typeForReceiver(receiver)) {
    auto *ptrType = getPtrType(type);
    auto *structType = getStructType(ptrType ? ptrType->pointeeType()
                                              : type);
    if (!structType)
      return result;
    for (const auto &field : structType->fields())
      addName(field.name());
    appendStructMethods(structType);
    return result;
  }

  auto packageName = canonicalizeImportedName(receiver);
  auto packagePrefix = packageName + ".";
  for (const auto &[name, symbol] : _symbols.functions()) {
    (void)symbol;
    if (name.rfind(packagePrefix, 0) != 0)
      continue;
    auto functionName = std::string_view(name).substr(packagePrefix.size());
    if (functionName.find('.') == std::string_view::npos)
      addName(functionName);
  }
  for (const auto &[name, symbol] : _symbols.genericFunctions()) {
    (void)symbol;
    if (name.rfind(packagePrefix, 0) != 0)
      continue;
    auto functionName = std::string_view(name).substr(packagePrefix.size());
    if (functionName.find('.') == std::string_view::npos)
      addName(functionName);
  }
  return result;
}

auto SemaImpl::addBuiltins() -> void {
  for (auto kind : {BuiltinTypeKind::Unit, BuiltinTypeKind::Bool,
                    BuiltinTypeKind::UInt8, BuiltinTypeKind::UInt64,
                    BuiltinTypeKind::Int64, BuiltinTypeKind::Integer,
                    BuiltinTypeKind::Float32, BuiltinTypeKind::Float64}) {
    auto *type = _typeContext.getBuiltinType(kind);
    (void)declareType(type->name(), type, {}, Visibility::Public);
  }
}

auto SemaImpl::isAccessible(std::string_view packageName,
                            Visibility visibility) const -> bool {
  return visibility == Visibility::Public || packageName.empty() ||
         packageName == _currentPackageName;
}

auto SemaImpl::lookupFunction(std::string_view name) -> const FunctionSymbol * {
  auto lookup = [&](std::string_view candidate) -> const FunctionSymbol * {
    auto *signature = _symbols.lookupFunction(candidate);
    if (!signature ||
        !isAccessible(signature->packageName, signature->visibility))
      return nullptr;
    return signature;
  };

  if (auto *signature = lookup(name))
    return signature;

  auto importedName = canonicalizeImportedName(name);
  if (auto *signature = lookup(importedName))
    return signature;

  return lookup(qualifyCurrentPackageName(name));
}

auto SemaImpl::lookupFunctionDecl(std::string_view name)
    -> const FunctionDecl * {
  auto lookup = [&](std::string_view candidate) -> const FunctionDecl * {
    auto function = _functionDecls.find(std::string(candidate));
    if (function == _functionDecls.end() || !lookupFunction(candidate))
      return nullptr;
    return function->second;
  };

  if (auto *decl = lookup(name))
    return decl;

  auto importedName = canonicalizeImportedName(name);
  if (auto *decl = lookup(importedName))
    return decl;

  return lookup(qualifyCurrentPackageName(name));
}

auto SemaImpl::registerFunctionDecl(std::string_view name,
                                    const FunctionDecl *decl) -> void {
  _functionDecls[std::string(name)] = decl;
}

auto SemaImpl::lookupGenericFunction(std::string_view name)
    -> const GenericFunctionSymbol * {
  auto lookup = [&](std::string_view candidate)
      -> const GenericFunctionSymbol * {
    auto *genericFunction = _symbols.lookupGenericFunction(candidate);
    if (!genericFunction ||
        !isAccessible(genericFunction->packageName,
                      genericFunction->visibility))
      return nullptr;
    return genericFunction;
  };

  if (auto *genericFunction = lookup(name))
    return genericFunction;

  auto importedName = canonicalizeImportedName(name);
  if (auto *genericFunction = lookup(importedName))
    return genericFunction;

  return lookup(qualifyCurrentPackageName(name));
}

auto SemaImpl::emptyImportAliases()
    -> const std::map<std::string, std::string> & {
  static const std::map<std::string, std::string> aliases;
  return aliases;
}

auto SemaImpl::canonicalizeImportedName(std::string_view name) const
    -> std::string {
  auto importedName = _importAliases.find(std::string(name));
  if (importedName != _importAliases.end())
    return importedName->second;

  auto dot = name.find('.');
  if (dot == std::string_view::npos)
    return std::string(name);

  auto alias = _importAliases.find(std::string(name.substr(0, dot)));
  if (alias == _importAliases.end())
    return std::string(name);

  std::string fullName = alias->second;
  fullName += ".";
  fullName += name.substr(dot + 1);
  return fullName;
}

auto SemaImpl::qualifyCurrentPackageName(std::string_view name) const
    -> std::string {
  if (_currentPackageName.empty() ||
      name.find('.') != std::string_view::npos)
    return std::string(name);

  std::string fullName = _currentPackageName;
  fullName += ".";
  fullName += name;
  return fullName;
}

auto SemaImpl::lookupType(std::string_view name) -> const Type * {
  auto lookup = [&](std::string_view candidate) -> const Type * {
    auto *symbol = _symbols.lookupTypeSymbol(candidate);
    if (!symbol)
      return nullptr;
    if (!isAccessible(symbol->packageName, symbol->visibility)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "type lookup rejected `" << candidate << "` from package `"
                 << _currentPackageName << "`: declared in `"
                 << symbol->packageName << "`, visibility="
                 << static_cast<int>(symbol->visibility) << "\n");
      return nullptr;
    }
    return symbol->type;
  };

  if (auto *type = lookup(name))
    return type;

  auto importedName = canonicalizeImportedName(name);
  if (auto *type = lookup(importedName))
    return type;

  return lookup(qualifyCurrentPackageName(name));
}

auto SemaImpl::lookupTypeSymbol(std::string_view name) -> const TypeSymbol * {
  auto lookup = [&](std::string_view candidate) -> const TypeSymbol * {
    return _symbols.lookupTypeSymbol(candidate);
  };

  if (auto *symbol = lookup(name))
    return symbol;

  auto importedName = canonicalizeImportedName(name);
  if (auto *symbol = lookup(importedName))
    return symbol;

  return lookup(qualifyCurrentPackageName(name));
}

auto SemaImpl::lookupDataDeclSymbol(std::string_view name)
    -> const DataDeclSymbol * {
  auto lookup = [&](std::string_view candidate) -> const DataDeclSymbol * {
    return _symbols.lookupDataDecl(candidate);
  };

  if (auto *symbol = lookup(name))
    return symbol;

  auto importedName = canonicalizeImportedName(name);
  if (auto *symbol = lookup(importedName))
    return symbol;

  return lookup(qualifyCurrentPackageName(name));
}

auto SemaImpl::lookupComptimeTypeAliasSymbol(std::string_view name)
    -> const ComptimeTypeAliasSymbol * {
  auto lookup = [&](std::string_view candidate)
      -> const ComptimeTypeAliasSymbol * {
    return _symbols.lookupComptimeTypeAlias(candidate);
  };

  if (auto *symbol = lookup(name))
    return symbol;

  auto importedName = canonicalizeImportedName(name);
  if (auto *symbol = lookup(importedName))
    return symbol;

  return lookup(qualifyCurrentPackageName(name));
}

auto SemaImpl::comptimeTypeAliasName(std::string_view name) -> std::string {
  auto isAliasAccessible = [&](std::string_view candidate) {
    auto *alias = _symbols.lookupComptimeTypeAlias(candidate);
    return alias &&
           isAccessible(alias->packageName, alias->visibility);
  };
  if (isAliasAccessible(name))
    return std::string(name);

  auto importedName = canonicalizeImportedName(name);
  if (isAliasAccessible(importedName))
    return importedName;

  auto packageName = qualifyCurrentPackageName(name);
  if (isAliasAccessible(packageName))
    return packageName;

  return {};
}

auto SemaImpl::lookupComptimeTypeAlias(std::string_view name)
    -> const ComptimeTypeAliasSymbol * {
  auto aliasName = comptimeTypeAliasName(name);
  if (aliasName.empty())
    return nullptr;
  return _symbols.lookupComptimeTypeAlias(aliasName);
}

auto SemaImpl::dataDeclName(std::string_view name) -> std::string {
  auto isDataAccessible = [&](std::string_view candidate) {
    auto *data = _symbols.lookupDataDecl(candidate);
    return data && isAccessible(data->packageName, data->visibility);
  };
  if (isDataAccessible(name))
    return std::string(name);

  auto importedName = canonicalizeImportedName(name);
  if (isDataAccessible(importedName))
    return importedName;

  auto packageName = qualifyCurrentPackageName(name);
  if (isDataAccessible(packageName))
    return packageName;

  return {};
}

auto SemaImpl::lookupDataConstructor(std::string_view name,
                                     std::string &resolvedName)
    -> const DataConstructorSymbol * {
  auto isConstructorAccessible = [&](std::string_view candidate,
                                     const DataConstructorSymbol *constructor) {
    return constructor &&
           this->isAccessible(
               packageNameOf(candidate),
               constructor->decl->isPublic() ? Visibility::Public
                                             : Visibility::Private);
  };
  if (auto *constructor = _symbols.lookupDataConstructor(name);
      isConstructorAccessible(name, constructor)) {
    resolvedName = std::string(name);
    return constructor;
  }

  auto importedName = canonicalizeImportedName(name);
  if (auto *constructor = _symbols.lookupDataConstructor(importedName);
      isConstructorAccessible(importedName, constructor)) {
    resolvedName = importedName;
    return constructor;
  }

  auto packageName = qualifyCurrentPackageName(name);
  if (auto *constructor = _symbols.lookupDataConstructor(packageName);
      isConstructorAccessible(packageName, constructor)) {
    resolvedName = packageName;
    return constructor;
  }

  return nullptr;
}

auto SemaImpl::declareVariable(std::string_view name, const Type *type,
                               bool isConstBinding,
                               bool canMutateObject,
                               std::optional<ComptimeValue> comptimeValue,
                               bool isComptimeOnly) -> llvm::LogicalResult {
  std::optional<uint64_t> replSlot;
  if (_persistent && !isComptimeOnly &&
      _symbols.variableScopeDepth() == 1)
    replSlot = _nextReplSlot++;

  return _symbols.declareVariable(name, type, isConstBinding,
                                  canMutateObject, std::move(comptimeValue),
                                  isComptimeOnly, replSlot);
}

auto SemaImpl::declareType(std::string_view name, const Type *type,
                           std::string_view packageName,
                           Visibility visibility)
    -> llvm::LogicalResult {
  return _symbols.declareType(name, type, packageName, visibility);
}

auto SemaImpl::declareStructType(const StructType *type,
                                 std::string_view packageName,
                                 Visibility visibility)
    -> llvm::LogicalResult {
  return declareType(type->name(), type, packageName, visibility);
}

auto SemaImpl::declareFunction(std::string_view name, const FunctionType *type,
                               bool isExtern, std::string_view packageName,
                               Visibility visibility)
    -> llvm::LogicalResult {
  if (packageName.empty())
    packageName = _currentPackageName;
  _functionPackages[std::string(name)] = std::string(packageName);
  return _symbols.declareFunction(name, type, isExtern, packageName,
                                  visibility);
}

auto SemaImpl::declareGenericFunction(std::string_view name,
                                      const FunctionDecl *decl,
                                      std::string_view packageName,
                                      Visibility visibility)
    -> llvm::LogicalResult {
  if (packageName.empty())
    packageName = _currentPackageName;
  _genericFunctionPackages[std::string(name)] = std::string(packageName);
  return _symbols.declareGenericFunction(name, decl, packageName, visibility);
}

auto SemaImpl::sema(Module &node, bool checkReplResult)
    -> llvm::LogicalResult {
  if (!_persistent)
    return semaModule(node, checkReplResult);

  auto savedState = saveState();

  auto result = semaModule(node, checkReplResult);
  if (llvm::succeeded(result))
    return result;

  // A failed submission must not reserve names or runtime slots in the live
  // session. Type objects are intentionally arena-like and can remain unused.
  restoreState(std::move(savedState));
  return result;
}

auto SemaImpl::saveState() const -> State {
  State state;
  state.symbols = _symbols;
  state.genericStructTypes = _genericStructTypes;
  state.dataTypes = _dataTypes;
  state.functionDecls = _functionDecls;
  state.instantiatedFunctionSymbols = _instantiatedFunctionSymbols;
  state.functionPackages = _functionPackages;
  state.genericFunctionPackages = _genericFunctionPackages;
  state.instantiatedFunctionPackages = _instantiatedFunctionPackages;
  state.fromTrait = _fromTrait;
  state.instantiatedFunctions = _instantiatedFunctions.size();
  state.lambdaFunctions = _lambdaFunctions.size();
  state.currentPackageName = _currentPackageName;
  state.whileDepth = _whileDepth;
  state.noncapturingLambdaDepth = _noncapturingLambdaDepth;
  state.lambdaCounter = _lambdaCounter;
  state.nextReplSlot = _nextReplSlot;
  return state;
}

auto SemaImpl::restoreState(State state) -> void {
  _symbols = std::move(state.symbols);
  _genericStructTypes = std::move(state.genericStructTypes);
  _dataTypes = std::move(state.dataTypes);
  _functionDecls = std::move(state.functionDecls);
  _functionPackages = std::move(state.functionPackages);
  _genericFunctionPackages = std::move(state.genericFunctionPackages);
  _instantiatedFunctionPackages = std::move(state.instantiatedFunctionPackages);
  _fromTrait = state.fromTrait;

  // FunctionSymbol pointers point into Symbols. Rebind them after replacing
  // the symbol table so rollback cannot retain pointers into the old table.
  auto instantiatedFunctionNames =
      std::move(state.instantiatedFunctionSymbols);
  _instantiatedFunctionSymbols.clear();
  for (const auto &[name, symbol] : instantiatedFunctionNames) {
    (void)symbol;
    if (auto *function = _symbols.lookupFunction(name))
      _instantiatedFunctionSymbols.insert({name, function});
  }

  _instantiatedFunctions.resize(state.instantiatedFunctions);
  _lambdaFunctions.resize(state.lambdaFunctions);
  _currentPackageName = std::move(state.currentPackageName);
  _whileDepth = state.whileDepth;
  _noncapturingLambdaDepth = state.noncapturingLambdaDepth;
  _lambdaCounter = state.lambdaCounter;
  _nextReplSlot = state.nextReplSlot;
}

auto SemaImpl::semaModule(Module &node, bool checkReplResult)
    -> llvm::LogicalResult {
  // All declarations in a source module start in the module package. Each
  // imported declaration temporarily switches to its defining package below.
  SemaImpl::PackageScope modulePackageScope(_currentPackageName,
                                            node.packageName());

  if (_persistent) {
    std::vector<ReplFunctionBinding> functions;
    for (const auto &[name, symbol] : _symbols.functions())
      functions.push_back({name, symbol.type, symbol.isExtern});
    node.setReplFunctions(std::move(functions));
  }

  for (auto &decl : node)
    if (llvm::failed(sema(decl.get())))
      return failure();

  size_t instantiatedFunctions = 0;
  auto semaInstantiatedFunctions = [&]() -> llvm::LogicalResult {
    while (instantiatedFunctions < _instantiatedFunctions.size()) {
      if (llvm::failed(
              sema(_instantiatedFunctions[instantiatedFunctions].get())))
        return failure();
      ++instantiatedFunctions;
    }
    return success();
  };

  if (llvm::failed(semaInstantiatedFunctions()))
    return failure();

  if (node.isRepl()) {
    for (auto &statement : node.statements())
      if (llvm::failed(sema(statement.get())))
        return failure();

    if (llvm::failed(semaInstantiatedFunctions()))
      return failure();

    if (checkReplResult && !node.statements().empty()) {
      auto *last = node.statements().back().get();
      if (auto *expression = llvm::dyn_cast<ExprStat>(last)) {
        if (!isUnitType(expression->expression()->type())) {
          VectorUniquePtr<Expr> arguments;
          arguments.push_back(std::move(expression->expression()));
          auto display = std::make_unique<CallExpr>(
              expression->location(), "std.io.print", std::move(arguments));
          expression->expression() = std::move(display);
          if (llvm::failed(sema(expression->expression().get())) ||
              llvm::failed(semaInstantiatedFunctions()))
            return failure();
        }
      }
    }

    auto declarations = node.takeDeclarations();
    for (auto &function : _instantiatedFunctions)
      declarations.push_back(std::move(function));
    _instantiatedFunctions.clear();
    for (auto &function : _lambdaFunctions)
      declarations.push_back(std::move(function));
    _lambdaFunctions.clear();
    node.setDeclarations(std::move(declarations));

    if (_persistent) {
      std::vector<ReplVariableBinding> variables;
      for (const auto &[name, symbol] : _symbols.currentVariables()) {
        if (!symbol.replSlot)
          continue;
        variables.push_back({name, symbol.type, *symbol.replSlot,
                             symbol.isConstBinding,
                             symbol.canMutateObject});
      }
      node.setReplVariables(std::move(variables));
    }
    return success();
  }

  auto declarations = node.takeDeclarations();
  for (auto &function : _instantiatedFunctions)
    declarations.push_back(std::move(function));
  _instantiatedFunctions.clear();
  for (auto &function : _lambdaFunctions)
    declarations.push_back(std::move(function));
  _lambdaFunctions.clear();
  node.setDeclarations(std::move(declarations));

  std::string mainName = node.packageName().empty()
                             ? "main"
                             : std::string(node.packageName()) + ".main";
  auto *mainSignature = lookupFunction(mainName);
  if (!mainSignature || !mainSignature->type->parameterTypes().empty() ||
      !isUInt64Type(mainSignature->type->returnType()))
    return emitError(llvm::SMLoc{}, diag::undefined_main);
  return success();
}

auto SemaImpl::sema(Decl *node) -> llvm::LogicalResult {
  switch (node->getKind()) {
  case Decl::Decl_Import:
    return success();
  case Decl::Decl_Function:
    return DeclarationSema(*this).sema(cast<FunctionDecl>(node));
  case Decl::Decl_Struct:
    return DeclarationSema(*this).sema(cast<StructDecl>(node));
  case Decl::Decl_Data:
    return DeclarationSema(*this).sema(cast<DataDecl>(node));
  case Decl::Decl_ComptimeTypeAlias:
    return DeclarationSema(*this).sema(cast<ComptimeTypeAliasDecl>(node));
  case Decl::Decl_Trait:
    return TraitSema(*this).sema(cast<TraitDecl>(node));
  case Decl::Decl_Impl:
    return TraitSema(*this).sema(cast<ImplDecl>(node));
  }
}

auto SemaImpl::sema(Expr *node) -> llvm::LogicalResult {
  return ExpressionSema(*this).sema(node);
}

auto SemaImpl::sema(Expr *node, const Type *type) -> llvm::LogicalResult {
  return ExpressionSema(*this).sema(node, type);
}

auto SemaImpl::sema(BlockExpr *node) -> llvm::LogicalResult {
  return StatementSema(*this).sema(node);
}

auto SemaImpl::semaExpected(std::unique_ptr<Expr> &node, const Type *type)
    -> llvm::LogicalResult {
  return ExpressionSema(*this).semaExpected(node, type);
}

auto SemaImpl::sema(Stat *node) -> llvm::LogicalResult {
  return StatementSema(*this).sema(node);
}

auto sema(const llvm::SourceMgr &sourceManager, Module &moduleAST)
    -> llvm::LogicalResult {
  return SemaImpl(sourceManager).sema(moduleAST);
}

auto sema(const llvm::SourceMgr &sourceManager, Module &moduleAST,
          const std::map<std::string, std::string> &importAliases)
    -> llvm::LogicalResult {
  return SemaImpl(sourceManager, importAliases).sema(moduleAST);
}

SemaSession::SemaSession(
    const llvm::SourceMgr &sourceManager,
    const std::map<std::string, std::string> &importAliases)
    : _impl{std::make_unique<SemaImpl>(sourceManager, importAliases,
                                       /*persistent=*/true)} {}

struct SemaSession::PendingState {
  SemaImpl::State semaState;
  size_t retainedDeclarationCount = 0;
};

SemaSession::~SemaSession() = default;

auto SemaSession::sema(Module &module, bool checkReplResult)
    -> llvm::LogicalResult {
  auto pendingState = std::make_unique<PendingState>();
  pendingState->semaState = _impl->saveState();
  pendingState->retainedDeclarationCount = _retainedDeclarations.size();

  auto result = _impl->sema(module, checkReplResult);
  if (llvm::failed(result))
    return result;

  _pendingState = std::move(pendingState);
  return success();
}

auto SemaSession::retainDeclarations(VectorUniquePtr<Decl> declarations)
    -> void {
  for (auto &declaration : declarations)
    _retainedDeclarations.push_back(std::move(declaration));
}

auto SemaSession::commitReplSubmission() -> void { _pendingState.reset(); }

auto SemaSession::rollbackReplSubmission() -> void {
  if (!_pendingState)
    return;

  auto pendingState = std::move(_pendingState);
  _impl->restoreState(std::move(pendingState->semaState));
  _retainedDeclarations.resize(pendingState->retainedDeclarationCount);
}

auto SemaSession::completionNames() const -> std::vector<std::string> {
  return _impl->completionNames();
}

auto SemaSession::completionMembers(std::string_view receiver) const
    -> std::vector<std::string> {
  return _impl->completionMembers(receiver);
}

} // end namespace mulberry
