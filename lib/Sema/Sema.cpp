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
#include "SemaTraits.h"
#include "mulberry/AST/AST.h"
#include "mulberry/Basic/Builtins.h"
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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

auto SemaImpl::lookupVariable(std::string_view name) -> const VariableSymbol * {
    return _symbols.lookupVariable(name);
  }

auto SemaImpl::addBuiltins() -> void {
    for (auto kind : {BuiltinTypeKind::Unit, BuiltinTypeKind::Bool,
                      BuiltinTypeKind::UInt8, BuiltinTypeKind::UInt64,
                      BuiltinTypeKind::Integer, BuiltinTypeKind::Float32}) {
      auto *type = _typeContext.getBuiltinType(kind);
      (void)declareType(type->name(), type);
    }
  }

auto SemaImpl::lookupFunction(std::string_view name) -> const FunctionSymbol * {
    if (auto *signature = _symbols.lookupFunction(name))
      return signature;

    auto importedName = canonicalizeImportedName(name);
    if (auto *signature = _symbols.lookupFunction(importedName))
      return signature;

    return _symbols.lookupFunction(qualifyCurrentPackageName(name));
  }

auto SemaImpl::lookupGenericFunction(std::string_view name)
      -> const GenericFunctionSymbol * {
    if (auto *genericFunction = _symbols.lookupGenericFunction(name))
      return genericFunction;

    auto importedName = canonicalizeImportedName(name);
    if (auto *genericFunction = _symbols.lookupGenericFunction(importedName))
      return genericFunction;

    return _symbols.lookupGenericFunction(qualifyCurrentPackageName(name));
  }

auto SemaImpl::emptyImportAliases()
      -> const std::map<std::string, std::string> & {
    static const std::map<std::string, std::string> aliases;
    return aliases;
  }

auto SemaImpl::canonicalizeImportedName(std::string_view name) const -> std::string {
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

auto SemaImpl::qualifyCurrentPackageName(std::string_view name) const -> std::string {
    if (_currentPackageName.empty() ||
        name.find('.') != std::string_view::npos)
      return std::string(name);

    std::string fullName = _currentPackageName;
    fullName += ".";
    fullName += name;
    return fullName;
  }

auto SemaImpl::lookupType(std::string_view name) -> const Type * {
    if (auto *type = _symbols.lookupType(name))
      return type;

    auto importedName = canonicalizeImportedName(name);
    if (auto *type = _symbols.lookupType(importedName))
      return type;

    return _symbols.lookupType(qualifyCurrentPackageName(name));
  }

auto SemaImpl::comptimeTypeAliasName(std::string_view name) -> std::string {
    if (_symbols.lookupComptimeTypeAlias(name))
      return std::string(name);

    auto importedName = canonicalizeImportedName(name);
    if (_symbols.lookupComptimeTypeAlias(importedName))
      return importedName;

    auto packageName = qualifyCurrentPackageName(name);
    if (_symbols.lookupComptimeTypeAlias(packageName))
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
    if (_symbols.lookupDataDecl(name))
      return std::string(name);

    auto importedName = canonicalizeImportedName(name);
    if (_symbols.lookupDataDecl(importedName))
      return importedName;

    auto packageName = qualifyCurrentPackageName(name);
    if (_symbols.lookupDataDecl(packageName))
      return packageName;

    return {};
  }

auto SemaImpl::lookupDataConstructor(std::string_view name,
                             std::string &resolvedName)
      -> const DataConstructorSymbol * {
    if (auto *constructor = _symbols.lookupDataConstructor(name)) {
      resolvedName = std::string(name);
      return constructor;
    }

    auto importedName = canonicalizeImportedName(name);
    if (auto *constructor =
            _symbols.lookupDataConstructor(importedName)) {
      resolvedName = importedName;
      return constructor;
    }

    auto packageName = qualifyCurrentPackageName(name);
    if (auto *constructor =
            _symbols.lookupDataConstructor(packageName)) {
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
  return _symbols.declareVariable(name, type, isConstBinding,
                                  canMutateObject, std::move(comptimeValue),
                                  isComptimeOnly);
}

auto SemaImpl::declareType(std::string_view name, const Type *type)
    -> llvm::LogicalResult {
  return _symbols.declareType(name, type);
}

auto SemaImpl::declareStructType(const StructType *type)
    -> llvm::LogicalResult {
  return declareType(type->name(), type);
}

auto SemaImpl::declareFunction(std::string_view name, const FunctionType *type,
                               bool isExtern, std::string_view packageName)
    -> llvm::LogicalResult {
  if (packageName.empty())
    packageName = _currentPackageName;
  _functionPackages[std::string(name)] = std::string(packageName);
  return _symbols.declareFunction(name, type, isExtern);
}

auto SemaImpl::declareGenericFunction(std::string_view name,
                                      const FunctionDecl *decl,
                                      std::string_view packageName)
    -> llvm::LogicalResult {
  if (packageName.empty())
    packageName = _currentPackageName;
  _genericFunctionPackages[std::string(name)] = std::string(packageName);
  return _symbols.declareGenericFunction(name, decl);
}

auto SemaImpl::sema(Module &node) -> llvm::LogicalResult {
  for (auto &decl : node)
    if (llvm::failed(sema(decl.get())))
      return failure();

  for (size_t i = 0; i < _instantiatedFunctions.size(); ++i) {
    if (llvm::failed(sema(_instantiatedFunctions[i].get())))
      return failure();
  }

  auto declarations = node.takeDeclarations();
  for (auto &function : _instantiatedFunctions)
    declarations.push_back(std::move(function));
  for (auto &function : _lambdaFunctions)
    declarations.push_back(std::move(function));
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

} // end namespace mulberry
