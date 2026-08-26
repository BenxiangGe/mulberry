//===--- SemaExpressions.cpp - Expression semantic analysis -------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "SemaExpressions.h"
#include "SemaComptime.h"
#include "SemaDeclarations.h"
#include "SemaImpl.h"
#include "SemaSupport.h"
#include "SemaStatements.h"
#include "SemaTraits.h"
#include "Symbols.h"
#include "mulberry/Basic/Builtins.h"
#include "llvm/Support/Debug.h"
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using namespace mulberry;
using llvm::cast;
using llvm::dyn_cast;

#undef DEBUG_TYPE
#define DEBUG_TYPE "Sema"

struct ResultTypeArguments {
  const Type *valueType;
  const Type *errorType;
};

auto getResultTypeArguments(const Type *type)
    -> std::optional<ResultTypeArguments> {
  auto *dataType = getDataType(type);
  if (!dataType || dataType->declarationName() != "std.result.Result")
    return std::nullopt;

  auto &arguments = dataType->arguments();
  auto &constructors = dataType->constructors();
  if (arguments.size() != 2 ||
      arguments[0].kind() != ComptimeValue::Kind::Type ||
      arguments[1].kind() != ComptimeValue::Kind::Type ||
      constructors.size() != 2 ||
      constructors[0].name() != "std.result.Ok" ||
      constructors[1].name() != "std.result.Err" ||
      constructors[0].payloadTypes().size() != 1 ||
      constructors[1].payloadTypes().size() != 1 ||
      !sameType(constructors[0].payloadTypes()[0], arguments[0].type()) ||
      !sameType(constructors[1].payloadTypes()[0], arguments[1].type()))
    return std::nullopt;

  return ResultTypeArguments{arguments[0].type(), arguments[1].type()};
}

auto isIntegerLikeType(const Type *type) -> bool {
  return isIntegerType(type) || isFixedWidthIntegerType(type);
}

auto isFloatingPointType(const Type *type) -> bool {
  return isFloat32Type(type) || isFloat64Type(type);
}

// Static values are part of a concrete function symbol, so encode both their
// kind and complete value representation. This keeps distinct staged values
// from sharing a specialization when more comptime parameter kinds are added.
auto appendComptimeValueKey(std::string &result,
                            const ComptimeValue &value) -> void {
  static constexpr char hex[] = "0123456789abcdef";

  auto appendHex = [&](std::string_view bytes) {
    for (auto character : bytes) {
      auto byte = static_cast<unsigned char>(character);
      result += hex[byte >> 4];
      result += hex[byte & 0xf];
    }
  };

  switch (value.kind()) {
  case ComptimeValue::Kind::Type:
    result += "__type_";
    appendHex(formatType(value.type()));
    return;
  case ComptimeValue::Kind::Bool:
    result += "__bool_";
    result += value.boolValue() ? "1" : "0";
    return;
  case ComptimeValue::Kind::UInt8:
    result += "__u8_";
    result += std::to_string(value.uint8Value());
    return;
  case ComptimeValue::Kind::UInt64:
    result += "__u64_";
    result += std::to_string(value.uint64Value());
    return;
  case ComptimeValue::Kind::String:
    result += "__str_";
    appendHex(value.stringValue());
    return;
  }
}

auto getStdlibListElementType(const Type *type) -> const Type * {
  auto *structType = getStructType(type);
  if (!structType)
    return nullptr;

  auto *origin = structType->origin();
  if (!origin || origin->aliasName() != "std.list.List")
    return nullptr;

  auto &fields = structType->fields();
  if (fields.size() != 3)
    return nullptr;
  if (fields[0].name() != "length" || !isUInt64Type(fields[0].type()))
    return nullptr;
  if (fields[1].name() != "capacity" || !isUInt64Type(fields[1].type()))
    return nullptr;
  if (fields[2].name() != "data")
    return nullptr;

  auto *dataPtrType = getPtrType(fields[2].type());
  if (!dataPtrType)
    return nullptr;
  return dataPtrType->pointeeType();
}

auto getTensorElementType(const Type *type) -> const Type * {
  auto *structType = getStructType(type);
  if (!structType)
    return nullptr;

  auto *origin = structType->origin();
  if (!origin || origin->aliasName() != "std.tensor.Tensor")
    return nullptr;

  auto &arguments = origin->arguments();
  if (arguments.size() != 1)
    return nullptr;
  if (arguments[0].kind() != ComptimeValue::Kind::Type)
    return nullptr;
  return arguments[0].type();
}

auto getTensorStorageElementType(const Type *type) -> const Type * {
  auto *structType = getStructType(type);
  if (!structType)
    return nullptr;

  auto *origin = structType->origin();
  if (!origin || origin->aliasName() != "std.tensor.TensorStorage")
    return nullptr;

  auto &arguments = origin->arguments();
  if (arguments.size() != 1 ||
      arguments[0].kind() != ComptimeValue::Kind::Type)
    return nullptr;
  return arguments[0].type();
}

} // end namespace

namespace mulberry {
using llvm::cast;
using llvm::dyn_cast;

auto ExpressionSema::resolveFunctionName(std::string_view name)
    -> std::string {
  if (_sema._symbols.lookupFunction(name))
    return std::string(name);

  auto importedName = _sema.canonicalizeImportedName(name);
  if (_sema._symbols.lookupFunction(importedName))
    return importedName;

  auto qualifiedName = _sema.qualifyCurrentPackageName(name);
  if (_sema._symbols.lookupFunction(qualifiedName))
    return qualifiedName;

  return {};
}

auto ExpressionSema::sema(Expr *node) -> llvm::LogicalResult {
  switch (node->getKind()) {
  case Expr::Expr_Unit:
    return sema(cast<UnitExpr>(node));
  case Expr::Expr_Lambda:
    return sema(cast<LambdaExpr>(node));
  case Expr::Expr_Match:
    return sema(cast<MatchExpr>(node));
  case Expr::Expr_Try:
    return sema(cast<TryExpr>(node));
  case Expr::Expr_DataConstructor:
    return sema(cast<DataConstructorExpr>(node));
  case Expr::Expr_IntegerLiteral:
    return sema(cast<IntegerLiteralExpr>(node));
  case Expr::Expr_IntegerWiden:
    return sema(cast<IntegerWidenExpr>(node));
  case Expr::Expr_FloatLiteral:
    return sema(cast<FloatLiteralExpr>(node));
  case Expr::Expr_BoolLiteral:
    return sema(cast<BoolLiteralExpr>(node));
  case Expr::Expr_StringLiteral:
    return sema(cast<StringLiteralExpr>(node));
  case Expr::Expr_ObjectIdentity:
    return sema(cast<ObjectIdentityExpr>(node));
  case Expr::Expr_CharLiteral:
    return sema(cast<CharLiteralExpr>(node));
  case Expr::Expr_TypeInfo:
    return _sema.emitError(node, diag::expected_comptime_value);
  case Expr::Expr_CompileError:
    return sema(cast<CompileErrorExpr>(node));
  case Expr::Expr_ComptimeBlock:
    return sema(cast<ComptimeBlockExpr>(node));
  case Expr::Expr_TypeLayout:
    return sema(cast<TypeLayoutExpr>(node));
  case Expr::Expr_HeapAlloc:
    return sema(cast<HeapAllocExpr>(node));
  case Expr::Expr_Block:
    return StatementSema(_sema).sema(cast<BlockExpr>(node));
  case Expr::Expr_Call:
    return sema(cast<CallExpr>(node));
  case Expr::Expr_StructLiteral:
    return sema(cast<StructLiteralExpr>(node));
  case Expr::Expr_Variable:
    return sema(cast<VariableExpr>(node));
  case Expr::Expr_Member:
    return sema(cast<MemberExpr>(node));
  case Expr::Expr_Assign:
    return sema(cast<AssignExpr>(node));
  case Expr::Expr_ArrayLiteral:
    return sema(cast<ArrayLiteralExpr>(node));
  case Expr::Expr_Index:
    return sema(cast<IndexExpr>(node));
  case Expr::Expr_Binary:
    return sema(cast<BinaryExpr>(node));
  }
  llvm_unreachable("Unexpected expression");
}

auto ExpressionSema::sema(Expr *node, const Type *type) -> llvm::LogicalResult {
  if (auto *matchExpr = dyn_cast<MatchExpr>(node))
    return sema(matchExpr, type);

  if (auto *integerLiteral = dyn_cast<IntegerLiteralExpr>(node))
    return sema(integerLiteral, type);

  if (auto *floatLiteral = dyn_cast<FloatLiteralExpr>(node))
    return sema(floatLiteral, type);

  if (auto *binary = dyn_cast<BinaryExpr>(node))
    return sema(binary, type);

  if (auto *constructor = dyn_cast<DataConstructorExpr>(node)) {
    auto *dataType = getDataType(type);
    if (!dataType)
      return _sema.emitError(constructor, diag::mismatch_type);
    return sema(constructor, dataType);
  }

  if (auto *lambda = dyn_cast<LambdaExpr>(node)) {
    auto *functionType = getFunctionType(type);
    if (!functionType)
      return _sema.emitError(lambda, diag::mismatch_type);
    return sema(lambda, functionType);
  }

  auto *arrayLiteral = dyn_cast<ArrayLiteralExpr>(node);
  if (arrayLiteral) {
    // Source `[...]` defaults to Array. Explicit Array annotations keep
    // target-typed literal semantics.
    if (auto *arrayType = mulberry::getArrayType(type))
      return sema(arrayLiteral, arrayType);
  }

  auto *call = dyn_cast<CallExpr>(node);
  if (call && !call->hasReceiver()) {
    if (auto *callee = _sema.lookupVariable(call->name())) {
      auto *functionType = getFunctionType(callee->type);
      if (!functionType)
        return _sema.emitError(call, diag::mismatch_type);
      return semaIndirectCall(call, functionType, type);
    }

    auto callName = call->name();
    auto dot = callName.find('.');
    if (dot != std::string_view::npos &&
        _sema.lookupVariable(callName.substr(0, dot)))
      return semaDottedMethodCall(call, type);

    auto name = _sema.canonicalizeImportedName(call->name());
    call->setName(name);
    if (auto *handler = _sema.lookupBuiltinHandler(name)) {
      LLVM_DEBUG(llvm::dbgs() << "dispatch builtin Sema handler `" << name
                              << "`\n");
      return (*handler)(call, type);
    }
  }

  if (call && call->hasReceiver())
    return semaMethodCall(call, type);

  if (call) {
    auto name = _sema.canonicalizeImportedName(call->name());
    if (auto *genericFunction = _sema.lookupGenericFunction(name)) {
      call->setName(name);
      return semaGenericCall(call, genericFunction, type);
    }
  }

  return sema(node);
}

auto ExpressionSema::semaExpected(std::unique_ptr<Expr> &node, const Type *type)
    -> llvm::LogicalResult {
  if (llvm::failed(sema(node.get(), type)))
    return failure();

  if (sameType(node->type(), type))
    return success();

  if (!isIntegerWidening(node->type(), type))
    return _sema.emitError(node.get(), diag::mismatch_type);

  auto value = std::move(node);
  node = std::make_unique<IntegerWidenExpr>(value->location(),
                                             std::move(value));
  node->setType(type);
  LLVM_DEBUG(llvm::dbgs() << "widen fixed-width integer to Integer\n");
  return success();
}

auto ExpressionSema::sema(DataConstructorExpr *node) -> llvm::LogicalResult {
  std::string resolvedName;
  auto *symbol = _sema.lookupDataConstructor(node->name(), resolvedName);
  if (!symbol)
    return _sema.emitError(node, diag::undefined_data_constructor);
  if (symbol->decl->isGeneric())
    return _sema.emitError(node, diag::mismatch_type);

  auto *dataType = getDataType(_sema.lookupType(symbol->decl->name()));
  if (!dataType)
    return _sema.emitError(node, diag::mismatch_type);
  return sema(node, dataType);
}

auto ExpressionSema::sema(DataConstructorExpr *node,
                    const DataType *dataType) -> llvm::LogicalResult {
  std::string resolvedName;
  auto *symbol = _sema.lookupDataConstructor(node->name(), resolvedName);
  if (!symbol)
    return _sema.emitError(node, diag::undefined_data_constructor);
  if (symbol->decl->name() != dataType->declarationName())
    return _sema.emitError(node, diag::mismatch_type);

  auto &constructors = dataType->constructors();
  if (symbol->index >= constructors.size())
    return _sema.emitError(node, diag::mismatch_type);
  auto &constructor = constructors[symbol->index];
  auto &expressions = node->expressions();
  if (expressions.size() != constructor.payloadTypes().size())
    return _sema.emitError(node, diag::wrong_num_arg);

  for (size_t i = 0; i < expressions.size(); ++i) {
    auto *payloadType = constructor.payloadTypes()[i];
    if (llvm::failed(semaExpected(expressions[i], payloadType)))
      return failure();
  }

  node->setName(resolvedName);
  node->setConstructorIndex(symbol->index);
  node->setType(dataType);
  LLVM_DEBUG(llvm::dbgs() << "bind data constructor `" << resolvedName
                          << "` as " << formatType(dataType) << " tag "
                          << symbol->index << "\n");
  return success();
}

auto ExpressionSema::resolveSubstitutedType(
    const TypeNode *typeNode,
    const std::vector<TypeSubstitution> &substitutions) -> const Type * {
  auto substitutedTypeNode = substituteTypeNode(typeNode, substitutions);
  return _sema.resolveType(substitutedTypeNode.get());
}

auto ExpressionSema::genericFunctionName(std::string_view name,
                                          const Type *argumentType) const
    -> std::string {
  std::string result = mangleTypeName(std::string(name));
  result += "__";
  result += mangleTypeName(formatType(argumentType));
  return result;
}

auto ExpressionSema::genericFunctionName(
    std::string_view name,
    const std::vector<InferredComptimeArgument> &arguments) const
    -> std::string {
  return genericFunctionName(name, arguments, {});
}

auto ExpressionSema::genericFunctionName(
    std::string_view name,
    const std::vector<InferredComptimeArgument> &arguments,
    const std::vector<ComptimeValue> &comptimeArguments) const
    -> std::string {
  std::string result = mangleTypeName(std::string(name));
  for (auto &argument : arguments) {
    result += "__";
    if (argument.kind == ComptimeParam::Kind::Type)
      result += mangleTypeName(formatType(argument.type));
    else if (argument.kind == ComptimeParam::Kind::TypePack) {
      result += "pack";
      for (auto *type : argument.types) {
        result += "_";
        result += mangleTypeName(formatType(type));
      }
    }
    else
      result += std::to_string(*argument.uint64Value);
  }
  for (auto &argument : comptimeArguments)
    appendComptimeValueKey(result, argument);
  return result;
}

auto ExpressionSema::genericFunctionPackageName(
    std::string_view name) const -> std::string {
  auto package = _sema._genericFunctionPackages.find(std::string(name));
  if (package != _sema._genericFunctionPackages.end())
    return package->second;
  return packageNameOf(name);
}

auto ExpressionSema::sameCallArgumentType(const Type *parameterType,
                                           const Type *actualType,
                                           bool allowAddressOf) const -> bool {
  if (sameType(parameterType, actualType))
    return true;
  if (!allowAddressOf)
    return false;

  auto *ptrType = getPtrType(parameterType);
  return ptrType && sameType(ptrType->pointeeType(), actualType);
}

auto ExpressionSema::instantiateGenericFunction(
    const Node *diagnosticNode, std::string_view name,
    const Type *argumentType, std::string &concreteName)
    -> llvm::LogicalResult {
  auto *symbol = _sema.lookupGenericFunction(name);
  if (!symbol) {
    auto diagnostic = formatNameDiagnostic(diag::undefined_func, name);
    return _sema.emitError(diagnosticNode, diagnostic);
  }

  auto *genericFunction = symbol->decl;
  auto *genericProto = genericFunction->proto().get();
  auto &genericParameters = genericProto->comptimeParameters();
  if (genericParameters.size() != 1 ||
      genericParameters.front().kind != ComptimeParam::Kind::Type)
    return _sema.emitError(diagnosticNode, diag::mismatch_type);

  concreteName = genericFunctionName(name, argumentType);

  auto cached = _sema._instantiatedFunctionSymbols.find(concreteName);
  if (cached != _sema._instantiatedFunctionSymbols.end())
    return success();

  auto argumentTypeNode =
      typeToTypeNode(argumentType, genericProto->location());
  auto concreteFunction = instantiateFunctionDecl(
      genericFunction, concreteName,
      std::vector<TypeSubstitution>{TypeSubstitution{
          genericParameters.front().name, argumentTypeNode.get(), std::nullopt}});
  concreteFunction->setSpecializationOrigin(name, diagnosticNode->location());
  _sema._instantiatedFunctionPackages[concreteName] =
      genericFunctionPackageName(name);

  SemaImpl::VariableScope signatureScope(_sema._symbols);
  SemaImpl::PackageScope packageScope(
      _sema._currentPackageName, genericFunctionPackageName(name));
  if (llvm::failed(DeclarationSema(_sema).semaFunctionSignature(
          concreteFunction->proto().get(), /*isExtern=*/false,
          genericFunctionPackageName(name), concreteFunction->visibility())))
    return failure();
  auto *signature = _sema.lookupFunction(concreteName);
  if (!signature)
    return failure();

  _sema._instantiatedFunctionSymbols.insert({concreteName, signature});
  _sema.registerFunctionDecl(concreteName, concreteFunction.get());
  _sema._instantiatedFunctions.push_back(std::move(concreteFunction));
  return success();
}

auto ExpressionSema::semaGenericCall(CallExpr *node,
                                     const GenericFunctionSymbol *symbol,
                                     const Type *expectedType)
    -> llvm::LogicalResult {
  if (llvm::failed(expandPackArguments(node)))
    return failure();
  auto *genericFunction = symbol->decl;
  auto *genericProto = genericFunction->proto().get();
  auto name = genericProto->id()->name();
  auto callerPackageName = _sema._currentPackageName;
  SemaImpl::PackageScope packageScope(_sema._currentPackageName,
                                      genericFunctionPackageName(name));
  auto semaCallerArgument = [&](std::unique_ptr<Expr> &argument) {
    SemaImpl::PackageScope callerScope(_sema._currentPackageName,
                                       callerPackageName);
    return sema(argument.get());
  };
  auto semaCallerExpected = [&](std::unique_ptr<Expr> &argument,
                                const Type *type) {
    SemaImpl::PackageScope callerScope(_sema._currentPackageName,
                                       callerPackageName);
    return semaExpected(argument, type);
  };
  auto &expressions = node->expressions();
  auto &parameters = genericProto->parameters();
  auto valuePackIndex = parameters.size();
  for (size_t index = 0; index < parameters.size(); ++index) {
    if (parameters[index]->isPack()) {
      valuePackIndex = index;
      break;
    }
  }

  auto fixedParameterCount = valuePackIndex == parameters.size()
                                 ? parameters.size()
                                 : valuePackIndex;
  if (expressions.size() < fixedParameterCount ||
      (valuePackIndex == parameters.size() &&
       expressions.size() != parameters.size())) {
    auto diagnostic =
        formatNameSizeDiagnostic(diag::func_param, name, fixedParameterCount);
    return _sema.emitError(node, diagnostic);
  }

  auto parameterForArgument = [&](size_t index) -> const ParameterDecl * {
    if (valuePackIndex != parameters.size() && index >= valuePackIndex)
      return parameters[valuePackIndex].get();
    return parameters[index].get();
  };

  auto &comptimeParameters = genericProto->comptimeParameters();
  auto inferredArguments = makeInferredComptimeArguments(comptimeParameters);
  std::vector<ComptimeValue> comptimeArguments;
  std::vector<const Type *> arrayLeafConstraints(comptimeParameters.size());
  auto returnHasComputedType =
      hasComputedType(genericProto->returnTypeNode());
  if (expectedType) {
    if (!matchGenericType(genericProto->returnTypeNode(), expectedType,
                          comptimeParameters, inferredArguments,
                          &arrayLeafConstraints)) {
      inferredArguments = makeInferredComptimeArguments(comptimeParameters);
      arrayLeafConstraints.assign(comptimeParameters.size(), nullptr);
      if (!returnHasComputedType)
        return _sema.emitError(node, diag::mismatch_type);
      LLVM_DEBUG(llvm::dbgs()
                 << "defer computed return type of `" << name << "`\n");
    }
  }

  auto semaArgument = [&](std::unique_ptr<Expr> &argument,
                          const TypeNode *parameterTypeNode)
      -> llvm::LogicalResult {
    auto *literal = dyn_cast<ArrayLiteralExpr>(argument.get());
    auto *namedType = dyn_cast<NamedTypeNode>(parameterTypeNode);
    auto parameterIndex = namedType
                              ? comptimeParameterIndex(
                                    comptimeParameters, namedType->name())
                              : std::nullopt;
    if (literal && parameterIndex &&
        !inferredArguments[*parameterIndex].isResolved() &&
        arrayLeafConstraints[*parameterIndex]) {
      auto *arrayType = arrayLiteralTypeWithLeaf(
          literal, arrayLeafConstraints[*parameterIndex]);
      if (!arrayType)
        return _sema.emitError(literal, diag::expected_expr);
      LLVM_DEBUG(llvm::dbgs()
                 << "target Array literal leaf from computed return of `"
                 << name << "`: " << formatType(arrayType) << "\n");
      return semaCallerExpected(argument, arrayType);
    }

    if (parameterIndex &&
        comptimeParameters[*parameterIndex].kind ==
            ComptimeParam::Kind::TypePack)
      return semaCallerArgument(argument);

    auto knownArguments = true;
    for (auto &inferredArgument : inferredArguments)
      knownArguments = knownArguments && inferredArgument.isResolved();
    if (!knownArguments)
      return semaCallerArgument(argument);

    auto *parameterType = resolveSubstitutedType(
        parameterTypeNode,
        comptimeSubstitutions(comptimeParameters, inferredArguments));
    if (!parameterType)
      return failure();
    return semaCallerExpected(argument, parameterType);
  };

  // Sibling arguments often determine a lambda's generic parameter types.
  // Analyze ordinary arguments first, then infer the callback result from its
  // body and feed that type back into generic matching.
  std::vector<size_t> deferredLambdas;
  std::vector<size_t> deferredParameters;
  for (size_t i = 0; i < expressions.size(); ++i) {
    auto *parameter = parameterForArgument(i);
    auto *parameterTypeNode = parameter->typeNode();
    if (parameter->isComptime()) {
      auto *parameterType = _sema.resolveType(parameterTypeNode);
      auto *stringType = _sema.lookupType("String");
      if (!parameterType || !stringType ||
          !sameType(parameterType, stringType))
        return _sema.emitError(expressions[i].get(), diag::mismatch_type);
      if (llvm::failed(semaCallerExpected(expressions[i], parameterType)))
        return failure();

      auto evaluation = ComptimeSema(_sema).evaluateComptime(
          expressions[i].get());
      if (evaluation.kind == ComptimeEvaluation::Kind::Error)
        return failure();
      if (evaluation.kind != ComptimeEvaluation::Kind::Static ||
          !evaluation.value ||
          evaluation.value->kind() != ComptimeValue::Kind::String)
        return _sema.emitError(expressions[i].get(),
                               diag::expected_comptime_value);
      comptimeArguments.push_back(*evaluation.value);
      LLVM_DEBUG(llvm::dbgs() << "bind comptime String parameter " << i
                              << " of `" << name << "`\n");
      continue;
    }
    if (dyn_cast<LambdaExpr>(expressions[i].get())) {
      deferredLambdas.push_back(i);
      LLVM_DEBUG(llvm::dbgs()
                 << "defer lambda parameter " << i << " of `" << name
                 << "`\n");
      continue;
    }

    if (hasComputedType(parameterTypeNode)) {
      deferredParameters.push_back(i);
      LLVM_DEBUG(llvm::dbgs()
                 << "defer computed parameter " << i << " of `" << name
                 << "`\n");
      continue;
    }

    if (llvm::failed(semaArgument(expressions[i], parameterTypeNode)))
      return failure();
    auto matched =
        node->isLoweredMethodCall() && i == 0
            ? matchMethodReceiverType(parameterTypeNode, expressions[i]->type(),
                                      comptimeParameters, inferredArguments)
            : matchGenericType(parameterTypeNode, expressions[i]->type(),
                               comptimeParameters, inferredArguments);
    if (!matched)
      return _sema.emitError(expressions[i].get(), diag::mismatch_type);
  }

  for (auto index : deferredLambdas) {
    auto *functionPattern =
        dyn_cast<FunctionTypeNode>(parameterForArgument(index)->typeNode());
    if (!functionPattern)
      return _sema.emitError(expressions[index].get(), diag::mismatch_type);

    std::vector<ComptimeParam> unresolvedParameters;
    for (size_t i = 0; i < comptimeParameters.size(); ++i)
      if (!inferredArguments[i].isResolved())
        unresolvedParameters.push_back(comptimeParameters[i]);

    auto substitutions =
        comptimeSubstitutions(comptimeParameters, inferredArguments);
    std::vector<const Type *> lambdaParameterTypes;
    for (auto &parameterTypeNode : functionPattern->parameterTypes()) {
      if (containsComptimeParameter(parameterTypeNode.get(),
                                    unresolvedParameters))
        return _sema.emitError(expressions[index].get(), diag::mismatch_type);

      auto *parameterType =
          resolveSubstitutedType(parameterTypeNode.get(), substitutions);
      if (!parameterType)
        return failure();
      lambdaParameterTypes.push_back(parameterType);
    }

    auto *lambda = cast<LambdaExpr>(expressions[index].get());
    if (llvm::failed(semaLambda(lambda, lambdaParameterTypes,
                   functionPattern->parameterCanMutateObject(),
                   /*expectedReturnType=*/nullptr, callerPackageName)))
      return failure();
    if (!matchGenericType(functionPattern, lambda->type(), comptimeParameters,
                          inferredArguments))
      return _sema.emitError(lambda, diag::mismatch_type);
  }

  if (valuePackIndex != parameters.size()) {
    auto *packType = dyn_cast<NamedTypeNode>(
        parameters[valuePackIndex]->typeNode());
    auto packIndex = packType
                         ? comptimeParameterIndex(comptimeParameters,
                                                   packType->name())
                         : std::nullopt;
    if (!packIndex ||
        comptimeParameters[*packIndex].kind != ComptimeParam::Kind::TypePack)
      return _sema.emitError(parameters[valuePackIndex].get(),
                             diag::mismatch_type);
    inferredArguments[*packIndex].packResolved = true;
    LLVM_DEBUG(llvm::dbgs() << "resolve type pack `" << packType->name()
                            << "` with "
                            << inferredArguments[*packIndex].types.size()
                            << " elements\n");
  }

  for (auto &argument : inferredArguments)
    if (!argument.isResolved())
      return _sema.emitError(node, diag::mismatch_type);

  if (llvm::failed(TraitSema(_sema).checkConstraints(
          node, comptimeParameters, inferredArguments)))
    return failure();

  auto substitutions =
      comptimeSubstitutions(comptimeParameters, inferredArguments);
  for (auto index : deferredParameters) {
    auto *parameterType = resolveSubstitutedType(
        parameterForArgument(index)->typeNode(), substitutions);
    if (!parameterType)
      return failure();

    auto &argument = expressions[index];
    if (node->isLoweredMethodCall() && index == 0) {
      if (llvm::failed(semaCallerArgument(argument)))
        return failure();
    } else if (llvm::failed(semaCallerExpected(argument, parameterType))) {
      return failure();
    }
  }

  VectorUniquePtr<Expr> runtimeExpressions;
  for (size_t i = 0; i < expressions.size(); ++i)
    if (!parameterForArgument(i)->isComptime())
      runtimeExpressions.push_back(std::move(expressions[i]));
  expressions = std::move(runtimeExpressions);

  auto concreteName =
      genericFunctionName(name, inferredArguments, comptimeArguments);
  LLVM_DEBUG(llvm::dbgs() << "specialize generic `" << name << "` as `"
                          << concreteName << "`\n");
  auto cached = _sema._instantiatedFunctionSymbols.find(concreteName);
  if (cached == _sema._instantiatedFunctionSymbols.end()) {
    auto concreteFunction = instantiateFunctionDecl(
        genericFunction, concreteName, substitutions, comptimeArguments,
        &inferredArguments);
    concreteFunction->setSpecializationOrigin(name, node->location());
    _sema._instantiatedFunctionPackages[concreteName] =
        genericFunctionPackageName(name);

    SemaImpl::VariableScope signatureScope(_sema._symbols);
    if (llvm::failed(DeclarationSema(_sema).semaFunctionSignature(
            concreteFunction->proto().get(), /*isExtern=*/false,
            genericFunctionPackageName(name), concreteFunction->visibility())))
      return failure();
    auto *signature = _sema.lookupFunction(concreteName);
    if (!signature)
      return failure();

    cached = _sema._instantiatedFunctionSymbols
                 .insert({concreteName, signature})
                 .first;
    _sema.registerFunctionDecl(concreteName, concreteFunction.get());
    _sema._instantiatedFunctions.push_back(std::move(concreteFunction));
  }

  auto *signature = cached->second;
  if (expectedType &&
      !sameType(expectedType, signature->type->returnType()))
    return _sema.emitError(node, diag::mismatch_type);

  for (size_t i = 0; i < expressions.size(); ++i) {
    auto &arg = expressions[i];
    auto *parameterType = signature->type->parameterTypes()[i];
    if (!sameCallArgumentType(parameterType, arg->type(),
                              node->isLoweredMethodCall() && i == 0))
      return _sema.emitError(arg.get(), diag::mismatch_type);
    if (llvm::failed(checkMutableObjectArgument(signature->type, i, arg.get())))
      return failure();
  }

  node->setName(concreteName);
  node->setType(signature->type->returnType());
  return success();
}

auto ExpressionSema::sema(UnitExpr *node) -> llvm::LogicalResult {
  _sema.setBuiltinType(node, BuiltinTypeKind::Unit);
  return success();
}

auto ExpressionSema::sema(ComptimeBlockExpr *node) -> llvm::LogicalResult {
  auto evaluation = ComptimeSema(_sema).evaluateComptime(node);
  if (evaluation.kind == ComptimeEvaluation::Kind::Error)
    return failure();
  if (evaluation.kind != ComptimeEvaluation::Kind::Static)
    return _sema.emitError(node, diag::expected_comptime_value);
  return success();
}

auto ExpressionSema::sema(CompileErrorExpr *node) -> llvm::LogicalResult {
  auto evaluation = ComptimeSema(_sema).evaluateComptime(node);
  if (evaluation.kind == ComptimeEvaluation::Kind::Error)
    return failure();
  return _sema.emitError(node, diag::expected_comptime_value);
}

auto ExpressionSema::sema(LambdaExpr *node) -> llvm::LogicalResult {
  return _sema.emitError(node, diag::lambda_expected_function_type);
}

auto ExpressionSema::sema(MatchExpr *node, const Type *expectedType)
    -> llvm::LogicalResult {
  auto *value = node->value().get();
  if (llvm::failed(sema(value)))
    return failure();

  auto *dataType = getDataType(value->type());
  if (!dataType)
    return _sema.emitError(value, diag::match_expected_data_type);

  std::vector<bool> covered(dataType->constructors().size(), false);
  auto *resultType = expectedType;
  auto canMutateObject = true;
  StatementSema statementSema(_sema);
  for (auto &arm : node->arms()) {
    auto *pattern = arm->pattern().get();
    const DataConstructor *constructor = nullptr;
    if (llvm::failed(statementSema.checkMatchPattern(pattern, dataType, covered,
                                                     constructor)))
      return failure();
    if (containsReturnStat(arm->bodyBlock().get()))
      return _sema.emitError(arm.get(), diag::match_expression_arm_return);

    SemaImpl::VariableScope patternScope(_sema._symbols);
    if (llvm::failed(
            statementSema.declareMatchPatternBindings(pattern, constructor)))
      return failure();

    SemaImpl::VariableScope bodyScope(_sema._symbols);
    SemaImpl::IsolatedWhileScope isolatedWhileScope(_sema._whileDepth);
    for (auto &statement : arm->bodyBlock()->statements())
      if (llvm::failed(_sema.sema(statement.get())))
        return failure();
    arm->bodyBlock()->setType(
        _sema._typeContext.getBuiltinType(BuiltinTypeKind::Unit));

    auto &armResult = arm->resultExpr();
    if (resultType) {
      if (llvm::failed(semaExpected(armResult, resultType)))
        return failure();
    } else {
      if (llvm::failed(sema(armResult.get())))
        return failure();
      resultType = armResult->type();
    }
    if (!resultType || !sameType(resultType, armResult->type()))
      return _sema.emitError(armResult.get(), diag::mismatch_type);
    if (isMutableSourceObjectType(resultType) &&
        !canMutateObjectReference(armResult.get()))
      canMutateObject = false;
  }

  if (llvm::failed(statementSema.checkExhaustiveMatch(node, covered)))
    return failure();
  if (!resultType)
    return _sema.emitError(node, diag::mismatch_type);
  node->setType(resultType);
  node->setCanMutateObject(canMutateObject);
  return success();
}

auto ExpressionSema::sema(TryExpr *node) -> llvm::LogicalResult {
  if (!_sema._currentFunctionReturnType)
    return _sema.emitError(node, diag::try_outside_result_function);

  auto *value = node->value().get();
  if (llvm::failed(sema(value)))
    return failure();

  auto operandTypes = getResultTypeArguments(value->type());
  if (!operandTypes)
    return _sema.emitError(value, diag::try_expected_result);

  auto functionTypes = getResultTypeArguments(_sema._currentFunctionReturnType);
  if (!functionTypes)
    return _sema.emitError(node, diag::try_outside_result_function);

  node->setErrorTypes(operandTypes->errorType, functionTypes->errorType);
  if (!sameType(operandTypes->errorType, functionTypes->errorType)) {
    std::string functionName;
    if (llvm::failed(TraitSema(_sema).resolveFromConversion(
            node, operandTypes->errorType, functionTypes->errorType,
            functionName)))
      return failure();
    if (functionName.empty())
      return _sema.emitError(
          node, formatMissingFromDiagnostic(operandTypes->errorType,
                                            functionTypes->errorType));
    node->setErrorConverter(std::move(functionName));
  }

  node->setType(operandTypes->valueType);
  node->setCanMutateObject(canMutateObjectReference(value));
  LLVM_DEBUG(llvm::dbgs() << "propagate `" << formatType(value->type())
                          << "` error from `?`\n");
  return success();
}

auto ExpressionSema::sema(LambdaExpr *node, const FunctionType *functionType)
    -> llvm::LogicalResult {
  auto packageName = _sema._currentPackageName;
  return semaLambda(node, functionType->parameterTypes(),
                    functionType->parameterCanMutateObject(),
                    functionType->returnType(), packageName);
}

auto ExpressionSema::semaLambda(
    LambdaExpr *node, const std::vector<const Type *> &parameterTypes,
    const std::vector<bool> &parameterCanMutateObject,
    const Type *expectedReturnType, std::string_view packageName)
    -> llvm::LogicalResult {
  auto &parameters = node->parameters();
  if (parameters.size() != parameterTypes.size())
    return _sema.emitError(node, diag::wrong_num_arg);

  SemaImpl::PackageScope packageScope(_sema._currentPackageName, packageName);
  SemaImpl::VariableScope variableScope(_sema._symbols);
  SemaImpl::NoncapturingLambdaScope lambdaScope(_sema._noncapturingLambdaDepth);
  for (size_t i = 0; i < parameters.size(); ++i) {
    auto &parameter = parameters[i];
    parameter.type = parameterTypes[i];
    auto canMutateObject = parameterCanMutateObject[i];
    if (llvm::failed(_sema.declareVariable(parameter.name, parameter.type,
                        /*isConstBinding=*/!canMutateObject,
                        canMutateObject)))
      return _sema.emitError(parameter.location, diag::redefinition_var);
  }

  auto &body = node->body();
  SemaImpl::FunctionReturnTypeScope returnTypeScope(_sema._currentFunctionReturnType,
                                          expectedReturnType);
  if (expectedReturnType ? llvm::failed(semaExpected(body, expectedReturnType))
                         : llvm::failed(sema(body.get())))
    return failure();
  auto *returnType = body->type();
  if (!returnType)
    return _sema.emitError(body.get(), diag::mismatch_type);
  if (expectedReturnType && !sameReturnType(expectedReturnType, returnType))
    return _sema.emitError(body.get(), diag::wrong_return_type);

  auto *functionType = _sema._typeContext.createFunctionType(
      std::vector<const Type *>(parameterTypes.begin(), parameterTypes.end()),
      parameterCanMutateObject, returnType);
  node->setType(functionType);

  // A noncapturing lambda is a normal private function plus a function value;
  // no closure environment or lambda-specific lowering is needed.
  std::string functionName =
      "__mulberry_lambda_" + std::to_string(_sema._lambdaCounter++);
  node->setFunctionName(functionName);

  VectorUniquePtr<ParameterDecl> parameterDecls;
  for (size_t i = 0; i < parameters.size(); ++i) {
    auto &parameter = parameters[i];
    auto variable = std::make_unique<VariableExpr>(parameter.location,
                                                   parameter.name);
    variable->setType(parameter.type);
    auto parameterDecl = std::make_unique<ParameterDecl>(
        parameter.location, std::move(variable),
        typeToTypeNode(parameter.type, parameter.location),
        parameterCanMutateObject[i]);
    parameterDecl->setType(parameter.type);
    parameterDecls.push_back(std::move(parameterDecl));
  }

  auto functionNameNode =
      std::make_unique<FunctionName>(node->location(), functionName);
  auto prototype = std::make_unique<Prototype>(
      node->location(), std::move(functionNameNode),
      std::move(parameterDecls), typeToTypeNode(returnType, node->location()));
  prototype->setType(returnType);

  VectorUniquePtr<Stat> statements;
  auto bodyExpression = node->takeBody();
  if (isUnitType(returnType)) {
    statements.push_back(std::make_unique<ExprStat>(
        bodyExpression->location(), std::move(bodyExpression)));
  } else {
    statements.push_back(std::make_unique<ReturnStat>(
        bodyExpression->location(), std::move(bodyExpression)));
  }
  auto functionBody =
      std::make_unique<BlockExpr>(node->location(), std::move(statements));
  functionBody->setType(_sema._typeContext.getBuiltinType(BuiltinTypeKind::Unit));

  if (llvm::failed(_sema.declareFunction(functionName, functionType, /*isExtern=*/false,
                      packageName))) {
    auto diagnostic =
        formatNameDiagnostic(diag::redefinition_func, functionName);
    return _sema.emitError(node, diagnostic);
  }

  LLVM_DEBUG(llvm::dbgs() << "lift lambda to `" << functionName << "` as `"
                          << formatType(functionType) << "`\n");
  _sema._lambdaFunctions.push_back(std::make_unique<FunctionDecl>(
      node->location(), std::move(prototype), std::move(functionBody)));
  return success();
}

auto ExpressionSema::expandPackArguments(CallExpr *node)
    -> llvm::LogicalResult {
  auto &expressions = node->expressions();
  auto hasExpansion = false;
  for (auto &expression : expressions)
    hasExpansion = hasExpansion || expression->isPackExpansion();
  if (!hasExpansion)
    return success();

  if (!_sema._activeComptimeFrame)
    return _sema.emitError(node, diag::comptime_pack_expansion);

  VectorUniquePtr<Expr> expandedExpressions;
  for (auto &expression : expressions) {
    if (!expression->isPackExpansion()) {
      expandedExpressions.push_back(std::move(expression));
      continue;
    }

    auto *variable = dyn_cast<VariableExpr>(expression.get());
    auto *binding = variable
                        ? _sema._activeComptimeFrame->lookup(variable->name())
                        : nullptr;
    if (!binding || binding->kind != ComptimeBinding::Kind::Pack)
      return _sema.emitError(expression.get(), diag::comptime_pack_expansion);

    for (auto &element : binding->elements) {
      if (element.kind != ComptimeBinding::Kind::Residual ||
          !element.residual)
        return _sema.emitError(expression.get(), diag::comptime_pack_escape);
      auto expanded = substituteExpr(element.residual.get(), {});
      expanded->setType(element.type);
      expandedExpressions.push_back(std::move(expanded));
    }
    LLVM_DEBUG(llvm::dbgs() << "expand value pack `" << variable->name()
                            << "` into " << binding->elements.size()
                            << " call arguments\n");
  }

  expressions = std::move(expandedExpressions);
  return success();
}

auto ExpressionSema::sema(CallExpr *node) -> llvm::LogicalResult {
  if (llvm::failed(expandPackArguments(node)))
    return failure();
  if (node->hasReceiver())
    return semaMethodCall(node);

  if (auto *callee = _sema.lookupVariable(node->name())) {
    auto *functionType = getFunctionType(callee->type);
    if (!functionType)
      return _sema.emitError(node, diag::mismatch_type);
    return semaIndirectCall(node, functionType);
  }

  auto callName = node->name();
  auto dot = callName.find('.');
  if (dot != std::string_view::npos &&
      _sema.lookupVariable(callName.substr(0, dot)))
    return semaDottedMethodCall(node);

  node->setName(_sema.canonicalizeImportedName(node->name()));
  if (node->name().rfind("std.internal.", 0) == 0 &&
      llvm::failed(_sema.checkInternalFeature(node->location())))
    return failure();

  if (auto *handler = _sema.lookupBuiltinHandler(node->name())) {
    LLVM_DEBUG(llvm::dbgs() << "dispatch builtin Sema handler `"
                            << node->name() << "`\n");
    return (*handler)(node, nullptr);
  }

  auto name = node->name();

  auto *signature = _sema.lookupFunction(name);
  if (!signature) {
    if (auto *genericFunction = _sema.lookupGenericFunction(name))
      return semaGenericCall(node, genericFunction);

    if (name.find('.') != std::string_view::npos)
      return semaDottedMethodCall(node);

    auto diagnostic = formatNameDiagnostic(diag::undefined_func, name);
    return _sema.emitError(node, diagnostic);
  }

  auto &expressions = node->expressions();
  if (expressions.size() != signature->type->parameterTypes().size()) {
    auto diagnostic = formatNameSizeDiagnostic(
        diag::func_param, name, signature->type->parameterTypes().size());
    return _sema.emitError(node, diagnostic);
  }

  for (size_t i = 0; i < expressions.size(); ++i) {
    auto &arg = expressions[i];
    auto *parameterType = signature->type->parameterTypes()[i];
    if (node->isLoweredMethodCall() && i == 0) {
      if (llvm::failed(sema(arg.get())))
        return failure();
    } else if (llvm::failed(semaExpected(arg, parameterType))) {
      return failure();
    }
    if (!sameCallArgumentType(parameterType, arg->type(),
                              node->isLoweredMethodCall() && i == 0))
      return _sema.emitError(arg.get(), diag::mismatch_type);
    if (llvm::failed(checkMutableObjectArgument(signature->type, i, arg.get())))
      return failure();
  }

  auto resolvedName = resolveFunctionName(name);
  if (!resolvedName.empty())
    node->setName(resolvedName);

  node->setType(signature->type->returnType());
  return success();
}

auto ExpressionSema::semaIndirectCall(CallExpr *node,
                                const FunctionType *functionType,
                                const Type *expectedType) -> llvm::LogicalResult {
  auto &expressions = node->expressions();
  auto &parameterTypes = functionType->parameterTypes();
  if (expressions.size() != parameterTypes.size()) {
    auto diagnostic = formatNameSizeDiagnostic(
        diag::func_param, node->name(), parameterTypes.size());
    return _sema.emitError(node, diagnostic);
  }

  for (size_t i = 0; i < expressions.size(); ++i) {
    auto &argument = expressions[i];
    if (llvm::failed(semaExpected(argument, parameterTypes[i])))
      return failure();
    if (llvm::failed(checkMutableObjectArgument(functionType, i, argument.get())))
      return failure();
  }

  auto *returnType = functionType->returnType();
  if (expectedType && !sameType(expectedType, returnType))
    return _sema.emitError(node, diag::mismatch_type);
  node->setIndirectCall();
  node->setType(returnType);
  return success();
}

auto ExpressionSema::semaMethodCall(CallExpr *node, const Type *expectedType)
    -> llvm::LogicalResult {
  if (!node->hasReceiver())
    return semaDottedMethodCall(node, expectedType);

  if (llvm::failed(sema(node->receiver().get())))
    return failure();

  auto *receiverType = node->receiver()->type();
  auto *ptrType = mulberry::getPtrType(receiverType);
  auto *structType = ptrType ? mulberry::getStructType(ptrType->pointeeType())
                             : mulberry::getStructType(receiverType);
  auto methodName = std::string(node->name());
  if (structType) {
    std::vector<std::string> owners;
    if (auto *origin = structType->origin())
      owners.push_back(std::string(origin->aliasName()));
    owners.push_back(std::string(structType->name()));

    for (auto &owner : owners) {
      auto fullName = methodFunctionName(owner, methodName);
      if (auto *genericFunction = _sema.lookupGenericFunction(fullName)) {
        node->lowerMethodCall(fullName);
        return semaGenericCall(node, genericFunction, expectedType);
      }

      if (_sema.lookupFunction(fullName)) {
        node->lowerMethodCall(fullName);
        return sema(node);
      }
    }
  }

  auto *traitReceiverType = ptrType ? ptrType->pointeeType() : receiverType;
  if (auto *functionName =
          _sema._symbols.lookupTraitMethod(traitReceiverType, methodName)) {
    LLVM_DEBUG(llvm::dbgs() << "resolve trait method `" << methodName
                            << "` for `" << formatType(traitReceiverType)
                            << "` to `" << *functionName << "`\n");
    node->lowerMethodCall(*functionName);
    return sema(node);
  }

  std::string functionName;
  if (llvm::failed(TraitSema(_sema).materializeMethod(
          node, traitReceiverType, methodName, functionName)))
    return failure();
  if (!functionName.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "resolve conditional trait method `"
                            << methodName << "` for `"
                            << formatType(traitReceiverType) << "` to `"
                            << functionName << "`\n");
    node->lowerMethodCall(functionName);
    return sema(node);
  }

  auto diagnostic = formatNameDiagnostic(diag::undefined_func, methodName);
  return _sema.emitError(node, diagnostic);
}

auto ExpressionSema::semaDottedMethodCall(CallExpr *node, const Type *expectedType)
    -> llvm::LogicalResult {
  auto name = std::string(node->name());
  auto dot = name.rfind('.');
  if (dot == std::string::npos) {
    auto diagnostic = formatNameDiagnostic(diag::undefined_func, name);
    return _sema.emitError(node, diagnostic);
  }

  auto receiverName = name.substr(0, dot);
  auto methodName = name.substr(dot + 1);
  if (auto *targetType = _sema.lookupType(receiverName)) {
    std::string functionName;
    if (llvm::failed(TraitSema(_sema).resolveStaticMethod(
            node, targetType, methodName, functionName)))
      return failure();
    if (!functionName.empty()) {
      node->setName(functionName);
      return sema(node, expectedType);
    }
    auto diagnostic = formatNameDiagnostic(diag::undefined_func, methodName);
    return _sema.emitError(node, diagnostic);
  }
  node->setReceiver(createMemberAccessChain(node->location(), receiverName),
                    methodName);
  return semaMethodCall(node, expectedType);
}

auto ExpressionSema::sema(StructLiteralExpr *node) -> llvm::LogicalResult {
  auto *type = _sema.resolveType(node->typeNode());
  auto *structType = mulberry::getStructType(type);
  if (!structType)
    return _sema.emitError(node, diag::undefined_type);
  node->setStructType(structType);

  auto &expressions = node->expressions();
  auto &fields = structType->fields();
  if (expressions.size() != fields.size())
    return _sema.emitError(node, diag::wrong_num_arg);

  for (size_t i = 0; i < expressions.size(); ++i) {
    auto &expr = expressions[i];
    auto &field = fields[i];
    if (llvm::failed(semaExpected(expr, field.type())))
      return failure();
  }

  node->setType(structType);
  return success();
}

auto ExpressionSema::sema(VariableExpr *node) -> llvm::LogicalResult {
  if (_sema._activeComptimeFrame) {
    if (auto *binding = _sema._activeComptimeFrame->lookup(node->name());
        binding && binding->kind == ComptimeBinding::Kind::Pack)
      return _sema.emitError(node, diag::comptime_pack_escape);
  }

  auto *symbol = _sema.lookupVariable(node->name());
  if (symbol && _sema._noncapturingLambdaDepth > 0 &&
      !_sema._symbols.lookupCurrentVariable(node->name()))
    return _sema.emitError(node, diag::lambda_capture);

  if (!symbol) {
    auto functionName = resolveFunctionName(node->name());
    if (functionName.empty())
      return _sema.emitError(node, diag::undefined_var);

    auto *function = _sema._symbols.lookupFunction(functionName);
    if (!function)
      return _sema.emitError(node, diag::undefined_var);
    if (function->isExtern)
      return _sema.emitError(node, diag::extern_function_value);

    node->setName(functionName);
    node->setFunctionValue();
    node->setType(function->type);
    return success();
  }

  if (symbol->isComptimeOnly) {
    assert(symbol->comptimeValue && "comptime variable has no value");
    auto *type = ComptimeSema(_sema).comptimeRuntimeType(
        *symbol->comptimeValue);
    if (!type)
      return _sema.emitError(node, diag::comptime_type_runtime);
    node->setType(type);
    node->setComptimeValue(*symbol->comptimeValue);
    return success();
  }

  node->setType(symbol->type);
  return success();
}

auto ExpressionSema::sema(MemberExpr *node) -> llvm::LogicalResult {
  if (llvm::failed(sema(node->base().get())))
    return failure();

  auto *baseType = node->base()->type();
  auto *ptrType = mulberry::getPtrType(baseType);
  auto *structType = ptrType ? mulberry::getStructType(ptrType->pointeeType())
                             : mulberry::getStructType(baseType);
  if (!structType)
    return _sema.emitError(node->base().get(), diag::mismatch_type);

  auto *field = structType->field(node->fieldName());
  if (!field)
    return _sema.emitError(node, diag::undefined_field);
  if (!field->type())
    return _sema.emitError(node, diag::mismatch_type);
  if (mulberry::isPtrType(field->type()) &&
      llvm::failed(_sema.checkInternalFeature(node->location())))
    return failure();

  node->setType(field->type());
  node->setFieldIndex(field->index());
  node->setLvalue(ptrType || node->base()->isLvalue());
  return success();
}

auto ExpressionSema::sema(IntegerLiteralExpr *node) -> llvm::LogicalResult {
  if (!node->hasValidSpelling())
    return _sema.emitError(node, diag::invalid_integer_literal);
  if (node->isNegative() ? !node->getInt64Value() : !node->getUInt64Value())
    return _sema.emitError(node, diag::integer_literal_overflows);
  _sema.setBuiltinType(node, node->isNegative() ? BuiltinTypeKind::Int64
                                                : BuiltinTypeKind::UInt64);
  LLVM_DEBUG(llvm::dbgs() << "type integer literal `" << node->spelling()
                          << "` as "
                          << (node->isNegative() ? "Int64" : "UInt64")
                          << "\n");
  return success();
}

auto ExpressionSema::sema(IntegerLiteralExpr *node, const Type *type)
    -> llvm::LogicalResult {
  if (!node->hasValidSpelling())
    return _sema.emitError(node, diag::invalid_integer_literal);

  if (isIntegerType(type)) {
    node->setType(type);
    LLVM_DEBUG(llvm::dbgs() << "type integer literal `" << node->spelling()
                            << "` as Integer\n");
    return success();
  }

  if (isInt64Type(type)) {
    if (!node->getInt64Value())
      return _sema.emitError(node, diag::integer_literal_overflows);
    node->setType(type);
    return success();
  }

  if (node->isNegative())
    return _sema.emitError(node, diag::mismatch_type);

  auto value = node->getUInt64Value();
  if (!value)
    return _sema.emitError(node, diag::integer_literal_overflows);

  if (isUInt8Type(type)) {
    if (*value > 255)
      return _sema.emitError(node, diag::mismatch_type);
    node->setType(type);
    return success();
  }

  if (isUInt64Type(type)) {
    node->setType(type);
    return success();
  }

  return sema(node);
}

auto ExpressionSema::sema(IntegerWidenExpr *node) -> llvm::LogicalResult {
  if (llvm::failed(sema(node->value().get())))
    return failure();
  if (!isIntegerWidening(
          node->value()->type(),
          _sema._typeContext.getBuiltinType(BuiltinTypeKind::Integer)))
    return _sema.emitError(node, diag::mismatch_type);
  _sema.setBuiltinType(node, BuiltinTypeKind::Integer);
  return success();
}

auto ExpressionSema::sema(FloatLiteralExpr *node) -> llvm::LogicalResult {
  _sema.setBuiltinType(node, BuiltinTypeKind::Float32);
  return success();
}

auto ExpressionSema::sema(FloatLiteralExpr *node, const Type *type)
    -> llvm::LogicalResult {
  if (isFloatingPointType(type)) {
    node->setType(type);
    return success();
  }
  return sema(node);
}

auto ExpressionSema::sema(BoolLiteralExpr *node) -> llvm::LogicalResult {
  _sema.setBuiltinType(node, BuiltinTypeKind::Bool);
  return success();
}

auto ExpressionSema::sema(StringLiteralExpr *node) -> llvm::LogicalResult {
  auto *type = _sema.lookupType("String");
  if (!type)
    return _sema.emitError(node, diag::undefined_type);
  node->setType(type);
  return success();
}

auto ExpressionSema::sema(ObjectIdentityExpr *node) -> llvm::LogicalResult {
  if (llvm::failed(_sema.checkInternalFeature(node->location())))
    return failure();

  if (llvm::failed(sema(node->value().get())))
    return failure();

  auto *valueType = node->value()->type();
  if (!isSourceObjectType(valueType))
    return _sema.emitError(node->value().get(), diag::mismatch_type);

  auto *stringType = _sema.lookupType("String");
  if (!stringType)
    return _sema.emitError(node, diag::undefined_type);

  constexpr std::string_view functionName =
      "mulberry_string_object_identity";
  auto resolvedName = resolveFunctionName(functionName);
  auto *signature = _sema.lookupFunction(resolvedName);
  if (!signature || signature->type->parameterTypes().size() != 2 ||
      !sameType(signature->type->parameterTypes()[0], stringType) ||
      !sameType(signature->type->returnType(), stringType)) {
    auto diagnostic = formatNameDiagnostic(diag::undefined_func, functionName);
    return _sema.emitError(node, diagnostic);
  }

  auto *objectPtr = getPtrType(signature->type->parameterTypes()[1]);
  if (!objectPtr || !isUInt8Type(objectPtr->pointeeType())) {
    auto diagnostic = formatNameDiagnostic(diag::undefined_func, functionName);
    return _sema.emitError(node, diagnostic);
  }

  node->setTypeName(formatStringificationType(valueType));
  node->setType(stringType);
  return success();
}

auto ExpressionSema::sema(CharLiteralExpr *node) -> llvm::LogicalResult {
  _sema.setBuiltinType(node, BuiltinTypeKind::UInt8);
  return success();
}

auto ExpressionSema::sema(TypeLayoutExpr *node) -> llvm::LogicalResult {
  auto *queriedType = _sema.resolveType(node->typeNode());
  if (!queriedType)
    return failure();

  auto value = node->query() == TypeLayoutExpr::Query::SizeOf
                   ? sizeOfType(queriedType)
                   : alignOfType(queriedType);
  if (!value)
    return _sema.emitError(node, diag::unsupported_type_layout);

  node->setQueriedType(queriedType);
  node->setValue(*value);
  _sema.setBuiltinType(node, BuiltinTypeKind::UInt64);
  return success();
}

auto ExpressionSema::sema(HeapAllocExpr *node) -> llvm::LogicalResult {
  if (llvm::failed(_sema.checkInternalFeature(node->location())))
    return failure();

  auto *allocatedType = _sema.checkType(node->typeNode(), SemaImpl::UnitPolicy::Reject);
  if (!allocatedType)
    return failure();

  if (node->count()) {
    if (llvm::failed(sema(node->count().get())))
      return failure();
    if (!isUInt64Type(node->count()->type()))
      return _sema.emitError(node->count().get(), diag::mismatch_type);
  }

  node->setAllocatedType(allocatedType);
  node->setType(_sema._typeContext.createPtrType(allocatedType));
  return success();
}

auto ExpressionSema::sema(AssignExpr *node) -> llvm::LogicalResult {
  if (llvm::failed(sema(node->lhs().get())) ||
      llvm::failed(semaExpected(node->rhs(), node->lhs()->type())))
    return failure();
  if (!node->lhs()->isLvalue())
    return _sema.emitError(node->lhs().get(), diag::expected_lvalue);
  if (llvm::failed(checkAssignable(node->lhs().get())))
    return failure();
  _sema.setBuiltinType(node, BuiltinTypeKind::Unit);
  return success();
}

auto ExpressionSema::sema(BinaryExpr *node) -> llvm::LogicalResult {
  return sema(node, nullptr);
}

auto ExpressionSema::sema(BinaryExpr *node, const Type *expectedType)
    -> llvm::LogicalResult {
  using Operator = BinaryExpr::Operator;
  auto &lhs = node->lhs();
  auto &rhs = node->rhs();
  const Type *operandType = expectedType;

  // A literal-only subtraction has no operand type to guide it. Use the
  // signed domain when both literals fit, so `1 - 2` behaves naturally in
  // the REPL while typed UInt64 expressions keep their existing semantics.
  if (!expectedType && node->opEnum() == Operator::Diff) {
    auto *lhsLiteral = dyn_cast<IntegerLiteralExpr>(lhs.get());
    auto *rhsLiteral = dyn_cast<IntegerLiteralExpr>(rhs.get());
    auto *int64Type =
        _sema._typeContext.getBuiltinType(BuiltinTypeKind::Int64);
    if (lhsLiteral && rhsLiteral && lhsLiteral->getInt64Value() &&
        rhsLiteral->getInt64Value())
      operandType = int64Type;
  }

  if (node->opEnum() == Operator::ShiftLeft ||
      node->opEnum() == Operator::ShiftRight) {
    if (expectedType && isIntegerLikeType(expectedType)) {
      if (llvm::failed(semaExpected(lhs, expectedType)))
        return failure();
    } else if (llvm::failed(sema(lhs.get()))) {
      return failure();
    }

    auto *countType = _sema._typeContext.getBuiltinType(BuiltinTypeKind::UInt64);
    if (llvm::failed(semaExpected(rhs, countType)))
      return failure();
    if (!isIntegerLikeType(lhs->type()) || !isUInt64Type(rhs->type()))
      return _sema.emitError(lhs.get(), diag::mismatch_type);
    if (expectedType && !sameType(lhs->type(), expectedType))
      return _sema.emitError(lhs.get(), diag::mismatch_type);

    node->setType(lhs->type());
    return success();
  }

  if (operandType && isNumericType(operandType)) {
    if (llvm::failed(semaExpected(lhs, operandType)) ||
        llvm::failed(semaExpected(rhs, operandType)))
      return failure();
  } else {
    if (llvm::failed(sema(lhs.get())))
      return failure();

    auto *stringType = _sema.lookupType("String");
    if (node->opEnum() == Operator::Add && stringType &&
        sameType(lhs->type(), stringType)) {
      if (llvm::failed(sema(rhs.get())))
        return failure();
      if (!sameType(rhs->type(), stringType))
        return _sema.emitError(rhs.get(), diag::mismatch_type);
      if (llvm::failed(checkStringConcatFunction(node, stringType)))
        return failure();
      node->setType(stringType);
      return success();
    }

    auto *integerType =
        _sema._typeContext.getBuiltinType(BuiltinTypeKind::Integer);
    if (isIntegerType(lhs->type())) {
      if (llvm::failed(semaExpected(rhs, integerType)))
        return failure();
    } else if (llvm::failed(sema(rhs.get()))) {
      return failure();
    }

    auto *lhsType = lhs->type();
    auto *rhsType = rhs->type();
    if (isIntegerType(lhsType) || isIntegerType(rhsType)) {
      if (llvm::failed(semaExpected(lhs, integerType)) ||
          llvm::failed(semaExpected(rhs, integerType)))
        return failure();
    } else if (isIntegerLikeType(lhsType) &&
               isIntegerLikeType(rhsType) && !sameType(lhsType, rhsType)) {
      auto *lhsLiteral = dyn_cast<IntegerLiteralExpr>(lhs.get());
      auto *rhsLiteral = dyn_cast<IntegerLiteralExpr>(rhs.get());
      if (lhsLiteral && rhsLiteral && lhsLiteral->isNegative() !=
                                         rhsLiteral->isNegative()) {
        if (lhsLiteral->isNegative()) {
          if (llvm::failed(semaExpected(rhs, lhsType)))
            return failure();
        } else if (llvm::failed(semaExpected(lhs, rhsType))) {
          return failure();
        }
      } else if (lhsLiteral) {
        if (llvm::failed(semaExpected(lhs, rhsType)))
          return failure();
      } else if (rhsLiteral) {
        if (llvm::failed(semaExpected(rhs, lhsType)))
          return failure();
      }
    } else if (isFloatingPointType(lhsType) &&
               isFloatingPointType(rhsType) && !sameType(lhsType, rhsType)) {
      if (dyn_cast<FloatLiteralExpr>(lhs.get())) {
        if (llvm::failed(semaExpected(lhs, rhsType)))
          return failure();
      } else if (dyn_cast<FloatLiteralExpr>(rhs.get())) {
        if (llvm::failed(semaExpected(rhs, lhsType)))
          return failure();
      }
    }
  }

  auto *lhsType = lhs->type();
  auto *rhsType = rhs->type();
  if (!sameType(lhsType, rhsType))
    return _sema.emitError(lhs.get(), diag::mismatch_type);

  switch (node->opEnum()) {
  case Operator::ShiftLeft:
  case Operator::ShiftRight:
    llvm_unreachable("shift operations are handled before type comparison");
  case Operator::Add:
  case Operator::Mul:
  case Operator::Diff: {
    if (!isNumericType(lhsType))
      return _sema.emitError(lhs.get(), diag::mismatch_type);
    node->setType(lhsType);
    return success();
  }
  case Operator::Div: {
    // BigInt division is fallible and remains exposed as bigint.div().
    if (isIntegerType(lhsType) || !isNumericType(lhsType))
      return _sema.emitError(lhs.get(), diag::mismatch_type);
    node->setType(lhsType);
    return success();
  }
  case Operator::Rem: {
    if (!isUInt64Type(lhsType) && !isInt64Type(lhsType))
      return _sema.emitError(lhs.get(), diag::mismatch_type);
    node->setType(lhsType);
    return success();
  }
  case Operator::BitAnd:
  case Operator::BitOr:
  case Operator::BitXor: {
    if (!isIntegerLikeType(lhsType))
      return _sema.emitError(lhs.get(), diag::mismatch_type);
    node->setType(lhsType);
    return success();
  }
  case Operator::And:
  case Operator::Or: {
    if (!isBoolType(lhsType))
      return _sema.emitError(lhs.get(), diag::mismatch_type);
    _sema.setBuiltinType(node, BuiltinTypeKind::Bool);
    return success();
  }
  case Operator::EQ:
  case Operator::NEQ: {
    if (!isEquatableType(lhsType))
      return _sema.emitError(lhs.get(), diag::mismatch_type);
    _sema.setBuiltinType(node, BuiltinTypeKind::Bool);
    return success();
  }
  case Operator::LT:
  case Operator::LE:
  case Operator::GT:
  case Operator::GE: {
    if (!isNumericType(lhsType))
      return _sema.emitError(lhs.get(), diag::mismatch_type);
    _sema.setBuiltinType(node, BuiltinTypeKind::Bool);
    return success();
  }
  }

  llvm_unreachable("Unexpected BinaryExpr operator");
}

auto ExpressionSema::checkStringConcatFunction(Expr *node,
                                         const Type *stringType)
    -> llvm::LogicalResult {
  auto functionName = resolveFunctionName("std.string.concat");
  auto *signature = _sema.lookupFunction(functionName);
  if (signature && signature->type->parameterTypes().size() == 2 &&
      sameType(signature->type->parameterTypes()[0], stringType) &&
      sameType(signature->type->parameterTypes()[1], stringType) &&
      sameType(signature->type->returnType(), stringType))
    return success();

  auto diagnostic =
      formatNameDiagnostic(diag::undefined_func, "std.string.concat");
  return _sema.emitError(node, diagnostic);
}

auto ExpressionSema::checkAssignable(const Expr *expr) -> llvm::LogicalResult {
  if (auto *var = llvm::dyn_cast<VariableExpr>(expr)) {
    auto *symbol = _sema.lookupVariable(var->name());
    if (!symbol)
      return _sema.emitError(var->location(), diag::undefined_var);
    if (symbol->isConstBinding)
      return _sema.emitError(var->location(), diag::assign_const);
    return success();
  }

  if (auto *index = llvm::dyn_cast<IndexExpr>(expr))
    return checkConstObjectUseAsMutable(index->base().get());

  if (auto *memberAccess = llvm::dyn_cast<MemberExpr>(expr)) {
    if (!memberAccess->isLvalue())
      return _sema.emitError(memberAccess, diag::expected_lvalue);
    return checkConstObjectUseAsMutable(memberAccess->base().get());
  }

  return success();
}

auto ExpressionSema::canMutateObjectReference(const Expr *expr) -> bool {
  if (auto *matchExpr = llvm::dyn_cast<MatchExpr>(expr))
    return matchExpr->canMutateObject();

  if (auto *tryExpr = llvm::dyn_cast<TryExpr>(expr))
    return tryExpr->canMutateObject();

  if (auto *index = llvm::dyn_cast<IndexExpr>(expr))
    return canMutateObjectReference(index->base().get());

  if (auto *memberAccess = llvm::dyn_cast<MemberExpr>(expr))
    return canMutateObjectReference(memberAccess->base().get());

  auto *var = llvm::dyn_cast<VariableExpr>(expr);
  if (!var)
    return true;

  auto *symbol = _sema.lookupVariable(var->name());
  return !symbol || symbol->canMutateObject;
}

auto ExpressionSema::checkConstObjectUseAsMutable(const Expr *expr)
    -> llvm::LogicalResult {
  if (auto *matchExpr = llvm::dyn_cast<MatchExpr>(expr)) {
    if (!matchExpr->canMutateObject())
      return _sema.emitError(matchExpr, diag::readonly_to_mutable_reference);
    return success();
  }

  if (auto *tryExpr = llvm::dyn_cast<TryExpr>(expr)) {
    if (!tryExpr->canMutateObject())
      return _sema.emitError(expr, diag::readonly_to_mutable_reference);
    return success();
  }

  if (auto *index = llvm::dyn_cast<IndexExpr>(expr))
    return checkConstObjectUseAsMutable(index->base().get());

  if (auto *memberAccess = llvm::dyn_cast<MemberExpr>(expr))
    return checkConstObjectUseAsMutable(memberAccess->base().get());

  auto *var = llvm::dyn_cast<VariableExpr>(expr);
  if (!var)
    return success();

  auto *symbol = _sema.lookupVariable(var->name());
  if (!symbol)
    return _sema.emitError(var->location(), diag::undefined_var);
  if (!symbol->canMutateObject)
    return _sema.emitError(var->location(), diag::readonly_to_mutable_reference);
  return success();
}

auto ExpressionSema::checkMutableObjectArgument(const FunctionType *functionType,
                                          size_t index, const Expr *arg)
    -> llvm::LogicalResult {
  if (!functionType->parameterCanMutateObject()[index])
    return success();
  if (!isMutableSourceObjectType(functionType->parameterTypes()[index]))
    return success();
  return checkConstObjectUseAsMutable(arg);
}

auto ExpressionSema::arrayLiteralTypeWithLeaf(const ArrayLiteralExpr *expr,
                                        const Type *leafType)
    -> const ArrayType * {
  auto &elements = expr->getElements();
  if (elements.empty())
    return nullptr;

  const Type *elementType = leafType;
  if (auto *nested = dyn_cast<ArrayLiteralExpr>(elements.front().get())) {
    elementType = arrayLiteralTypeWithLeaf(nested, leafType);
    if (!elementType)
      return nullptr;
  }
  return _sema._typeContext.createArrayType(elementType, elements.size());
}

auto ExpressionSema::semaDefaultArrayLiteral(ArrayLiteralExpr *expr)
    -> llvm::LogicalResult {
  auto &elements = expr->getElements();
  if (elements.empty())
    return _sema.emitError(expr, diag::expected_expr);

  auto semaElement = [&](Expr *element) -> llvm::LogicalResult {
    if (auto *nestedLiteral = dyn_cast<ArrayLiteralExpr>(element))
      return semaDefaultArrayLiteral(nestedLiteral);
    return sema(element);
  };

  if (llvm::failed(semaElement(elements.front().get())))
    return failure();

  auto *elementType = elements.front()->type();
  if (!isArrayElementType(elementType))
    return _sema.emitError(elements.front().get(), diag::mismatch_type);

  for (size_t index = 1; index < elements.size(); ++index) {
    auto &element = elements[index];
    if (llvm::failed(semaElement(element.get())))
      return failure();
    if (!sameType(elementType, element->type()))
      return _sema.emitError(element.get(), diag::mismatch_type);
  }

  auto *arrayType =
      _sema._typeContext.createArrayType(elementType, elements.size());
  expr->setType(arrayType);
  return success();
}

auto ExpressionSema::semaTensorDisposeCall(CallExpr *node) -> llvm::LogicalResult {
  auto &expressions = node->expressions();
  if (expressions.size() != 1) {
    auto diagnostic =
        formatNameSizeDiagnostic(diag::func_param, node->name(), 1);
    return _sema.emitError(node, diagnostic);
  }

  auto *tensor = expressions.front().get();
  if (llvm::failed(sema(tensor)) || !getTensorElementType(tensor->type()))
    return _sema.emitError(tensor, diag::mismatch_type);
  if (llvm::failed(checkConstObjectUseAsMutable(tensor)))
    return failure();

  _sema.setBuiltinType(node, BuiltinTypeKind::Unit);
  return success();
}

auto ExpressionSema::semaTensorIsDisposedCall(CallExpr *node)
    -> llvm::LogicalResult {
  auto &expressions = node->expressions();
  if (expressions.size() != 1) {
    auto diagnostic =
        formatNameSizeDiagnostic(diag::func_param, node->name(), 1);
    return _sema.emitError(node, diagnostic);
  }

  auto *tensor = expressions.front().get();
  if (llvm::failed(sema(tensor)) || !getTensorElementType(tensor->type()))
    return _sema.emitError(tensor, diag::mismatch_type);

  _sema.setBuiltinType(node, BuiltinTypeKind::Bool);
  return success();
}

auto ExpressionSema::semaTensorStorageAllocCall(CallExpr *node,
                                          const Type *expectedType)
    -> llvm::LogicalResult {
  if (llvm::failed(_sema.checkInternalFeature(node->location())))
    return failure();

  auto &expressions = node->expressions();
  if (expressions.size() != 1) {
    auto diagnostic =
        formatNameSizeDiagnostic(diag::func_param, node->name(), 1);
    return _sema.emitError(node, diagnostic);
  }

  if (!expectedType || !getTensorStorageElementType(expectedType))
    return _sema.emitError(node, diag::mismatch_type);
  if (llvm::failed(sema(expressions.front().get())) ||
      !isUInt64Type(expressions.front()->type()))
    return _sema.emitError(expressions.front().get(), diag::mismatch_type);

  node->setType(expectedType);
  return success();
}

auto ExpressionSema::sema(ArrayLiteralExpr *expr) -> llvm::LogicalResult {
  return semaDefaultArrayLiteral(expr);
}

auto ExpressionSema::sema(ArrayLiteralExpr *expr, const ArrayType *type)
    -> llvm::LogicalResult {
  auto &elements = expr->getElements();
  if (elements.size() != type->size())
    return _sema.emitError(expr, diag::mismatch_type);

  for (auto &element : elements) {
    if (llvm::failed(semaArrayLiteralElement(element, type->elementType())))
      return failure();
  }

  expr->setType(type);
  return success();
}

auto ExpressionSema::semaArrayLiteralElement(std::unique_ptr<Expr> &expr,
                                       const Type *type)
    -> llvm::LogicalResult {
  if (auto *arrayLiteral = llvm::dyn_cast<ArrayLiteralExpr>(expr.get())) {
    auto *arrayType = mulberry::getArrayType(type);
    if (!arrayType)
      return _sema.emitError(expr.get(), diag::mismatch_type);
    return sema(arrayLiteral, arrayType);
  }

  if (mulberry::getArrayType(type))
    return _sema.emitError(expr.get(), diag::mismatch_type);

  return semaExpected(expr, type);
}

auto ExpressionSema::sema(IndexExpr *expr) -> llvm::LogicalResult {
  if (llvm::failed(sema(expr->base().get())))
    return failure();

  if (auto *elementType = getTensorElementType(expr->base()->type())) {
    if (expr->indices().empty())
      return _sema.emitError(expr, diag::mismatch_type);

    for (auto &index : expr->indices()) {
      if (llvm::failed(sema(index.get())))
        return failure();
      if (!isUInt64Type(index->type()))
        return _sema.emitError(index.get(), diag::mismatch_type);
    }

    expr->setType(elementType);
    expr->setLvalue(true);
    expr->setStdlibTensorIndex();
    return success();
  }

  if (auto *elementType = getStdlibListElementType(expr->base()->type())) {
    if (expr->indices().size() != 1)
      return _sema.emitError(expr, diag::mismatch_type);
    auto &index = expr->indices().front();
    if (llvm::failed(sema(index.get())))
      return failure();
    if (!isUInt64Type(index->type()))
      return _sema.emitError(index.get(), diag::mismatch_type);

    std::string getFunctionName;
    std::string setFunctionName;
    if (llvm::failed(instantiateGenericFunction(
            expr, "std.list.List.get", elementType, getFunctionName)) ||
        llvm::failed(instantiateGenericFunction(
            expr, "std.list.List.set", elementType, setFunctionName)))
      return failure();

    expr->setType(elementType);
    expr->setLvalue(true);
    expr->setStdlibListIndex(getFunctionName, setFunctionName);
    return success();
  }

  if (auto *arrayType = mulberry::getArrayType(expr->base()->type())) {
    if (expr->indices().size() != 1)
      return _sema.emitError(expr, diag::mismatch_type);
    auto &index = expr->indices().front();
    if (llvm::failed(sema(index.get())))
      return failure();
    if (!isUInt64Type(index->type()))
      return _sema.emitError(index.get(), diag::mismatch_type);

    expr->setType(arrayType->elementType());
    expr->setLvalue(true);
    expr->setArrayIndex();
    return success();
  }

  auto *ptrType = mulberry::getPtrType(expr->base()->type());
  if (ptrType) {
    if (expr->indices().size() != 1)
      return _sema.emitError(expr, diag::mismatch_type);
    auto &index = expr->indices().front();
    if (llvm::failed(sema(index.get())))
      return failure();
    if (!isUInt64Type(index->type()))
      return _sema.emitError(index.get(), diag::mismatch_type);

    expr->setType(ptrType->pointeeType());
    expr->setLvalue(true);
    expr->setPtrIndex();
    return success();
  }

  return _sema.emitError(expr, diag::mismatch_type);
}

} // namespace mulberry
