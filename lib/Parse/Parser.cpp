//===--- Parser.cpp - Mulberry Language Parser ---------------------------------//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "mulberry/Parse/Parser.h"
#include "mulberry/Basic/Builtins.h"
using namespace mulberry;
using std::make_unique;
using std::unique_ptr;

namespace {
constexpr std::string_view kTensorPack = "tensor.pack";
constexpr std::string_view kStdTensorPack = "std.tensor.pack";
constexpr std::string_view kTensorView = "tensor.view";
constexpr std::string_view kStdTensorView = "std.tensor.view";

auto isTypeLikeName(std::string_view name) -> bool {
  auto tail = name.substr(name.rfind('.') + 1);
  return !tail.empty() &&
         std::isupper(static_cast<unsigned char>(tail.front()));
}

auto isTensorPackName(std::string_view name) -> bool {
  return name == kTensorPack || name == kStdTensorPack;
}

auto isTensorViewName(std::string_view name) -> bool {
  return name == kTensorView || name == kStdTensorView;
}

auto createMemberAccessChain(llvm::SMLoc location, std::string_view name)
    -> std::unique_ptr<Expr> {
  auto dot = name.find('.');
  if (dot == std::string_view::npos)
    return make_unique<VariableExpr>(location, name);

  std::unique_ptr<Expr> expr =
      make_unique<VariableExpr>(location, name.substr(0, dot));
  while (dot != std::string_view::npos) {
    auto nextDot = name.find('.', dot + 1);
    auto fieldName = name.substr(dot + 1, nextDot - dot - 1);
    expr = make_unique<MemberExpr>(location, std::move(expr), fieldName);
    dot = nextDot;
  }
  return expr;
}
} // namespace

auto Parser::parseModule(unique_ptr<Module> &module) -> MulberryResult {
  auto loc = tokenLoc();
  if (tokenIs(Token::kw_package) && parsePackageDecl())
    return failure();

  VectorUniquePtr<Decl> declarations;
  while (!tokenIs(Token::eof)) {
    unique_ptr<Decl> decl;
    if (parseDeclaration(decl))
      return failure();
    declarations.push_back(std::move(decl));
  }

  module = make_unique<Module>(loc, std::move(declarations));
  module->setPackageName(_packageName);
  return success();
}

template <typename T, typename ParseElement>
auto Parser::parseList(Token::Kind separator, Token::Kind end,
                       const char *const separator_error,
                       const char *const end_error,
                       VectorUniquePtr<T> &elements,
                       ParseElement parseElement)
    -> MulberryResult {
  while (!tokenIs(end) && !tokenIs(Token::eof)) {
    unique_ptr<T> exp;
    if (parseElement(exp))
      return failure();
    elements.push_back(std::move(exp));

    if (tokenIs(end))
      break;

    if (parseToken(separator, separator_error))
      return failure();
  }
  return parseToken(end, end_error);
}

// _____________________________________________________________________________
// Parse Identifiers

auto Parser::parsePackageDecl() -> MulberryResult {
  consume(Token::kw_package);
  if (parseQualifiedName(_packageName, diag::expected_id) ||
      parseToken(Token::semi, diag::expected_semi))
    return failure();
  return success();
}

auto Parser::parseUnitType(unique_ptr<TypeNode> &typeNode) -> MulberryResult {
  auto location = tokenLoc();
  consume(Token::l_paren);
  if (parseToken(Token::r_paren, diag::expected_l_paren))
    return failure();
  typeNode = make_unique<UnitTypeNode>(location);
  return success();
}

auto Parser::parseType(unique_ptr<TypeNode> &typeNode) -> MulberryResult {
  if (tokenIs(Token::l_paren))
    return parseUnitType(typeNode);

  auto location = tokenLoc();
  std::string name;
  if (parseQualifiedName(name, diag::expected_type))
    return failure();

  if (name == "Ptr")
    return parsePtrType(typeNode, location);

  if (tokenIs(Token::less)) {
    std::vector<ComptimeArg> arguments;
    if (parseGenericTypeArgs(arguments))
      return failure();
    typeNode = make_unique<GenericTypeNode>(
        location, name, std::move(arguments));
    return success();
  }

  auto elementTypeNode = make_unique<NamedTypeNode>(location, name);
  if (tokenIs(Token::l_square)) {
    std::vector<int64_t> shape;
    if (parseTensorTypeSuffix(shape))
      return failure();
    typeNode = make_unique<TensorTypeNode>(
        std::move(elementTypeNode), std::move(shape), location);
    return success();
  }

  typeNode = std::move(elementTypeNode);
  return success();
}

auto Parser::parseGenericTypeArgs(std::vector<ComptimeArg> &arguments)
    -> MulberryResult {
  consume(Token::less);
  if (tokenIs(Token::greater))
    return emitError(diag::expected_type);

  while (!tokenIs(Token::greater) && !tokenIs(Token::eof)) {
    if (tokenIs(Token::decimal)) {
      auto location = tokenLoc();
      if (auto value = token().getUInt64IntegerValue()) {
        consume(Token::decimal);
        arguments.push_back(ComptimeArg(location, *value));
      } else {
        return emitError(diag::integer_literal_overflows);
      }
    } else {
      unique_ptr<TypeNode> argumentTypeNode;
      if (parseType(argumentTypeNode))
        return failure();
      arguments.push_back(ComptimeArg(std::move(argumentTypeNode)));
    }

    if (tokenIs(Token::greater))
      break;

    if (parseToken(Token::comma, diag::expected_comma_or_r_paren))
      return failure();
  }
  return parseToken(Token::greater, diag::expected_greater);
}

auto Parser::parseComptimeParams(std::vector<ComptimeParam> &parameters)
    -> MulberryResult {
  consume(Token::less);
  if (tokenIs(Token::greater))
    return emitError(diag::expected_id);

  while (!tokenIs(Token::greater) && !tokenIs(Token::eof)) {
    auto parameterName = spelling();
    if (parseToken(Token::identifier, diag::expected_id))
      return failure();
    auto parameterKind = ComptimeParam::Kind::Type;
    if (consumeIf(Token::colon)) {
      if (!tokenIs(Token::identifier) || spelling() != "UInt64")
        return emitError(diag::expected_type);
      consume(Token::identifier);
      parameterKind = ComptimeParam::Kind::UInt64;
    }
    parameters.push_back(ComptimeParam(parameterName.str(), parameterKind));

    if (tokenIs(Token::greater))
      break;

    if (parseToken(Token::comma, diag::expected_comma_or_r_paren))
      return failure();
  }
  return parseToken(Token::greater, diag::expected_greater);
}

auto Parser::parseListType(unique_ptr<TypeNode> &typeNode,
                           llvm::SMLoc location) -> MulberryResult {
  if (parseToken(Token::less, diag::expected_less))
    return failure();

  unique_ptr<TypeNode> elementTypeNode;
  if (parseType(elementTypeNode) ||
      parseToken(Token::greater, diag::expected_greater))
    return failure();

  typeNode = make_unique<ListTypeNode>(std::move(elementTypeNode), location);
  return success();
}

auto Parser::parsePtrType(unique_ptr<TypeNode> &typeNode,
                          llvm::SMLoc location) -> MulberryResult {
  if (parseToken(Token::less, diag::expected_less))
    return failure();

  unique_ptr<TypeNode> pointeeTypeNode;
  if (parseType(pointeeTypeNode) ||
      parseToken(Token::greater, diag::expected_greater))
    return failure();

  typeNode = make_unique<PtrTypeNode>(std::move(pointeeTypeNode), location);
  return success();
}

auto Parser::parseQualifiedName(std::string &name, const char *const message)
    -> MulberryResult {
  name = std::string(spelling());
  if (parseToken(Token::identifier, message))
    return failure();

  while (consumeIf(Token::dot)) {
    name += ".";
    name += spelling();
    if (parseToken(Token::identifier, diag::expected_id))
      return failure();
  }

  return success();
}

auto Parser::parseFunctionName(unique_ptr<FunctionName> &functionName,
                               const char *const message) -> MulberryResult {
  auto location = tokenLoc();
  std::string name;
  if (parseQualifiedName(name, message))
    return failure();
  name = qualifyPackageName(name);
  functionName = make_unique<FunctionName>(location, name);
  return success();
}

auto Parser::parseStructName(unique_ptr<StructName> &structName,
                             const char *const message) -> MulberryResult {
  auto location = tokenLoc();
  std::string name;
  if (parseQualifiedName(name, message))
    return failure();
  name = qualifyPackageName(name);
  structName = make_unique<StructName>(location, name);
  return success();
}

auto Parser::qualifyPackageName(std::string_view name) const -> std::string {
  if (_packageName.empty() || name.find('.') != std::string_view::npos)
    return std::string(name);

  std::string qualifiedName = _packageName;
  qualifiedName += ".";
  qualifiedName += name;
  return qualifiedName;
}

// _____________________________________________________________________________
// Parse Declarations

auto Parser::parseDeclaration(unique_ptr<Decl> &decl) -> MulberryResult {
  switch (tokenKind()) {
  case Token::kw_import:
    return parseImportDecl(decl);
  case Token::kw_extern:
    return parseExternFunctionDecl(decl);
  case Token::kw_fn:
    return parseFunctionDecl(decl);
  case Token::kw_struct:
    return parseStructDecl(decl);
  case Token::kw_comptime:
    return parseComptimeTypeAliasDecl(decl);
  default:
    return emitError(diag::expected_fun_struct);
  }
}

auto Parser::parseImportDecl(unique_ptr<Decl> &decl) -> MulberryResult {
  auto loc = tokenLoc();
  consume(Token::kw_import);

  std::string moduleName;
  if (parseQualifiedName(moduleName, diag::expected_id) ||
      parseToken(Token::semi, diag::expected_semi))
    return failure();

  decl = make_unique<ImportDecl>(loc, moduleName);
  return success();
}

auto Parser::parseFunctionDecl(unique_ptr<Decl> &decl) -> MulberryResult {
  auto loc = tokenLoc();
  unique_ptr<Prototype> proto;
  unique_ptr<FunctionDecl> functionDecl;
  if (parsePrototype(proto) ||
      parseFunctionDeclBody(loc, std::move(proto), functionDecl))
    return failure();

  decl = std::move(functionDecl);
  return success();
}

auto Parser::parseExternFunctionDecl(unique_ptr<Decl> &decl) -> MulberryResult {
  auto loc = tokenLoc();
  consume(Token::kw_extern);

  unique_ptr<Prototype> proto;
  if (parsePrototype(proto, /*qualifyName=*/false) ||
      parseToken(Token::semi, diag::expected_semi))
    return failure();

  decl = make_unique<FunctionDecl>(loc, std::move(proto), nullptr,
                                   /*isExtern=*/true);
  return success();
}

auto Parser::parsePrototype(unique_ptr<Prototype> &proto, bool qualifyName)
    -> MulberryResult {
  auto location = tokenLoc();
  consume(Token::kw_fn);

  // Parse name
  unique_ptr<FunctionName> name;
  std::vector<ComptimeParam> comptimeParameters;
  if (qualifyName) {
    if (parseFunctionName(name, diag::expected_id))
      return failure();
  } else {
    auto nameLocation = tokenLoc();
    std::string methodName;
    if (parseQualifiedName(methodName, diag::expected_id))
      return failure();
    name = make_unique<FunctionName>(nameLocation, methodName);
  }

  if (tokenIs(Token::less) && parseComptimeParams(comptimeParameters))
    return failure();
  if (parseToken(Token::l_paren, diag::expected_l_paren))
    return failure();

  // Parse List
  VectorUniquePtr<VariableStat> parameters;
  unique_ptr<TypeNode> typeNode;
  if (parseList(Token::comma, Token::r_paren, diag::expected_comma_or_r_paren,
                diag::expected_r_paren, parameters,
                [this](unique_ptr<VariableStat> &elem) -> MulberryResult {
                  unique_ptr<VariableExpr> param;
                  unique_ptr<TypeNode> typeNode;
                  if (parseVariableExpr(param) ||
                      parseToken(Token::colon, diag::expected_colon) ||
                      parseType(typeNode))
                    return failure();
                  elem = make_unique<VariableStat>(
                      param->location(), std::move(param), std::move(typeNode),
                      nullptr);
                  return success();
                }) ||
      parseToken(Token::colon, diag::expected_colon) || parseType(typeNode))
    return failure();

  // Make Proto
  proto = make_unique<Prototype>(location, std::move(name),
                                 std::move(parameters), std::move(typeNode),
                                 std::move(comptimeParameters));
  return success();
}

auto Parser::parseFunctionDeclBody(llvm::SMLoc location,
                                   unique_ptr<Prototype> proto,
                                   unique_ptr<FunctionDecl> &functionDecl)
    -> MulberryResult {
  unique_ptr<BlockExpr> body;
  if (parseToken(Token::l_brace, diag::expected_l_brace) ||
      parseBlockExpr(body))
    return failure();

  functionDecl = make_unique<FunctionDecl>(
      location, std::move(proto), std::move(body));
  return success();
}

auto Parser::parseStructMethod(unique_ptr<FunctionDecl> &method)
    -> MulberryResult {
  auto loc = tokenLoc();
  consumeIf(Token::kw_pub);

  unique_ptr<Prototype> proto;
  if (parsePrototype(proto, /*qualifyName=*/false) ||
      parseFunctionDeclBody(loc, std::move(proto), method))
    return failure();

  return success();
}

auto Parser::parseBlockExpr(unique_ptr<BlockExpr> &block) -> MulberryResult {
  auto loc = tokenLoc();
  VectorUniquePtr<Stat> statements;
  while (true) {
    if (consumeIf(Token::r_brace)) {
      block = make_unique<BlockExpr>(
          loc, std::move(statements), make_unique<UnitExpr>(loc));
      return success();
    }

    unique_ptr<Stat> stat;
    if (parseStatementWithoutSemi(stat))
      return failure();

    auto isStatement = stat->getKind() != Stat::Stat_Expression;
    if (isStatement) {
      if (parseToken(Token::semi, diag::expected_semi))
        return failure();
      statements.push_back(std::move(stat));
      continue;
    }

    if (consumeIf(Token::semi)) {
      statements.push_back(std::move(stat));
      continue;
    }

    if (!consumeIf(Token::r_brace)) {
      statements.push_back(std::move(stat));
      continue;
    }

    unique_ptr<ExprStat> exprStat(static_cast<ExprStat *>(stat.release()));
    block = make_unique<BlockExpr>(loc, std::move(statements),
                                   std::move(exprStat->expression()));
    return success();
  }
}

auto Parser::parseStructDecl(unique_ptr<Decl> &decl) -> MulberryResult {
  auto loc = tokenLoc();
  consume(Token::kw_struct);

  // Parse name
  unique_ptr<StructName> name;
  if (parseStructName(name, diag::expected_id) ||
      parseToken(Token::l_brace, diag::expected_l_brace))
    return failure();

  VectorUniquePtr<VariableStat> fields;
  VectorUniquePtr<FunctionDecl> methods;
  if (parseStructMembers(fields, methods))
    return failure();

  decl = make_unique<StructDecl>(
      loc, std::move(name), std::move(fields), std::move(methods));
  return success();
}

auto Parser::parseStructMembers(VectorUniquePtr<VariableStat> &fields,
                                VectorUniquePtr<FunctionDecl> &methods)
    -> MulberryResult {
  while (!tokenIs(Token::r_brace) && !tokenIs(Token::eof)) {
    if (tokenIs(Token::kw_pub) || tokenIs(Token::kw_fn)) {
      unique_ptr<FunctionDecl> method;
      if (parseStructMethod(method))
        return failure();
      methods.push_back(std::move(method));
      consumeIf(Token::comma);
      continue;
    }

    unique_ptr<VariableExpr> var;
    unique_ptr<TypeNode> typeNode;
    if (parseVariableExpr(var) ||
        parseToken(Token::colon, diag::expected_colon) ||
        parseType(typeNode))
      return failure();
    fields.push_back(make_unique<VariableStat>(
        var->location(), std::move(var), std::move(typeNode), nullptr));

    if (!tokenIs(Token::r_brace) &&
        parseToken(Token::comma, diag::expected_comma_or_r_brace))
      return failure();
  }
  return parseToken(Token::r_brace, diag::expected_r_brace);
}

auto Parser::parseComptimeAliasBody(unique_ptr<TypeNode> &typeNode)
    -> MulberryResult {
  if (!tokenIs(Token::kw_struct))
    return parseType(typeNode);

  auto location = tokenLoc();
  consume(Token::kw_struct);
  if (parseToken(Token::l_brace, diag::expected_l_brace))
    return failure();

  VectorUniquePtr<VariableStat> fields;
  VectorUniquePtr<FunctionDecl> methods;
  if (parseStructMembers(fields, methods))
    return failure();

  typeNode = make_unique<StructTypeNode>(
      location, std::move(fields), std::move(methods));
  return success();
}

auto Parser::parseComptimeTypeAliasDecl(unique_ptr<Decl> &decl)
    -> MulberryResult {
  auto location = tokenLoc();
  consume(Token::kw_comptime);

  std::string name = std::string(spelling());
  if (parseToken(Token::identifier, diag::expected_id))
    return failure();
  name = qualifyPackageName(name);

  std::vector<ComptimeParam> parameters;
  if (tokenIs(Token::less)) {
    if (parseComptimeParams(parameters) ||
        parseToken(Token::assign, diag::expected_assign))
      return failure();
  } else if (parseToken(Token::assign, diag::expected_assign)) {
    return failure();
  }

  unique_ptr<TypeNode> bodyTypeNode;
  if (parseComptimeAliasBody(bodyTypeNode) ||
      parseToken(Token::semi, diag::expected_semi))
    return failure();

  decl = make_unique<ComptimeTypeAliasDecl>(
      location, name, std::move(parameters), std::move(bodyTypeNode));
  return success();
}

auto Parser::parseTensorTypeSuffix(std::vector<int64_t> &shape)
    -> MulberryResult {
  consume(Token::l_square);
  while (!tokenIs(Token::r_square) && !tokenIs(Token::eof)) {
    if (auto number = token().getUInt64IntegerValue()) {
      shape.push_back(*number);
      consume(Token::decimal);
    } else if (tokenIs(Token::question)) {
      shape.push_back(-1);
      consume(Token::question);
    } else {
      return emitError(diag::expected_expr);
    }

    if (tokenIs(Token::r_square))
      break;
    if (parseToken(Token::comma, diag::expected_comma_or_r_square))
      return failure();
  }

  if (shape.empty())
    return emitError(diag::expected_expr);
  return parseToken(Token::r_square, diag::expected_r_square);
}

auto Parser::parseArrayLiteral(unique_ptr<Expr> &expr) -> MulberryResult {
  auto loc = tokenLoc();
  consume(Token::l_square); // '['
  std::vector<std::unique_ptr<Expr>> elements;

  if (!tokenIs(Token::r_square)) {
    do {
      unique_ptr<Expr> subExpr;

      if (tokenIs(Token::l_square)) {
        if (parseArrayLiteral(subExpr))
          return failure();
      } else {
        if (parseExpression(subExpr))
          return failure();
      }
      elements.push_back(std::move(subExpr));
    } while (consumeIf(Token::comma));
  }
  if (parseToken(Token::r_square, diag::expected_r_square))
    return failure();
  expr = make_unique<ArrayLiteralExpr>(loc, std::move(elements));

  return success();
}

auto Parser::parseIndex(unique_ptr<Expr> &expr) -> MulberryResult {
  auto location = expr->location();
  consume(Token::l_square); // '['
  std::vector<std::unique_ptr<Expr>> indices;
  do {
    unique_ptr<Expr> subExpr;
    if (parseExpression(subExpr))
      return failure();
    indices.push_back(std::move(subExpr));
  } while (consumeIf(Token::comma) && !tokenIs(Token::r_square));
  if (parseToken(Token::r_square, diag::expected_r_square))
    return failure();

  expr = std::make_unique<IndexExpr>(location, std::move(expr),
                                     std::move(indices));
  return success();
}

// _____________________________________________________________________________
// Parse Expressions

auto Parser::parseExpression(unique_ptr<Expr> &expr) -> MulberryResult {
  if (parsePrimaryExpression(expr)) {
    return failure();
  } else {
    return parseBinaryExpRHS(0, expr);
  }
}

auto Parser::parseBlockCondition(unique_ptr<Expr> &expr) -> MulberryResult {
  auto oldStopBeforeStructLiteral = _stopBeforeStructLiteral;
  // In `if/while flag { ... }`, keep `{` for the block instead of parsing
  // `flag { ... }` as a struct literal.
  _stopBeforeStructLiteral = true;
  auto result = parseExpression(expr);
  _stopBeforeStructLiteral = oldStopBeforeStructLiteral;
  return result;
}

auto Parser::parseExpressions(VectorUniquePtr<Expr> &expressions,
                              Token::Kind separator, Token::Kind end,
                              const char *const separator_error,
                              const char *const end_error) -> MulberryResult {
  return parseList(separator, end, separator_error, end_error, expressions,
                   [this](unique_ptr<Expr> &elem) -> MulberryResult {
                     return parseExpression(elem);
                   });
}

auto Parser::parsePrimaryExpression(unique_ptr<Expr> &expr) -> MulberryResult {
  switch (tokenKind()) {
  case Token::decimal:
    return parseDecimal(expr);
  case Token::float_literal:
    return parseFloat(expr);
  case Token::string_literal:
    return parseString(expr);
  case Token::char_literal:
    return parseChar(expr);
  case Token::diff:
    return parseNegativeFloat(expr);
  case Token::mul:
    return parseDerefExpr(expr);
  case Token::identifier:
    return parseIdentifierExpr(expr);
  case Token::l_square:
    return parseArrayLiteral(expr);
  case Token::l_brace:
    return parseZeroInitExpr(tokenLoc(), expr);
  case Token::kw_if:
    return parseIfExpr(expr);
  case Token::kw_while:
    return parseWhileExpr(expr);
  case Token::kw_break:
    return parseBreakExpr(expr);
  case Token::kw_continue:
    return parseContinueExpr(expr);
  case Token::kw_for:
    return parseForExpr(expr);
  case Token::kw_true: {
    auto loc = tokenLoc();
    consume(Token::kw_true);
    expr = make_unique<BoolLiteralExpr>(loc, true);
    return success();
  }
  case Token::kw_false: {
    auto loc = tokenLoc();
    consume(Token::kw_false);
    expr = make_unique<BoolLiteralExpr>(loc, false);
    return success();
  }
  case Token::l_paren: {
    auto location = tokenLoc();
    consume(Token::l_paren);
    if (parseToken(Token::r_paren, diag::expected_l_paren))
      return failure();
    expr = make_unique<UnitExpr>(location);
    return success();
  }
  default:
    return emitError(diag::expected_expr);
  }
}

auto Parser::parseVariableExpr(unique_ptr<VariableExpr> &identifier)
    -> MulberryResult {
  auto location = tokenLoc();
  auto name = spelling();
  if (parseToken(Token::identifier, diag::expected_id))
    return failure();
  identifier = make_unique<VariableExpr>(location, name);
  return success();
}

auto Parser::parseIfExpr(std::unique_ptr<Expr> &expr) -> MulberryResult {
  auto loc = tokenLoc();
  consume(Token::kw_if);
  unique_ptr<Expr> condition;
  unique_ptr<BlockExpr> thenBlock;
  unique_ptr<BlockExpr> elseBlock;
  if (parseBlockCondition(condition) ||
      parseToken(Token::l_brace, diag::expected_l_brace) ||
      parseBlockExpr(thenBlock))
    return failure();

  if (consumeIf(Token::kw_else)) {
    if (parseToken(Token::l_brace, diag::expected_l_brace) ||
        parseBlockExpr(elseBlock))
      return failure();
  }

  expr = make_unique<IfExpr>(loc, std::move(condition), std::move(thenBlock),
                             std::move(elseBlock));
  return success();
}

auto Parser::parseWhileExpr(std::unique_ptr<Expr> &expr) -> MulberryResult {
  auto loc = tokenLoc();
  consume(Token::kw_while);
  unique_ptr<Expr> condition;
  unique_ptr<BlockExpr> bodyBlock;
  if (parseBlockCondition(condition) ||
      parseToken(Token::l_brace, diag::expected_l_brace) ||
      parseBlockExpr(bodyBlock))
    return failure();
  expr =
      make_unique<WhileExpr>(loc, std::move(condition), std::move(bodyBlock));
  return success();
}

auto Parser::parseBreakExpr(std::unique_ptr<Expr> &expr) -> MulberryResult {
  auto loc = tokenLoc();
  consume(Token::kw_break);
  expr = make_unique<BreakExpr>(loc);
  return success();
}

auto Parser::parseContinueExpr(std::unique_ptr<Expr> &expr) -> MulberryResult {
  auto loc = tokenLoc();
  consume(Token::kw_continue);
  expr = make_unique<ContinueExpr>(loc);
  return success();
}

auto Parser::parseForExpr(std::unique_ptr<Expr> &expr) -> MulberryResult {
  auto loc = tokenLoc();
  consume(Token::kw_for);

  auto variableName = spelling();
  if (parseToken(Token::identifier, diag::expected_id) ||
      parseToken(Token::kw_in, diag::expected_in))
    return failure();

  unique_ptr<Expr> startExpr;
  unique_ptr<Expr> endExpr;
  unique_ptr<BlockExpr> bodyBlock;
  if (parseExpression(startExpr) ||
      parseToken(Token::dot_dot, diag::expected_dot_dot) ||
      parseBlockCondition(endExpr) ||
      parseToken(Token::l_brace, diag::expected_l_brace) ||
      parseBlockExpr(bodyBlock))
    return failure();

  expr = make_unique<ForExpr>(loc, variableName, std::move(startExpr),
                              std::move(endExpr), std::move(bodyBlock));
  return success();
}

auto Parser::parseDecimal(unique_ptr<Expr> &expr) -> MulberryResult {
  auto loc = tokenLoc();
  if (auto value = token().getUInt64IntegerValue()) {
    consume(Token::decimal);
    expr = make_unique<DecimalLiteralExpr>(loc, std::move(*value));
    return success();
  }
  return emitError(diag::integer_literal_overflows);
}

auto Parser::parseFloat(unique_ptr<Expr> &expr) -> MulberryResult {
  auto loc = tokenLoc();
  if (auto value = token().getFloat32Value()) {
    consume(Token::float_literal);
    expr = make_unique<FloatLiteralExpr>(loc, std::move(*value));
    return success();
  }
  return emitError(diag::float_literal_invalid);
}

auto Parser::parseNegativeFloat(unique_ptr<Expr> &expr) -> MulberryResult {
  auto loc = tokenLoc();
  consume(Token::diff);
  if (!tokenIs(Token::float_literal))
    return emitError(diag::expected_expr);

  if (auto value = token().getFloat32Value()) {
    consume(Token::float_literal);
    value->changeSign();
    expr = make_unique<FloatLiteralExpr>(loc, std::move(*value));
    return success();
  }
  return emitError(diag::float_literal_invalid);
}

auto Parser::parseString(unique_ptr<Expr> &expr) -> MulberryResult {
  auto loc = tokenLoc();
  if (auto value = token().getStringLiteralValue()) {
    consume(Token::string_literal);
    expr = make_unique<StringLiteralExpr>(loc, std::move(*value));
    return success();
  }
  return emitError(diag::string_literal_invalid);
}

auto Parser::parseChar(unique_ptr<Expr> &expr) -> MulberryResult {
  auto loc = tokenLoc();
  if (auto value = token().getCharLiteralValue()) {
    consume(Token::char_literal);
    expr = make_unique<CharLiteralExpr>(loc, *value);
    return success();
  }
  return emitError(diag::expected_expr);
}

auto Parser::parseDerefExpr(unique_ptr<Expr> &expr) -> MulberryResult {
  auto location = tokenLoc();
  consume(Token::mul);

  unique_ptr<Expr> pointer;
  if (parsePrimaryExpression(pointer))
    return failure();
  while (tokenIs(Token::dot) || tokenIs(Token::l_square)) {
    if (tokenIs(Token::dot)) {
      if (parseMemberAccess(pointer))
        return failure();
      continue;
    }
    if (parseIndex(pointer))
      return failure();
  }

  expr = make_unique<DerefExpr>(location, std::move(pointer));
  return success();
}

auto Parser::parseIdentifierExpr(unique_ptr<Expr> &expr) -> MulberryResult {
  auto location = tokenLoc();
  std::string name;
  if (parseQualifiedName(name, diag::expected_id))
    return failure();
  switch (tokenKind()) {
  case Token::less:
    if (name != "heap.alloc" && isTypeLikeName(name))
      return parseGenericStructLiteral(location, name, expr);
    if (name == "heap.alloc")
      return parseHeapAllocExpr(location, name, expr);
    expr = createMemberAccessChain(location, name);
    return success();
  case Token::l_paren:
    if (name == builtins::sizeOf || name == builtins::alignOf)
      return parseTypeLayoutExpr(location, name, expr);
    if (isTensorPackName(name))
      return parseTensorPackExpr(location, name, expr);
    if (isTensorViewName(name))
      return parseTensorViewExpr(location, name, expr);
    return parseFunctionCall(location, name, expr);
  case Token::l_brace:
    if (_stopBeforeStructLiteral) {
      expr = createMemberAccessChain(location, name);
      return success();
    }
    return parseStructLiteral(
        location, make_unique<NamedTypeNode>(location, name), expr);
  default:
    expr = createMemberAccessChain(location, name);
    return success();
  }
}

auto Parser::parseTypeLayoutExpr(llvm::SMLoc location, std::string_view name,
                                 unique_ptr<Expr> &expr) -> MulberryResult {
  consume(Token::l_paren);

  unique_ptr<TypeNode> typeNode;
  if (parseType(typeNode) || parseToken(Token::r_paren, diag::expected_r_paren))
    return failure();

  auto query = name == builtins::sizeOf ? TypeLayoutExpr::Query::SizeOf
                                        : TypeLayoutExpr::Query::AlignOf;
  expr = make_unique<TypeLayoutExpr>(location, query, std::move(typeNode));
  return success();
}

auto Parser::parseHeapAllocExpr(llvm::SMLoc location, std::string_view name,
                                unique_ptr<Expr> &expr) -> MulberryResult {
  if (name != "heap.alloc")
    return emitError(diag::expected_expr);

  consume(Token::less);
  unique_ptr<TypeNode> typeNode;
  if (parseType(typeNode) ||
      parseToken(Token::greater, diag::expected_greater) ||
      parseToken(Token::l_paren, diag::expected_l_paren))
    return failure();

  unique_ptr<Expr> count;
  if (!tokenIs(Token::r_paren) && parseExpression(count))
    return failure();
  if (parseToken(Token::r_paren, diag::expected_r_paren))
    return failure();

  expr = make_unique<HeapAllocExpr>(location, std::move(typeNode),
                                    std::move(count));
  return success();
}

auto Parser::parseZeroInitExpr(llvm::SMLoc location,
                               unique_ptr<Expr> &expr) -> MulberryResult {
  consume(Token::l_brace);
  if (parseToken(Token::r_brace, diag::expected_r_brace))
    return failure();

  expr = make_unique<ZeroInitExpr>(location);
  return success();
}

auto Parser::parseTensorPackExpr(llvm::SMLoc location, std::string_view name,
                                 unique_ptr<Expr> &expr) -> MulberryResult {
  if (!isTensorPackName(name))
    return emitError(diag::expected_expr);

  consume(Token::l_paren);
  unique_ptr<Expr> tensor;
  if (parseExpression(tensor) ||
      parseToken(Token::r_paren, diag::expected_r_paren))
    return failure();

  expr = make_unique<TensorPackExpr>(location, std::move(tensor));
  return success();
}

auto Parser::parseTensorViewExpr(llvm::SMLoc location, std::string_view name,
                                 unique_ptr<Expr> &expr) -> MulberryResult {
  if (!isTensorViewName(name))
    return emitError(diag::expected_expr);

  consume(Token::l_paren);
  unique_ptr<Expr> tensorRecord;
  if (parseExpression(tensorRecord) ||
      parseToken(Token::r_paren, diag::expected_r_paren))
    return failure();

  expr = make_unique<TensorViewExpr>(location, std::move(tensorRecord));
  return success();
}

auto Parser::parseFunctionCall(llvm::SMLoc location, std::string_view name,
                               unique_ptr<Expr> &expr) -> MulberryResult {
  consume(Token::l_paren);
  auto expressions = VectorUniquePtr<Expr>();
  if (parseExpressions(expressions, Token::comma, Token::r_paren,
                       diag::expected_comma_or_r_paren, diag::expected_r_paren))
    return failure();
  expr = make_unique<CallExpr>(location, name, std::move(expressions));
  return success();
}

auto Parser::parseStructLiteral(llvm::SMLoc location,
                                unique_ptr<TypeNode> typeNode,
                                unique_ptr<Expr> &expr) -> MulberryResult {
  consume(Token::l_brace);
  auto expressions = VectorUniquePtr<Expr>();
  if (parseExpressions(expressions, Token::comma, Token::r_brace,
                       diag::expected_comma_or_r_brace, diag::expected_r_brace))
    return failure();
  expr = make_unique<StructLiteralExpr>(
      location, std::move(typeNode), std::move(expressions));
  return success();
}

auto Parser::parseGenericStructLiteral(llvm::SMLoc location,
                                       std::string_view name,
                                       unique_ptr<Expr> &expr) -> MulberryResult {
  std::vector<ComptimeArg> arguments;
  if (parseGenericTypeArgs(arguments))
    return failure();

  auto typeNode = make_unique<GenericTypeNode>(
      location, name, std::move(arguments));
  if (!tokenIs(Token::l_brace))
    return emitError(diag::expected_l_brace);
  return parseStructLiteral(location, std::move(typeNode), expr);
}

auto Parser::parseBinaryExpRHS(int exprPrec, std::unique_ptr<Expr> &expr)
    -> MulberryResult {
  while (true) {
    int tokPrec = getTokenPrecedence();
    if (tokPrec < exprPrec)
      return success();

    Token t = token();

    if (t.is(Token::dot)) {
      if (parseMemberAccess(expr))
        return failure();
      continue;
    }
    if (t.is(Token::l_square)) {
      if (parseIndex(expr))
        return failure();
      continue;
    }

    consume(t.getKind());
    auto location = tokenLoc();

    unique_ptr<Expr> rhs;
    if (parsePrimaryExpression(rhs))
      return emitError(diag::expected_expr);

    int nextPrec = getTokenPrecedence();
    if (tokPrec < nextPrec) {
      if (parseBinaryExpRHS(tokPrec + 1, rhs))
        return failure();
    } else if ((tokPrec == nextPrec) && isTokenRightAssociative()) {
      if (parseBinaryExpRHS(tokPrec, rhs))
        return failure();
    }
    if (t.is(Token::assign)) {
      expr = std::make_unique<AssignExpr>(location, std::move(expr),
                                          std::move(rhs));
      continue;
    }
    expr = std::make_unique<BinaryExpr>(location, tokenToOperator(t),
                                        std::move(expr), std::move(rhs));
  }
}

auto Parser::parseMemberAccess(std::unique_ptr<Expr> &expr)
    -> MulberryResult {
  auto location = tokenLoc();
  consume(Token::dot);

  auto fieldName = spelling();
  if (parseToken(Token::identifier, diag::expected_id))
    return failure();

  if (tokenIs(Token::l_paren)) {
    consume(Token::l_paren);
    auto expressions = VectorUniquePtr<Expr>();
    if (parseExpressions(expressions, Token::comma, Token::r_paren,
                         diag::expected_comma_or_r_paren,
                         diag::expected_r_paren))
      return failure();
    expr = std::make_unique<CallExpr>(
        location, std::move(expr), fieldName, std::move(expressions));
    return success();
  }

  expr = std::make_unique<MemberExpr>(location, std::move(expr), fieldName);
  return success();
}

auto Parser::getTokenPrecedence() -> int {
  switch (tokenKind()) {
  case Token::assign:
    return 100;
  case Token::kw_or:
    return 200;
  case Token::kw_and:
    return 300;
  case Token::kw_eq:
  case Token::kw_neq:
  case Token::eq:
  case Token::neq:
    return 400;
  case Token::kw_lt:
  case Token::kw_le:
  case Token::kw_gt:
  case Token::kw_ge:
  case Token::less:
  case Token::less_equal:
  case Token::greater:
  case Token::greater_equal:
    return 500;
  case Token::add:
  case Token::diff:
    return 600;
  case Token::mul:
  case Token::div:
  case Token::rem:
    return 700;
  case Token::dot:
  case Token::l_square:
    return 800;
  default:
    return -1;
  }
}

auto Parser::isTokenRightAssociative() -> bool {
  switch (tokenKind()) {
  case Token::assign:
    return true;
  default:
    return false;
  }
}

auto Parser::tokenToOperator(Token token) -> BinaryExpr::Operator {
  switch (token.getKind()) {
  case Token::add:
    return BinaryExpr::Operator::Add;
  case Token::diff:
    return BinaryExpr::Operator::Diff;
  case Token::mul:
    return BinaryExpr::Operator::Mul;
  case Token::div:
    return BinaryExpr::Operator::Div;
  case Token::rem:
    return BinaryExpr::Operator::Rem;
  case Token::kw_and:
    return BinaryExpr::Operator::And;
  case Token::kw_or:
    return BinaryExpr::Operator::Or;
  case Token::kw_eq:
  case Token::eq:
    return BinaryExpr::Operator::EQ;
  case Token::kw_neq:
  case Token::neq:
    return BinaryExpr::Operator::NEQ;
  case Token::kw_lt:
  case Token::less:
    return BinaryExpr::Operator::LT;
  case Token::kw_le:
  case Token::less_equal:
    return BinaryExpr::Operator::LE;
  case Token::kw_gt:
  case Token::greater:
    return BinaryExpr::Operator::GT;
  case Token::kw_ge:
  case Token::greater_equal:
    return BinaryExpr::Operator::GE;
  default:
    llvm_unreachable("Unexpected operator");
  }
}

// _____________________________________________________________________________
// Parse Statements

auto Parser::parseStatementWithoutSemi(unique_ptr<Stat> &stat) -> MulberryResult {
  switch (tokenKind()) {
  case Token::kw_var:
    return parseVarDecl(stat);
  case Token::kw_const:
    return parseConstDecl(stat);
  default: {
    auto loc = tokenLoc();
    unique_ptr<Expr> expr;
    if (parseExpression(expr))
      return failure();
    stat = make_unique<ExprStat>(loc, std::move(expr));
    return success();
  }
  }
}

auto Parser::parseVarDecl(unique_ptr<Stat> &stat) -> MulberryResult {
  return parseVariableDecl(stat, /*isConst=*/false);
}

auto Parser::parseConstDecl(unique_ptr<Stat> &stat) -> MulberryResult {
  return parseVariableDecl(stat, /*isConst=*/true);
}

auto Parser::parseVariableDecl(unique_ptr<Stat> &stat, bool isConst)
    -> MulberryResult {
  auto loc = tokenLoc();
  consume(isConst ? Token::kw_const : Token::kw_var);
  unique_ptr<VariableExpr> var;
  unique_ptr<TypeNode> typeNode;
  unique_ptr<Expr> e;
  if (parseVariableExpr(var) ||
      parseToken(Token::colon, diag::expected_colon) || parseType(typeNode) ||
      parseToken(Token::assign, diag::expected_assign) || parseExpression(e))
    return failure();
  stat = make_unique<VariableStat>(loc, std::move(var), std::move(typeNode),
                                   std::move(e), isConst);
  return success();
}
