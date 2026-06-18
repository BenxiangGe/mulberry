//===--- Parser.cpp - Cherry Language Parser ---------------------------------//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/Parse/Parser.h"
using namespace cherry;
using std::make_unique;
using std::unique_ptr;

namespace {
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

auto Parser::parseModule(unique_ptr<Module> &module) -> CherryResult {
  auto loc = tokenLoc();
  VectorUniquePtr<Decl> declarations;
  do {
    unique_ptr<Decl> decl;
    if (parseDeclaration(decl))
      return failure();
    declarations.push_back(std::move(decl));
  } while (!tokenIs(Token::eof));

  module = make_unique<Module>(loc, std::move(declarations));
  return success();
}

template <typename T, typename ParseElement>
auto Parser::parseList(Token::Kind separator, Token::Kind end,
                       const char *const separator_error,
                       const char *const end_error,
                       VectorUniquePtr<T> &elements,
                       ParseElement parseElement)
    -> CherryResult {
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

auto Parser::parseUnitType(unique_ptr<TypeNode> &typeNode) -> CherryResult {
  auto location = tokenLoc();
  consume(Token::l_paren);
  if (parseToken(Token::r_paren, diag::expected_l_paren))
    return failure();
  typeNode = make_unique<UnitTypeNode>(location);
  return success();
}

auto Parser::parseType(unique_ptr<TypeNode> &typeNode) -> CherryResult {
  if (tokenIs(Token::l_paren))
    return parseUnitType(typeNode);

  auto location = tokenLoc();
  std::string name;
  if (parseQualifiedName(name, diag::expected_type))
    return failure();

  if (name == "List")
    return parseListType(typeNode, location);

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

auto Parser::parseListType(unique_ptr<TypeNode> &typeNode,
                           llvm::SMLoc location) -> CherryResult {
  if (parseToken(Token::less, diag::expected_less))
    return failure();

  unique_ptr<TypeNode> elementTypeNode;
  if (parseType(elementTypeNode) ||
      parseToken(Token::greater, diag::expected_greater))
    return failure();

  typeNode = make_unique<ListTypeNode>(std::move(elementTypeNode), location);
  return success();
}

auto Parser::parseQualifiedName(std::string &name, const char *const message)
    -> CherryResult {
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
                               const char *const message) -> CherryResult {
  auto location = tokenLoc();
  std::string name;
  if (parseQualifiedName(name, message))
    return failure();
  functionName = make_unique<FunctionName>(location, name);
  return success();
}

auto Parser::parseStructName(unique_ptr<StructName> &structName,
                             const char *const message) -> CherryResult {
  auto location = tokenLoc();
  std::string name;
  if (parseQualifiedName(name, message))
    return failure();
  structName = make_unique<StructName>(location, name);
  return success();
}

// _____________________________________________________________________________
// Parse Declarations

auto Parser::parseDeclaration(unique_ptr<Decl> &decl) -> CherryResult {
  switch (tokenKind()) {
  case Token::kw_fn:
    return parseFunctionDecl(decl);
  case Token::kw_struct:
    return parseStructDecl(decl);
  default:
    return emitError(diag::expected_fun_struct);
  }
}

auto Parser::parseFunctionDecl(unique_ptr<Decl> &decl) -> CherryResult {
  auto loc = tokenLoc();
  unique_ptr<Prototype> proto;
  unique_ptr<BlockExpr> body;
  if (parsePrototype(proto) ||
      parseToken(Token::l_brace, diag::expected_l_brace) ||
      parseBlockExpr(body))
    return failure();

  decl = make_unique<FunctionDecl>(loc, std::move(proto), std::move(body));
  return success();
}

auto Parser::parsePrototype(unique_ptr<Prototype> &proto) -> CherryResult {
  auto location = tokenLoc();
  consume(Token::kw_fn);

  // Parse name
  unique_ptr<FunctionName> name;
  if (parseFunctionName(name, diag::expected_id) ||
      parseToken(Token::l_paren, diag::expected_l_paren))
    return failure();

  // Parse List
  VectorUniquePtr<VariableStat> parameters;
  unique_ptr<TypeNode> typeNode;
  if (parseList(Token::comma, Token::r_paren, diag::expected_comma_or_r_paren,
                diag::expected_r_paren, parameters,
                [this](unique_ptr<VariableStat> &elem) -> CherryResult {
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
                                 std::move(parameters), std::move(typeNode));
  return success();
}

auto Parser::parseBlockExpr(unique_ptr<BlockExpr> &block) -> CherryResult {
  auto loc = tokenLoc();
  VectorUniquePtr<Stat> statements;
  while (true) {
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

    if (parseToken(Token::r_brace, diag::expected_r_brace))
      return failure();

    unique_ptr<ExprStat> exprStat(static_cast<ExprStat *>(stat.release()));
    block = make_unique<BlockExpr>(loc, std::move(statements),
                                   std::move(exprStat->expression()));
    return success();
  }
}

auto Parser::parseStructDecl(unique_ptr<Decl> &decl) -> CherryResult {
  auto loc = tokenLoc();
  consume(Token::kw_struct);

  // Parse name
  unique_ptr<StructName> name;
  if (parseStructName(name, diag::expected_id) ||
      parseToken(Token::l_brace, diag::expected_l_brace))
    return failure();

  // Parse List
  VectorUniquePtr<VariableStat> fields;
  if (parseList(Token::comma, Token::r_brace, diag::expected_comma_or_r_brace,
                diag::expected_r_brace, fields,
                [this](unique_ptr<VariableStat> &elem) -> CherryResult {
                  unique_ptr<VariableExpr> var;
                  unique_ptr<TypeNode> typeNode;
                  if (parseVariableExpr(var) ||
                      parseToken(Token::colon, diag::expected_colon) ||
                      parseType(typeNode))
                    return failure();
                  elem = make_unique<VariableStat>(
                      var->location(), std::move(var), std::move(typeNode),
                      nullptr);
                  return success();
                }))
    return failure();

  // Make StructDecl
  decl = make_unique<StructDecl>(loc, std::move(name), std::move(fields));
  return success();
}

auto Parser::parseTensorTypeSuffix(std::vector<int64_t> &shape)
    -> CherryResult {
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

auto Parser::parseArrayLiteral(unique_ptr<Expr> &expr) -> CherryResult {
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

auto Parser::parseIndex(unique_ptr<Expr> &expr) -> CherryResult {
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

auto Parser::parseExpression(unique_ptr<Expr> &expr) -> CherryResult {
  if (parsePrimaryExpression(expr)) {
    return failure();
  } else {
    return parseBinaryExpRHS(0, expr);
  }
}

auto Parser::parseBlockCondition(unique_ptr<Expr> &expr) -> CherryResult {
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
                              const char *const end_error) -> CherryResult {
  return parseList(separator, end, separator_error, end_error, expressions,
                   [this](unique_ptr<Expr> &elem) -> CherryResult {
                     return parseExpression(elem);
                   });
}

auto Parser::parsePrimaryExpression(unique_ptr<Expr> &expr) -> CherryResult {
  switch (tokenKind()) {
  case Token::decimal:
    return parseDecimal(expr);
  case Token::float_literal:
    return parseFloat(expr);
  case Token::string_literal:
    return parseString(expr);
  case Token::diff:
    return parseNegativeFloat(expr);
  case Token::identifier:
    return parseIdentifierExpr(expr);
  case Token::l_square:
    return parseArrayLiteral(expr);
  case Token::kw_if:
    return parseIfExpr(expr);
  case Token::kw_while:
    return parseWhileExpr(expr);
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
    -> CherryResult {
  auto location = tokenLoc();
  auto name = spelling();
  if (parseToken(Token::identifier, diag::expected_id))
    return failure();
  identifier = make_unique<VariableExpr>(location, name);
  return success();
}

auto Parser::parseIfExpr(std::unique_ptr<Expr> &expr) -> CherryResult {
  auto loc = tokenLoc();
  consume(Token::kw_if);
  unique_ptr<Expr> condition;
  unique_ptr<BlockExpr> thenBlock;
  unique_ptr<BlockExpr> elseBlock;
  if (parseBlockCondition(condition) ||
      parseToken(Token::l_brace, diag::expected_l_brace) ||
      parseBlockExpr(thenBlock) ||
      parseToken(Token::kw_else, diag::expected_else) ||
      parseToken(Token::l_brace, diag::expected_l_brace) ||
      parseBlockExpr(elseBlock))
    return failure();

  expr = make_unique<IfExpr>(loc, std::move(condition), std::move(thenBlock),
                             std::move(elseBlock));
  return success();
}

auto Parser::parseWhileExpr(std::unique_ptr<Expr> &expr) -> CherryResult {
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

auto Parser::parseForExpr(std::unique_ptr<Expr> &expr) -> CherryResult {
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

auto Parser::parseDecimal(unique_ptr<Expr> &expr) -> CherryResult {
  auto loc = tokenLoc();
  if (auto value = token().getUInt64IntegerValue()) {
    consume(Token::decimal);
    expr = make_unique<DecimalLiteralExpr>(loc, std::move(*value));
    return success();
  }
  return emitError(diag::integer_literal_overflows);
}

auto Parser::parseFloat(unique_ptr<Expr> &expr) -> CherryResult {
  auto loc = tokenLoc();
  if (auto value = token().getFloat32Value()) {
    consume(Token::float_literal);
    expr = make_unique<FloatLiteralExpr>(loc, std::move(*value));
    return success();
  }
  return emitError(diag::float_literal_invalid);
}

auto Parser::parseNegativeFloat(unique_ptr<Expr> &expr) -> CherryResult {
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

auto Parser::parseString(unique_ptr<Expr> &expr) -> CherryResult {
  auto loc = tokenLoc();
  if (auto value = token().getStringLiteralValue()) {
    consume(Token::string_literal);
    expr = make_unique<StringLiteralExpr>(loc, std::move(*value));
    return success();
  }
  return emitError(diag::string_literal_invalid);
}

auto Parser::parseIdentifierExpr(unique_ptr<Expr> &expr) -> CherryResult {
  auto location = tokenLoc();
  std::string name;
  if (parseQualifiedName(name, diag::expected_id))
    return failure();
  switch (tokenKind()) {
  case Token::l_paren:
    return parseFunctionCall(location, name, expr);
  case Token::l_brace:
    if (_stopBeforeStructLiteral) {
      expr = createMemberAccessChain(location, name);
      return success();
    }
    return parseStructLiteral(location, name, expr);
  default:
    expr = createMemberAccessChain(location, name);
    return success();
  }
}

auto Parser::parseFunctionCall(llvm::SMLoc location, std::string_view name,
                               unique_ptr<Expr> &expr) -> CherryResult {
  consume(Token::l_paren);
  auto expressions = VectorUniquePtr<Expr>();
  if (parseExpressions(expressions, Token::comma, Token::r_paren,
                       diag::expected_comma_or_r_paren, diag::expected_r_paren))
    return failure();
  expr = make_unique<CallExpr>(location, name, std::move(expressions));
  return success();
}

auto Parser::parseStructLiteral(llvm::SMLoc location, std::string_view name,
                                unique_ptr<Expr> &expr) -> CherryResult {
  consume(Token::l_brace);
  auto expressions = VectorUniquePtr<Expr>();
  if (parseExpressions(expressions, Token::comma, Token::r_brace,
                       diag::expected_comma_or_r_brace, diag::expected_r_brace))
    return failure();
  expr = make_unique<StructLiteralExpr>(location, name, std::move(expressions));
  return success();
}

auto Parser::parseBinaryExpRHS(int exprPrec, std::unique_ptr<Expr> &expr)
    -> CherryResult {
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
    -> CherryResult {
  auto location = tokenLoc();
  consume(Token::dot);

  auto fieldName = spelling();
  if (parseToken(Token::identifier, diag::expected_id))
    return failure();

  expr =
      std::make_unique<MemberExpr>(location, std::move(expr), fieldName);
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
    return 400;
  case Token::kw_lt:
  case Token::kw_le:
  case Token::kw_gt:
  case Token::kw_ge:
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
    return BinaryExpr::Operator::EQ;
  case Token::kw_neq:
    return BinaryExpr::Operator::NEQ;
  case Token::kw_lt:
    return BinaryExpr::Operator::LT;
  case Token::kw_le:
    return BinaryExpr::Operator::LE;
  case Token::kw_gt:
    return BinaryExpr::Operator::GT;
  case Token::kw_ge:
    return BinaryExpr::Operator::GE;
  default:
    llvm_unreachable("Unexpected operator");
  }
}

// _____________________________________________________________________________
// Parse Statements

auto Parser::parseStatementWithoutSemi(unique_ptr<Stat> &stat) -> CherryResult {
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

auto Parser::parseVarDecl(unique_ptr<Stat> &stat) -> CherryResult {
  return parseVariableDecl(stat, /*isConst=*/false);
}

auto Parser::parseConstDecl(unique_ptr<Stat> &stat) -> CherryResult {
  return parseVariableDecl(stat, /*isConst=*/true);
}

auto Parser::parseVariableDecl(unique_ptr<Stat> &stat, bool isConst)
    -> CherryResult {
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
