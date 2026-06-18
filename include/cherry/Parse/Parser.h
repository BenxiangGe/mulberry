//===--- Parser.h - Cherry Language Parser ----------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_PARSER_H
#define CHERRY_PARSER_H

#include "cherry/AST/AST.h"
#include "cherry/Basic/CherryResult.h"
#include "cherry/Parse/DiagnosticsParse.h"
#include "cherry/Parse/Lexer.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <string>
#include <string_view>

namespace cherry {

using llvm::failure;
using llvm::success;

class Parser {
public:
  Parser(std::unique_ptr<Lexer> lexer, llvm::SourceMgr &sourceManager)
      : _token{lexer->lexToken()}, _lexer{std::move(lexer)},
        _sourceManager{sourceManager} {}

  auto parseModule(std::unique_ptr<Module> &module) -> CherryResult;

private:
  Token _token;
  std::unique_ptr<Lexer> _lexer;
  llvm::SourceMgr &_sourceManager;
  bool _stopBeforeStructLiteral = false;

  // ___________________________________________________________________________
  // Lex

  auto token() -> Token & { return _token; }

  auto tokenIs(Token::Kind kind) -> bool { return token().is(kind); }

  auto tokenKind() -> Token::Kind { return token().getKind(); }

  auto tokenLoc() -> llvm::SMLoc { return token().getLoc(); }

  auto spelling() -> llvm::StringRef { return token().getSpelling(); }

  auto consume(Token::Kind kind) -> void {
    assert(token().is(kind) && "consumed an unexpected token");
    token() = _lexer->lexToken();
  }

  auto consumeIf(Token::Kind kind) -> bool {
    if (!token().is(kind))
      return false;
    consume(kind);
    return true;
  }

  // ___________________________________________________________________________
  // Error

  CherryResult emitError(const llvm::Twine &msg) {
    _sourceManager.PrintMessage(tokenLoc(), llvm::SourceMgr::DiagKind::DK_Error,
                                msg);
    return failure();
  }

  // ___________________________________________________________________________

  auto parseToken(Token::Kind expected, const llvm::Twine &message)
      -> CherryResult {
    if (consumeIf(expected))
      return success();
    return emitError(message);
  }

  template <typename T, typename ParseElement>
  auto parseList(Token::Kind separator, Token::Kind end,
                 const char *const separator_error, const char *const end_error,
                 VectorUniquePtr<T> &elements, ParseElement parseElement)
      -> CherryResult;

  // _____________________________________________________________________________
  // Parse Identifiers

  auto parseType(std::unique_ptr<TypeNode> &typeNode) -> CherryResult;

  auto parseQualifiedName(std::string &name,
                          const char *const message) -> CherryResult;

  auto parseFunctionName(std::unique_ptr<FunctionName> &functionName,
                         const char *const message) -> CherryResult;

  auto parseStructName(std::unique_ptr<StructName> &structName,
                       const char *const message) -> CherryResult;

  auto parseUnitType(std::unique_ptr<TypeNode> &typeNode) -> CherryResult;
  auto parseListType(std::unique_ptr<TypeNode> &typeNode,
                     llvm::SMLoc location) -> CherryResult;

  // ___________________________________________________________________________
  // Parse Declarations

  auto parseDeclaration(std::unique_ptr<Decl> &decl) -> CherryResult;

  auto parseFunctionDecl(std::unique_ptr<Decl> &decl) -> CherryResult;

  auto parsePrototype(std::unique_ptr<Prototype> &proto) -> CherryResult;

  auto parseBlockExpr(std::unique_ptr<BlockExpr> &block) -> CherryResult;

  auto parseStructDecl(std::unique_ptr<Decl> &elem) -> CherryResult;

  auto parseTensorTypeSuffix(std::vector<int64_t> &shape) -> CherryResult;
  auto parseArrayLiteral(std::unique_ptr<Expr> &expr) -> CherryResult;
  auto parseIndex(std::unique_ptr<Expr> &expr) -> CherryResult;

  // ___________________________________________________________________________
  // Parse Expressions

  auto parseExpression(std::unique_ptr<Expr> &expr) -> CherryResult;
  auto parseBlockCondition(std::unique_ptr<Expr> &expr) -> CherryResult;

  auto parseExpressions(VectorUniquePtr<Expr> &elem, Token::Kind separator,
                        Token::Kind end, const char *const separator_error,
                        const char *const end_error) -> CherryResult;

  auto parsePrimaryExpression(std::unique_ptr<Expr> &expr) -> CherryResult;

  auto parseVariableExpr(std::unique_ptr<VariableExpr> &identifier)
      -> CherryResult;

  auto parseIfExpr(std::unique_ptr<Expr> &expr) -> CherryResult;

  auto parseWhileExpr(std::unique_ptr<Expr> &expr) -> CherryResult;
  auto parseForExpr(std::unique_ptr<Expr> &expr) -> CherryResult;

  auto parseDecimal(std::unique_ptr<Expr> &expr) -> CherryResult;
  auto parseFloat(std::unique_ptr<Expr> &expr) -> CherryResult;
  auto parseNegativeFloat(std::unique_ptr<Expr> &expr) -> CherryResult;
  auto parseString(std::unique_ptr<Expr> &expr) -> CherryResult;

  auto parseIdentifierExpr(std::unique_ptr<Expr> &expr) -> CherryResult;

  auto parseFunctionCall(llvm::SMLoc location, std::string_view name,
                         std::unique_ptr<Expr> &expr) -> CherryResult;
  auto parseStructLiteral(llvm::SMLoc location, std::string_view name,
                          std::unique_ptr<Expr> &expr) -> CherryResult;

  auto parseBinaryExpRHS(int exprPrec, std::unique_ptr<Expr> &expr)
      -> CherryResult;
  auto parseMemberAccess(std::unique_ptr<Expr> &expr) -> CherryResult;
  auto getTokenPrecedence() -> int;
  auto isTokenRightAssociative() -> bool;
  auto tokenToOperator(Token token) -> BinaryExpr::Operator;

  // ___________________________________________________________________________
  // Parse Statements

  auto parseStatementWithoutSemi(std::unique_ptr<Stat> &stat) -> CherryResult;

  auto parseVarDecl(std::unique_ptr<Stat> &stat) -> CherryResult;
  auto parseConstDecl(std::unique_ptr<Stat> &stat) -> CherryResult;
  auto parseVariableDecl(std::unique_ptr<Stat> &stat, bool isConst)
      -> CherryResult;
};

} // end namespace cherry

#endif // CHERRY_PARSER_H
