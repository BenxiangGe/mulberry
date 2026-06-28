//===--- Parser.h - Mulberry Language Parser ----------------------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_PARSER_H
#define MULBERRY_PARSER_H

#include "mulberry/AST/AST.h"
#include "mulberry/Basic/MulberryResult.h"
#include "mulberry/Parse/DiagnosticsParse.h"
#include "mulberry/Parse/Lexer.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace mulberry {

using llvm::failure;
using llvm::success;

class Parser {
public:
  Parser(std::unique_ptr<Lexer> lexer, llvm::SourceMgr &sourceManager)
      : _token{lexer->lexToken()}, _lexer{std::move(lexer)},
        _sourceManager{sourceManager} {}

  auto parseModule(std::unique_ptr<Module> &module) -> MulberryResult;

private:
  Token _token;
  std::unique_ptr<Lexer> _lexer;
  llvm::SourceMgr &_sourceManager;
  bool _stopBeforeStructLiteral = false;
  std::string _packageName;

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

  MulberryResult emitError(const llvm::Twine &msg) {
    _sourceManager.PrintMessage(tokenLoc(), llvm::SourceMgr::DiagKind::DK_Error,
                                msg);
    return failure();
  }

  // ___________________________________________________________________________

  auto parseToken(Token::Kind expected, const llvm::Twine &message)
      -> MulberryResult {
    if (consumeIf(expected))
      return success();
    return emitError(message);
  }

  template <typename T, typename ParseElement>
  auto parseList(Token::Kind separator, Token::Kind end,
                 const char *const separator_error, const char *const end_error,
                 VectorUniquePtr<T> &elements, ParseElement parseElement)
      -> MulberryResult;

  // _____________________________________________________________________________
  // Parse Identifiers

  auto parsePackageDecl() -> MulberryResult;
  auto parseType(std::unique_ptr<TypeNode> &typeNode) -> MulberryResult;

  auto parseQualifiedName(std::string &name,
                          const char *const message) -> MulberryResult;

  auto parseFunctionName(std::unique_ptr<FunctionName> &functionName,
                         const char *const message) -> MulberryResult;

  auto parseStructName(std::unique_ptr<StructName> &structName,
                       const char *const message) -> MulberryResult;

  auto qualifyPackageName(std::string_view name) const -> std::string;

  auto parseUnitType(std::unique_ptr<TypeNode> &typeNode) -> MulberryResult;
  auto parseGenericTypeArgs(std::vector<ComptimeArg> &arguments)
      -> MulberryResult;
  auto parseComptimeParams(std::vector<ComptimeParam> &parameters)
      -> MulberryResult;
  auto parseListType(std::unique_ptr<TypeNode> &typeNode,
                     llvm::SMLoc location) -> MulberryResult;
  auto parsePtrType(std::unique_ptr<TypeNode> &typeNode,
                    llvm::SMLoc location) -> MulberryResult;

  // ___________________________________________________________________________
  // Parse Declarations

  auto parseDeclaration(std::unique_ptr<Decl> &decl) -> MulberryResult;

  auto parseImportDecl(std::unique_ptr<Decl> &decl) -> MulberryResult;

  auto parseFunctionDecl(std::unique_ptr<Decl> &decl) -> MulberryResult;

  auto parseExternFunctionDecl(std::unique_ptr<Decl> &decl) -> MulberryResult;

  auto parsePrototype(std::unique_ptr<Prototype> &proto,
                      bool qualifyName = true) -> MulberryResult;
  auto parseFunctionDeclBody(llvm::SMLoc location,
                             std::unique_ptr<Prototype> proto,
                             std::unique_ptr<FunctionDecl> &functionDecl)
      -> MulberryResult;
  auto parseStructMethod(std::unique_ptr<FunctionDecl> &method)
      -> MulberryResult;

  auto parseBlockExpr(std::unique_ptr<BlockExpr> &block) -> MulberryResult;

  auto parseStructDecl(std::unique_ptr<Decl> &elem) -> MulberryResult;
  auto parseComptimeTypeAliasDecl(std::unique_ptr<Decl> &decl) -> MulberryResult;
  auto parseStructMembers(VectorUniquePtr<VariableStat> &fields,
                          VectorUniquePtr<FunctionDecl> &methods)
      -> MulberryResult;
  auto parseComptimeAliasBody(std::unique_ptr<TypeNode> &typeNode)
      -> MulberryResult;

  auto parseTensorTypeSuffix(std::vector<int64_t> &shape) -> MulberryResult;
  auto parseArrayLiteral(std::unique_ptr<Expr> &expr) -> MulberryResult;
  auto parseIndex(std::unique_ptr<Expr> &expr) -> MulberryResult;

  // ___________________________________________________________________________
  // Parse Expressions

  auto parseExpression(std::unique_ptr<Expr> &expr) -> MulberryResult;
  auto parseBlockCondition(std::unique_ptr<Expr> &expr) -> MulberryResult;

  auto parseExpressions(VectorUniquePtr<Expr> &elem, Token::Kind separator,
                        Token::Kind end, const char *const separator_error,
                        const char *const end_error) -> MulberryResult;

  auto parsePrimaryExpression(std::unique_ptr<Expr> &expr) -> MulberryResult;

  auto parseVariableExpr(std::unique_ptr<VariableExpr> &identifier)
      -> MulberryResult;

  auto parseIfExpr(std::unique_ptr<Expr> &expr) -> MulberryResult;

  auto parseWhileExpr(std::unique_ptr<Expr> &expr) -> MulberryResult;
  auto parseBreakExpr(std::unique_ptr<Expr> &expr) -> MulberryResult;
  auto parseContinueExpr(std::unique_ptr<Expr> &expr) -> MulberryResult;
  auto parseForExpr(std::unique_ptr<Expr> &expr) -> MulberryResult;

  auto parseDecimal(std::unique_ptr<Expr> &expr) -> MulberryResult;
  auto parseFloat(std::unique_ptr<Expr> &expr) -> MulberryResult;
  auto parseNegativeFloat(std::unique_ptr<Expr> &expr) -> MulberryResult;
  auto parseString(std::unique_ptr<Expr> &expr) -> MulberryResult;
  auto parseChar(std::unique_ptr<Expr> &expr) -> MulberryResult;
  auto parseDerefExpr(std::unique_ptr<Expr> &expr) -> MulberryResult;

  auto parseIdentifierExpr(std::unique_ptr<Expr> &expr) -> MulberryResult;

  auto parseTypeLayoutExpr(llvm::SMLoc location, std::string_view name,
                           std::unique_ptr<Expr> &expr) -> MulberryResult;

  auto parseHeapAllocExpr(llvm::SMLoc location, std::string_view name,
                          std::unique_ptr<Expr> &expr) -> MulberryResult;

  auto parseZeroInitExpr(llvm::SMLoc location,
                         std::unique_ptr<Expr> &expr) -> MulberryResult;

  auto parseTensorPackExpr(llvm::SMLoc location, std::string_view name,
                           std::unique_ptr<Expr> &expr) -> MulberryResult;

  auto parseTensorViewExpr(llvm::SMLoc location, std::string_view name,
                           std::unique_ptr<Expr> &expr) -> MulberryResult;

  auto parseFunctionCall(llvm::SMLoc location, std::string_view name,
                         std::unique_ptr<Expr> &expr) -> MulberryResult;
  auto parseStructLiteral(llvm::SMLoc location,
                          std::unique_ptr<TypeNode> typeNode,
                          std::unique_ptr<Expr> &expr) -> MulberryResult;
  auto parseGenericStructLiteral(llvm::SMLoc location, std::string_view name,
                                 std::unique_ptr<Expr> &expr) -> MulberryResult;

  auto parseBinaryExpRHS(int exprPrec, std::unique_ptr<Expr> &expr)
      -> MulberryResult;
  auto parseMemberAccess(std::unique_ptr<Expr> &expr) -> MulberryResult;
  auto getTokenPrecedence() -> int;
  auto isTokenRightAssociative() -> bool;
  auto tokenToOperator(Token token) -> BinaryExpr::Operator;

  // ___________________________________________________________________________
  // Parse Statements

  auto parseStatementWithoutSemi(std::unique_ptr<Stat> &stat) -> MulberryResult;

  auto parseVarDecl(std::unique_ptr<Stat> &stat) -> MulberryResult;
  auto parseConstDecl(std::unique_ptr<Stat> &stat) -> MulberryResult;
  auto parseVariableDecl(std::unique_ptr<Stat> &stat, bool isConst)
      -> MulberryResult;
};

} // end namespace mulberry

#endif // MULBERRY_PARSER_H
