//===--- Parser.h - Mulberry Language Parser ----------------------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_PARSER_H
#define MULBERRY_PARSER_H

#include "mulberry/AST/AST.h"
#include "mulberry/Parse/DiagnosticsParse.h"
#include "mulberry/Parse/Lexer.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "llvm/Support/LogicalResult.h"
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

  auto parseModule(std::unique_ptr<Module> &module) -> llvm::LogicalResult;

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

  llvm::LogicalResult emitError(const llvm::Twine &msg) {
    _sourceManager.PrintMessage(tokenLoc(), llvm::SourceMgr::DiagKind::DK_Error,
                                msg);
    return failure();
  }

  // ___________________________________________________________________________

  auto parseToken(Token::Kind expected, const llvm::Twine &message)
      -> llvm::LogicalResult {
    if (consumeIf(expected))
      return success();
    return emitError(message);
  }

  auto isGenericClosingToken() -> bool;
  auto parseGenericClose() -> llvm::LogicalResult;

  template <typename T, typename ParseElement>
  auto parseList(Token::Kind separator, Token::Kind end,
                 const char *const separator_error, const char *const end_error,
                 VectorUniquePtr<T> &elements, ParseElement parseElement)
      -> llvm::LogicalResult;

  // _____________________________________________________________________________
  // Parse Identifiers

  auto parsePackageDecl() -> llvm::LogicalResult;
  auto parseType(std::unique_ptr<TypeNode> &typeNode) -> llvm::LogicalResult;

  auto parseQualifiedName(std::string &name,
                          const char *const message) -> llvm::LogicalResult;

  auto parseFunctionName(std::unique_ptr<FunctionName> &functionName,
                         const char *const message) -> llvm::LogicalResult;

  auto parseStructName(std::unique_ptr<StructName> &structName,
                       const char *const message) -> llvm::LogicalResult;

  auto qualifyPackageName(std::string_view name) const -> std::string;

  auto parseUnitType(std::unique_ptr<TypeNode> &typeNode) -> llvm::LogicalResult;
  auto parseFunctionType(std::unique_ptr<TypeNode> &typeNode) -> llvm::LogicalResult;
  auto parseGenericTypeArgs(std::vector<ComptimeArg> &arguments)
      -> llvm::LogicalResult;
  auto parseComptimeParams(std::vector<ComptimeParam> &parameters,
                           bool allowTraitConstraint = false)
      -> llvm::LogicalResult;
  auto parsePtrType(std::unique_ptr<TypeNode> &typeNode,
                    llvm::SMLoc location) -> llvm::LogicalResult;

  // ___________________________________________________________________________
  // Parse Declarations

  auto parseDeclaration(std::unique_ptr<Decl> &decl) -> llvm::LogicalResult;

  auto parseImportDecl(std::unique_ptr<Decl> &decl) -> llvm::LogicalResult;

  auto parseFunctionDecl(std::unique_ptr<Decl> &decl) -> llvm::LogicalResult;

  auto parseExternFunctionDecl(std::unique_ptr<Decl> &decl) -> llvm::LogicalResult;

  auto parsePrototype(std::unique_ptr<Prototype> &proto,
                      bool qualifyName = true) -> llvm::LogicalResult;
  auto parseFunctionDeclBody(llvm::SMLoc location,
                             std::unique_ptr<Prototype> proto,
                             std::unique_ptr<FunctionDecl> &functionDecl)
      -> llvm::LogicalResult;
  auto parseStructMethod(std::unique_ptr<FunctionDecl> &method)
      -> llvm::LogicalResult;

  auto parseBlockExpr(std::unique_ptr<BlockExpr> &block) -> llvm::LogicalResult;
  auto parseComptimeBlock(std::unique_ptr<ComptimeBlockExpr> &block)
      -> llvm::LogicalResult;

  auto parseStructDecl(std::unique_ptr<Decl> &elem) -> llvm::LogicalResult;
  auto parseTraitDecl(std::unique_ptr<Decl> &decl) -> llvm::LogicalResult;
  auto parseTraitMethod(std::unique_ptr<TraitMethodDecl> &method)
      -> llvm::LogicalResult;
  auto parseImplDecl(std::unique_ptr<Decl> &decl) -> llvm::LogicalResult;
  auto parseDataDecl(std::unique_ptr<Decl> &decl) -> llvm::LogicalResult;
  auto parseComptimeTypeAliasDecl(std::unique_ptr<Decl> &decl) -> llvm::LogicalResult;
  auto parseStructMembers(VectorUniquePtr<FieldDecl> &fields,
                          VectorUniquePtr<FunctionDecl> &methods)
      -> llvm::LogicalResult;
  auto parseComptimeAliasBody(std::unique_ptr<TypeNode> &typeNode)
      -> llvm::LogicalResult;

  auto parseArrayTypeSuffix(std::vector<int64_t> &shape) -> llvm::LogicalResult;
  auto parseArrayLiteral(std::unique_ptr<Expr> &expr) -> llvm::LogicalResult;
  auto parseIndex(std::unique_ptr<Expr> &expr) -> llvm::LogicalResult;

  // ___________________________________________________________________________
  // Parse Expressions

  auto parseExpression(std::unique_ptr<Expr> &expr) -> llvm::LogicalResult;
  auto parseBlockCondition(std::unique_ptr<Expr> &expr) -> llvm::LogicalResult;

  auto parseExpressions(VectorUniquePtr<Expr> &elem, Token::Kind separator,
                        Token::Kind end, const char *const separator_error,
                        const char *const end_error) -> llvm::LogicalResult;

  auto parsePrimaryExpression(std::unique_ptr<Expr> &expr) -> llvm::LogicalResult;

  auto parseVariableExpr(std::unique_ptr<VariableExpr> &identifier)
      -> llvm::LogicalResult;

  auto parseLambdaExpr(std::unique_ptr<Expr> &expr) -> llvm::LogicalResult;
  auto parseMatchExpr(std::unique_ptr<Expr> &expr) -> llvm::LogicalResult;
  auto parseMatchExprArmBlock(std::unique_ptr<BlockExpr> &bodyBlock,
                              std::unique_ptr<Expr> &resultExpr)
      -> llvm::LogicalResult;
  auto parseDataPattern(std::unique_ptr<DataPattern> &pattern)
      -> llvm::LogicalResult;

  auto parseIntegerLiteral(std::unique_ptr<Expr> &expr) -> llvm::LogicalResult;
  auto parseFloat(std::unique_ptr<Expr> &expr) -> llvm::LogicalResult;
  auto parseNegativeFloat(std::unique_ptr<Expr> &expr) -> llvm::LogicalResult;
  auto parseString(std::unique_ptr<Expr> &expr) -> llvm::LogicalResult;
  auto parseChar(std::unique_ptr<Expr> &expr) -> llvm::LogicalResult;

  auto parseIdentifierExpr(std::unique_ptr<Expr> &expr) -> llvm::LogicalResult;

  auto parseTypeLayoutExpr(llvm::SMLoc location, std::string_view name,
                           std::unique_ptr<Expr> &expr) -> llvm::LogicalResult;

  auto parseTypeInfoExpr(llvm::SMLoc location,
                         std::unique_ptr<Expr> &expr) -> llvm::LogicalResult;

  auto parseIntrinsicExpr(llvm::SMLoc location,
                          std::unique_ptr<Expr> &expr) -> llvm::LogicalResult;

  auto parseObjectIdentityExpr(llvm::SMLoc location,
                               std::unique_ptr<Expr> &expr) -> llvm::LogicalResult;

  auto parseHeapAllocExpr(llvm::SMLoc location, std::string_view name,
                          std::unique_ptr<Expr> &expr) -> llvm::LogicalResult;

  auto parseFunctionCall(llvm::SMLoc location, std::string_view name,
                         std::unique_ptr<Expr> &expr) -> llvm::LogicalResult;
  auto parseDataConstructorExpr(llvm::SMLoc location, std::string_view name,
                                std::unique_ptr<Expr> &expr)
      -> llvm::LogicalResult;
  auto parseStructLiteral(llvm::SMLoc location,
                          std::unique_ptr<TypeNode> typeNode,
                          std::unique_ptr<Expr> &expr) -> llvm::LogicalResult;
  auto parseGenericStructLiteral(llvm::SMLoc location, std::string_view name,
                                 std::unique_ptr<Expr> &expr) -> llvm::LogicalResult;

  auto parseBinaryExpRHS(int exprPrec, std::unique_ptr<Expr> &expr)
      -> llvm::LogicalResult;
  auto parseMemberAccess(std::unique_ptr<Expr> &expr) -> llvm::LogicalResult;
  auto getTokenPrecedence() -> int;
  auto isTokenRightAssociative() -> bool;
  auto tokenToOperator(Token token) -> BinaryExpr::Operator;

  // ___________________________________________________________________________
  // Parse Statements

  auto parseStatement(std::unique_ptr<Stat> &stat) -> llvm::LogicalResult;
  auto parseStatementWithoutSemi(std::unique_ptr<Stat> &stat) -> llvm::LogicalResult;

  auto parseVarDecl(std::unique_ptr<Stat> &stat) -> llvm::LogicalResult;
  auto parseConstDecl(std::unique_ptr<Stat> &stat) -> llvm::LogicalResult;
  auto parseIfStat(std::unique_ptr<Stat> &stat) -> llvm::LogicalResult;
  auto parseMatchStat(std::unique_ptr<Stat> &stat) -> llvm::LogicalResult;
  auto parseWhileStat(std::unique_ptr<Stat> &stat) -> llvm::LogicalResult;
  auto parseForStat(std::unique_ptr<Stat> &stat) -> llvm::LogicalResult;
  auto parseBreakStat(std::unique_ptr<Stat> &stat) -> llvm::LogicalResult;
  auto parseContinueStat(std::unique_ptr<Stat> &stat) -> llvm::LogicalResult;
  auto parseReturnStat(std::unique_ptr<Stat> &stat) -> llvm::LogicalResult;
  auto parseVariableDecl(std::unique_ptr<Stat> &stat, bool isConst)
      -> llvm::LogicalResult;
};

} // end namespace mulberry

#endif // MULBERRY_PARSER_H
