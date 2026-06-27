//===--- Lexer.h - Mulberry Language Lexer ------------------------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_LEXER_H
#define MULBERRY_LEXER_H

#include "mulberry/Parse/Token.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

namespace mulberry {

class Lexer {
public:
  explicit Lexer(const llvm::SourceMgr &sourceMgr);
  Lexer(const llvm::SourceMgr &sourceMgr, unsigned bufferID);

  auto lexToken() -> Token;

  static auto tokenize(const llvm::SourceMgr &sourceManager, Lexer &lexer)
      -> void {
    while (true) {
      auto token = lexer.lexToken();
      if (token.is(Token::eof)) {
        break;
      }
      auto lineCol = sourceManager.getLineAndColumn(token.getLoc());
      auto line = std::get<0>(lineCol);
      auto col = std::get<1>(lineCol);
      llvm::errs() << token.getTokenName() << " '" << token.getSpelling()
                   << "' loc=" << line << ":" << col << "\n";
    }
  }

private:
  auto formToken(Token::Kind kind, const char *tokStart) -> Token {
    return Token(kind, llvm::StringRef(tokStart, _curPtr - tokStart));
  }

  auto lexIdentifierOrKeyword(const char *tokStart) -> Token;
  auto lexNumber(const char *tokStart) -> Token;
  auto lexString(const char *tokStart) -> Token;
  auto lexChar(const char *tokStart) -> Token;

  llvm::StringRef _curBuffer;
  const char *_curPtr;

  Lexer(const Lexer &) = delete;
  auto operator=(const Lexer &) -> void = delete;
};

} // end namespace mulberry

#endif // MULBERRY_LEXER_H
