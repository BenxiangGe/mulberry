//===--- Lexer.cpp - Mulberry Language Lexer --------------------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "mulberry/Parse/Lexer.h"
#include "llvm/ADT/StringSwitch.h"
#include <cctype>

using namespace mulberry;

namespace {
auto isDigit(char c) -> bool {
  return std::isdigit(static_cast<unsigned char>(c));
}

auto isIdentifierStart(char c) -> bool {
  return std::isalpha(static_cast<unsigned char>(c)) || c == '_';
}

auto isIdentifierContinue(char c) -> bool {
  return isIdentifierStart(c) || isDigit(c);
}
} // namespace

Lexer::Lexer(const llvm::SourceMgr &sourceMgr) {
  auto bufferID = sourceMgr.getMainFileID();
  _curBuffer = sourceMgr.getMemoryBuffer(bufferID)->getBuffer();
  _curPtr = _curBuffer.begin();
}

Lexer::Lexer(const llvm::SourceMgr &sourceMgr, unsigned bufferID) {
  _curBuffer = sourceMgr.getMemoryBuffer(bufferID)->getBuffer();
  _curPtr = _curBuffer.begin();
}

auto Lexer::lexToken() -> Token {
  while (true) {
  Restart:
    if (atEnd())
      return formToken(Token::eof, _curPtr);

    const char *tokStart = _curPtr;
    switch (*_curPtr++) {
    default:
      if (isIdentifierStart(_curPtr[-1]))
        return lexIdentifierOrKeyword(tokStart);

      return formToken(Token::error, tokStart);
    case ' ':
    case '\t':
    case '\n':
    case '\r':
      // Handle whitespace.
      continue;
    case 0:
      continue;
    case ';':
      return formToken(Token::semi, tokStart);
    case ',':
      return formToken(Token::comma, tokStart);
    case '?':
      return formToken(Token::question, tokStart);
    case '(':
      return formToken(Token::l_paren, tokStart);
    case ')':
      return formToken(Token::r_paren, tokStart);
    case '<':
      if (peek() == '=') {
        ++_curPtr;
        return formToken(Token::less_equal, tokStart);
      }
      return formToken(Token::less, tokStart);
    case '>':
      if (peek() == '=') {
        ++_curPtr;
        return formToken(Token::greater_equal, tokStart);
      }
      return formToken(Token::greater, tokStart);
    case '{':
      return formToken(Token::l_brace, tokStart);
    case '}':
      return formToken(Token::r_brace, tokStart);
    case '[':
      return formToken(Token::l_square, tokStart);
    case ']':
      return formToken(Token::r_square, tokStart);
    case ':':
      return formToken(Token::colon, tokStart);
    case '.':
      if (peek() == '.') {
        ++_curPtr;
        return formToken(Token::dot_dot, tokStart);
      }
      return formToken(Token::dot, tokStart);
    case '=':
      if (peek() == '=') {
        ++_curPtr;
        return formToken(Token::eq, tokStart);
      }
      if (peek() == '>') {
        ++_curPtr;
        return formToken(Token::fat_arrow, tokStart);
      }
      return formToken(Token::assign, tokStart);
    case '!':
      if (peek() == '=') {
        ++_curPtr;
        return formToken(Token::neq, tokStart);
      }
      return formToken(Token::error, tokStart);
    case '+':
      return formToken(Token::add, tokStart);
    case '-':
      return formToken(Token::diff, tokStart);
    case '*':
      return formToken(Token::star, tokStart);
    case '/':
      return formToken(Token::div, tokStart);
    case '%':
      return formToken(Token::rem, tokStart);
    case '|':
      return formToken(Token::pipe, tokStart);
    case '"':
      return lexString(tokStart);
    case '\'':
      return lexChar(tokStart);
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
      return lexNumber(tokStart);
    case '#': {
      if (_mode == Mode::StringInterpolation)
        return formToken(Token::error, tokStart);
      while (true) {
        if (peek() == '\n' || peek() == 0) {
          goto Restart;
        }
        _curPtr++;
      }
    }
    }
  }
}

auto Lexer::lexIdentifierOrKeyword(const char *tokStart) -> Token {
  // Match [A-Za-z_][A-Za-z0-9_]*
  while (isIdentifierContinue(peek()))
    ++_curPtr;

  llvm::StringRef spelling(tokStart, _curPtr - tokStart);

  Token::Kind kind = llvm::StringSwitch<Token::Kind>(spelling)
#define TOK_KEYWORD(SPELLING) .Case(#SPELLING, Token::kw_##SPELLING)
#include "TokenKinds.def"
                         .Default(Token::identifier);

  return Token(kind, spelling);
}

auto Lexer::lexNumber(const char *tokStart) -> Token {
  while (isDigit(peek()))
    ++_curPtr;

  bool isFloat = false;
  if (peek() == '.' && isDigit(peek(1))) {
    isFloat = true;
    ++_curPtr;
    while (isDigit(peek()))
      ++_curPtr;
  }

  if ((peek() == 'e' || peek() == 'E') &&
      (isDigit(peek(1)) ||
       ((peek(1) == '+' || peek(1) == '-') && isDigit(peek(2))))) {
    isFloat = true;
    ++_curPtr;
    if (peek() == '+' || peek() == '-')
      ++_curPtr;
    while (isDigit(peek()))
      ++_curPtr;
  }

  return formToken(isFloat ? Token::float_literal : Token::decimal, tokStart);
}

auto Lexer::lexString(const char *tokStart) -> Token {
  // Escape validation is handled when the token is decoded; lexing only needs
  // to keep escaped quotes from terminating the token early.
  while (!atEnd() && peek() != 0) {
    if (peek() == '"') {
      ++_curPtr;
      return formToken(Token::string_literal, tokStart);
    }
    if (peek() == '\n' || peek() == '\r')
      return formToken(Token::error, tokStart);
    if (peek() == '\\') {
      ++_curPtr;
      if (atEnd() || peek() == 0 || peek() == '\n' || peek() == '\r')
        return formToken(Token::error, tokStart);
    }
    ++_curPtr;
  }

  return formToken(Token::error, tokStart);
}

auto Lexer::lexChar(const char *tokStart) -> Token {
  if (atEnd() || peek() == 0 || peek() == '\n' || peek() == '\r')
    return formToken(Token::error, tokStart);

  if (peek() == '\\') {
    ++_curPtr;
    if (atEnd() || peek() == 0 || peek() == '\n' || peek() == '\r')
      return formToken(Token::error, tokStart);
  }

  ++_curPtr;
  if (atEnd() || peek() != '\'')
    return formToken(Token::error, tokStart);

  ++_curPtr;
  return formToken(Token::char_literal, tokStart);
}
