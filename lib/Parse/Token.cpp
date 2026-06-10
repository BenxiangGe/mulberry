//===--- Token.cpp - Cherry Language Token --------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/Parse/Token.h"
#include "llvm/ADT/StringSwitch.h"

using namespace cherry;
using llvm::SMLoc;
using llvm::SMRange;

auto Token::getUInt64IntegerValue() const -> std::optional<uint64_t> {
  uint64_t result = 0;
  if (_spelling.getAsInteger(10, result))
    return std::nullopt;
  return result;
}

auto Token::getFloat32Value() const -> std::optional<llvm::APFloat> {
  return llvm::APFloat(llvm::APFloat::IEEEsingle(), _spelling);
}

auto Token::getStringLiteralValue() const -> std::optional<std::string> {
  if (_kind != string_literal || _spelling.size() < 2 ||
      _spelling.front() != '"' || _spelling.back() != '"')
    return std::nullopt;

  // The lexer keeps the source spelling, including quotes and escapes. Decode
  // here so AST/Sema see the source-level String value, not token text.
  std::string result;
  for (size_t i = 1; i + 1 < _spelling.size(); ++i) {
    auto c = _spelling[i];
    if (c != '\\') {
      result.push_back(c);
      continue;
    }

    if (++i + 1 >= _spelling.size())
      return std::nullopt;

    switch (_spelling[i]) {
    case 'n':
      result.push_back('\n');
      break;
    case 't':
      result.push_back('\t');
      break;
    case '"':
      result.push_back('"');
      break;
    case '\\':
      result.push_back('\\');
      break;
    default:
      return std::nullopt;
    }
  }

  return result;
}

SMLoc Token::getLoc() const { return SMLoc::getFromPointer(_spelling.data()); }

SMLoc Token::getEndLoc() const {
  return SMLoc::getFromPointer(_spelling.data() + _spelling.size());
}

SMRange Token::getLocRange() const { return SMRange(getLoc(), getEndLoc()); }

auto Token::getTokenName() -> const char * {
  switch (_kind) {
#define TOK_MARKER(NAME)                                                       \
  case NAME:                                                                   \
    return #NAME;
#define TOK_IDENTIFIER(NAME)                                                   \
  case NAME:                                                                   \
    return #NAME;
#define TOK_LITERAL(NAME)                                                      \
  case NAME:                                                                   \
    return #NAME;
#define TOK_PUNCTUATION(NAME, SPELLING)                                        \
  case NAME:                                                                   \
    return #NAME;
#define TOK_KEYWORD(SPELLING)                                                  \
  case kw_##SPELLING:                                                          \
    return "kw_" #SPELLING;
#include "cherry/Parse/TokenKinds.def"
  }
}
