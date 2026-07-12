//===--- Token.cpp - Mulberry Language Token --------------------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "mulberry/Parse/Token.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mulberry;
using llvm::SMLoc;
using llvm::SMRange;

namespace {
auto decodeStringEscape(char value) -> std::optional<char> {
  switch (value) {
  case 'n':
    return '\n';
  case 't':
    return '\t';
  case '"':
    return '"';
  case '\\':
    return '\\';
  case '{':
    return '{';
  case '}':
    return '}';
  default:
    return std::nullopt;
  }
}
} // namespace

auto Token::getUInt64IntegerValue() const -> std::optional<uint64_t> {
  uint64_t result = 0;
  if (_spelling.getAsInteger(10, result))
    return std::nullopt;
  return result;
}

auto Token::getFloat32Value() const -> std::optional<llvm::APFloat> {
  return llvm::APFloat(llvm::APFloat::IEEEsingle(), _spelling);
}

auto Token::getStringLiteralSegments() const
    -> std::optional<std::vector<StringLiteralSegment>> {
  if (_kind != string_literal || _spelling.size() < 2 ||
      _spelling.front() != '"' || _spelling.back() != '"')
    return std::nullopt;

  std::vector<StringLiteralSegment> segments;
  std::string text;
  size_t textOffset = 1;
  bool hasInterpolation = false;

  for (size_t i = 1; i + 1 < _spelling.size();) {
    if (_spelling[i] == '\\') {
      if (i + 2 >= _spelling.size())
        return std::nullopt;
      auto decoded = decodeStringEscape(_spelling[i + 1]);
      if (!decoded)
        return std::nullopt;
      text.push_back(*decoded);
      i += 2;
      continue;
    }

    if (_spelling[i] == '{' && i + 2 < _spelling.size() &&
        _spelling[i + 1] == '$') {
      if (!text.empty()) {
        segments.push_back({StringLiteralSegment::Kind::Text, textOffset,
                            i - textOffset, std::move(text)});
        text.clear();
      }

      auto interpolationOffset = i + 2;
      auto interpolationEnd = interpolationOffset;
      while (interpolationEnd + 1 < _spelling.size() &&
             _spelling[interpolationEnd] != '}')
        ++interpolationEnd;
      if (interpolationEnd + 1 >= _spelling.size())
        return std::nullopt;

      segments.push_back({StringLiteralSegment::Kind::Interpolation,
                          interpolationOffset,
                          interpolationEnd - interpolationOffset, {}});
      hasInterpolation = true;
      i = interpolationEnd + 1;
      textOffset = i;
      continue;
    }

    text.push_back(_spelling[i]);
    ++i;
  }

  if (!text.empty() || !hasInterpolation)
    segments.push_back({StringLiteralSegment::Kind::Text, textOffset,
                        _spelling.size() - 1 - textOffset,
                        std::move(text)});

  return segments;
}

auto Token::getCharLiteralValue() const -> std::optional<uint8_t> {
  if (_kind != char_literal || _spelling.size() < 3 ||
      _spelling.front() != '\'' || _spelling.back() != '\'')
    return std::nullopt;

  if (_spelling.size() == 3)
    return static_cast<uint8_t>(_spelling[1]);

  if (_spelling.size() != 4 || _spelling[1] != '\\')
    return std::nullopt;

  switch (_spelling[2]) {
  case 'n':
    return static_cast<uint8_t>('\n');
  case 't':
    return static_cast<uint8_t>('\t');
  case 'r':
    return static_cast<uint8_t>('\r');
  case '\'':
    return static_cast<uint8_t>('\'');
  case '"':
    return static_cast<uint8_t>('"');
  case '\\':
    return static_cast<uint8_t>('\\');
  default:
    return std::nullopt;
  }
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
#include "mulberry/Parse/TokenKinds.def"
  }
}
