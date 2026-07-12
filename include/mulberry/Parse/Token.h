//===--- Token.h - Mulberry Language Token ------------------------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_TOKEN_H
#define MULBERRY_TOKEN_H

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mulberry {

struct StringLiteralSegment {
  enum class Kind {
    Text,
    Interpolation,
  };

  Kind kind;
  size_t sourceOffset;
  size_t sourceLength;
  std::string value;
};

class Token {
public:
  enum Kind : int {
#define TOK_MARKER(NAME) NAME,
#define TOK_IDENTIFIER(NAME) NAME,
#define TOK_LITERAL(NAME) NAME,
#define TOK_PUNCTUATION(NAME, SPELLING) NAME,
#define TOK_KEYWORD(SPELLING) kw_##SPELLING,
#include "TokenKinds.def"
  };

  Token(Kind kind, llvm::StringRef spelling)
      : _kind(kind), _spelling(spelling) {}

  auto getSpelling() const -> llvm::StringRef { return _spelling; }

  auto getKind() const -> Kind { return _kind; }
  auto is(Kind K) const -> bool { return _kind == K; }

  auto getUInt64IntegerValue() const -> std::optional<uint64_t>;
  auto getFloat32Value() const -> std::optional<llvm::APFloat>;
  auto getStringLiteralSegments() const
      -> std::optional<std::vector<StringLiteralSegment>>;
  auto getCharLiteralValue() const -> std::optional<uint8_t>;

  llvm::SMLoc getLoc() const;
  llvm::SMLoc getEndLoc() const;
  llvm::SMRange getLocRange() const;

  auto getTokenName() -> const char *;

private:
  Kind _kind;
  llvm::StringRef _spelling;
};

} // end namespace mulberry

#endif // MULBERRY_TOKEN_H
