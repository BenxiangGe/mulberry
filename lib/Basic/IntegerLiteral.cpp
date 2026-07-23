//===--- IntegerLiteral.cpp - Mulberry Integer Literal Helpers ----*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "mulberry/Basic/IntegerLiteral.h"

#include <cstddef>
#include <limits>

using namespace mulberry;

namespace {

auto hexDigitValue(char value) -> std::optional<uint8_t> {
  if (value >= '0' && value <= '9')
    return static_cast<uint8_t>(value - '0');
  if (value >= 'a' && value <= 'f')
    return static_cast<uint8_t>(value - 'a' + 10);
  if (value >= 'A' && value <= 'F')
    return static_cast<uint8_t>(value - 'A' + 10);
  return std::nullopt;
}

auto isHexLiteral(std::string_view spelling) -> bool {
  return spelling.size() >= 2 && spelling[0] == '0' && spelling[1] == 'x';
}

} // namespace

auto mulberry::isValidIntegerLiteralSpelling(std::string_view spelling)
    -> bool {
  if (spelling.empty())
    return false;

  if (!isHexLiteral(spelling)) {
    for (char value : spelling) {
      if (value < '0' || value > '9')
        return false;
    }
    return true;
  }

  if (spelling.size() == 2)
    return false;

  bool hasSeparator = false;
  bool isFirstGroup = true;
  size_t groupLength = 0;
  for (size_t index = 2; index < spelling.size(); ++index) {
    char value = spelling[index];
    if (value == '_') {
      if (groupLength == 0 || groupLength > 4 ||
          (!isFirstGroup && groupLength != 4))
        return false;
      hasSeparator = true;
      isFirstGroup = false;
      groupLength = 0;
      continue;
    }

    if (!hexDigitValue(value))
      return false;
    ++groupLength;
  }

  if (groupLength == 0)
    return false;
  if (hasSeparator && !isFirstGroup && groupLength != 4)
    return false;
  return true;
}

auto mulberry::parseUInt64IntegerLiteral(std::string_view spelling)
    -> std::optional<uint64_t> {
  if (!isValidIntegerLiteralSpelling(spelling))
    return std::nullopt;

  uint64_t radix = 10;
  size_t index = 0;
  if (isHexLiteral(spelling)) {
    radix = 16;
    index = 2;
  }

  uint64_t result = 0;
  for (; index < spelling.size(); ++index) {
    char value = spelling[index];
    if (value == '_')
      continue;

    auto digit = hexDigitValue(value);
    if (!digit || *digit >= radix)
      return std::nullopt;
    if (result > (std::numeric_limits<uint64_t>::max() - *digit) / radix)
      return std::nullopt;
    result = result * radix + *digit;
  }
  return result;
}
