//===--- IntegerLiteral.h - Mulberry Integer Literal Helpers ------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_INTEGER_LITERAL_H
#define MULBERRY_INTEGER_LITERAL_H

#include <cstdint>
#include <optional>
#include <string_view>

namespace mulberry {

auto isValidIntegerLiteralSpelling(std::string_view spelling) -> bool;
auto parseUInt64IntegerLiteral(std::string_view spelling)
    -> std::optional<uint64_t>;

} // namespace mulberry

#endif // MULBERRY_INTEGER_LITERAL_H
