//===--- BigInt.cpp - Mulberry BigInt runtime ----------------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "mulberry/Runtime/BigInt.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

extern "C" void* mulberry_boehm_malloc(uint64_t size);

namespace {

using Limb = uint32_t;
using Limbs = std::vector<Limb>;

auto requireValue(const MulberryBigInt* value) -> const MulberryBigInt* {
  if (value)
    return value;
  std::abort();
}

auto canonicalLimbCount(const Limbs& limbs) -> size_t {
  auto count = limbs.size();
  while (count != 0 && limbs[count - 1] == 0)
    --count;
  return count;
}

auto allocationSize(size_t limbCount) -> uint64_t {
  auto count = limbCount == 0 ? size_t{1} : limbCount;
  auto headerBytes = static_cast<uint64_t>(offsetof(MulberryBigInt, limbs));
  if (count > (std::numeric_limits<uint64_t>::max() - headerBytes) /
                  sizeof(Limb))
    std::abort();
  auto bytes = headerBytes + count * sizeof(Limb);
  if (bytes < sizeof(MulberryBigInt))
    bytes = sizeof(MulberryBigInt);
  return bytes;
}

auto makeValue(int8_t sign, const Limbs& limbs) -> MulberryBigInt* {
  auto count = canonicalLimbCount(limbs);
  auto* result = static_cast<MulberryBigInt*>
      (mulberry_boehm_malloc(allocationSize(count)));
  result->sign = count == 0 ? 0 : sign;
  result->limbCount = count;
  for (size_t index = 0; index < count; ++index)
    result->limbs[index] = limbs[index];
  return result;
}

auto makeString(std::string_view value) -> MulberryString {
  auto size = static_cast<uint64_t>(value.size());
  auto* data = static_cast<uint8_t*>(mulberry_boehm_malloc(size + 1));
  if (!value.empty())
    std::memcpy(data, value.data(), value.size());
  data[size] = 0;
  return {size, data};
}

auto copyValue(const MulberryBigInt* value) -> MulberryBigInt* {
  Limbs limbs(value->limbs, value->limbs + value->limbCount);
  return makeValue(value->sign, limbs);
}

auto compareMagnitude(const MulberryBigInt* lhs, const MulberryBigInt* rhs)
    -> int32_t {
  if (lhs->limbCount != rhs->limbCount)
    return lhs->limbCount < rhs->limbCount ? -1 : 1;

  for (size_t index = lhs->limbCount; index != 0; --index) {
    auto lhsLimb = lhs->limbs[index - 1];
    auto rhsLimb = rhs->limbs[index - 1];
    if (lhsLimb != rhsLimb)
      return lhsLimb < rhsLimb ? -1 : 1;
  }
  return 0;
}

auto addMagnitude(const MulberryBigInt* lhs, const MulberryBigInt* rhs)
    -> Limbs {
  Limbs result;
  auto count = lhs->limbCount > rhs->limbCount ? lhs->limbCount
                                               : rhs->limbCount;
  result.resize(count);

  uint64_t carry = 0;
  for (size_t index = 0; index < count; ++index) {
    auto lhsLimb = index < lhs->limbCount ? lhs->limbs[index] : 0;
    auto rhsLimb = index < rhs->limbCount ? rhs->limbs[index] : 0;
    auto sum = uint64_t{lhsLimb} + rhsLimb + carry;
    result[index] = static_cast<Limb>(sum);
    carry = sum >> 32;
  }
  if (carry != 0)
    result.push_back(static_cast<Limb>(carry));
  return result;
}

auto subtractMagnitude(const MulberryBigInt* larger,
                       const MulberryBigInt* smaller) -> Limbs {
  Limbs result;
  result.resize(larger->limbCount);

  uint64_t borrow = 0;
  for (size_t index = 0; index < larger->limbCount; ++index) {
    auto largerLimb = larger->limbs[index];
    auto smallerLimb = index < smaller->limbCount ? smaller->limbs[index] : 0;
    auto subtrahend = uint64_t{smallerLimb} + borrow;
    result[index] = static_cast<Limb>(uint64_t{largerLimb} - subtrahend);
    borrow = uint64_t{largerLimb} < subtrahend ? 1 : 0;
  }
  return result;
}

auto multiplyMagnitude(const MulberryBigInt* lhs, const MulberryBigInt* rhs)
    -> Limbs {
  if (lhs->limbCount == 0 || rhs->limbCount == 0)
    return {};

  Limbs result;
  result.resize(lhs->limbCount + rhs->limbCount);
  for (size_t lhsIndex = 0; lhsIndex < lhs->limbCount; ++lhsIndex) {
    uint64_t carry = 0;
    for (size_t rhsIndex = 0; rhsIndex < rhs->limbCount; ++rhsIndex) {
      auto index = lhsIndex + rhsIndex;
      auto product = uint64_t{lhs->limbs[lhsIndex]} * rhs->limbs[rhsIndex] +
                     result[index] + carry;
      result[index] = static_cast<Limb>(product);
      carry = product >> 32;
    }

    auto index = lhsIndex + rhs->limbCount;
    while (carry != 0) {
      auto sum = uint64_t{result[index]} + carry;
      result[index] = static_cast<Limb>(sum);
      carry = sum >> 32;
      ++index;
    }
  }
  return result;
}

auto checkedLimbCount(const MulberryBigInt* value) -> size_t {
  if (value->limbCount > std::numeric_limits<size_t>::max())
    std::abort();
  return static_cast<size_t>(value->limbCount);
}

auto trimMagnitude(Limbs& limbs) -> void {
  limbs.resize(canonicalLimbCount(limbs));
}

auto divideMagnitudeByWord(Limbs& dividend, uint32_t divisor) -> uint32_t {
  if (divisor == 0)
    std::abort();

  uint64_t remainder = 0;
  for (size_t index = dividend.size(); index != 0; --index) {
    auto value = (remainder << 32) | dividend[index - 1];
    dividend[index - 1] = static_cast<Limb>(value / divisor);
    remainder = value % divisor;
  }
  trimMagnitude(dividend);
  return static_cast<uint32_t>(remainder);
}

auto formatDecimal(const MulberryBigInt* value) -> MulberryString {
  value = requireValue(value);
  if (value->sign == 0)
    return makeString("0");

  Limbs magnitude(value->limbs, value->limbs + checkedLimbCount(value));
  std::vector<uint32_t> groups;
  // Repeated base-10^9 division keeps decimal formatting independent of a
  // host arbitrary-precision type while every intermediate fits in uint64_t.
  while (!magnitude.empty())
    groups.push_back(divideMagnitudeByWord(magnitude, 1000000000U));

  std::string spelling;
  if (value->sign < 0)
    spelling += '-';
  spelling += std::to_string(groups.back());
  for (size_t index = groups.size() - 1; index != 0; --index) {
    auto group = std::to_string(groups[index - 1]);
    spelling.append(9 - group.size(), '0');
    spelling += group;
  }
  return makeString(std::string_view(spelling));
}

auto appendHexMagnitude(std::string& digits, const MulberryBigInt* value)
    -> void {
  constexpr char hexDigits[] = "0123456789abcdef";
  auto limbCount = checkedLimbCount(value);
  auto topLimb = value->limbs[limbCount - 1];
  bool emittedDigit = false;
  for (int32_t shift = 28; shift >= 0; shift -= 4) {
    auto digit = (topLimb >> shift) & 0xfU;
    if (digit != 0 || emittedDigit) {
      digits += hexDigits[digit];
      emittedDigit = true;
    }
  }

  for (size_t index = limbCount - 1; index != 0; --index) {
    auto limb = value->limbs[index - 1];
    for (int32_t shift = 28; shift >= 0; shift -= 4)
      digits += hexDigits[(limb >> shift) & 0xfU];
  }
}

auto formatHex(const MulberryBigInt* value) -> MulberryString {
  value = requireValue(value);
  if (value->sign == 0)
    return makeString("0x0");

  std::string digits;
  appendHexMagnitude(digits, value);

  std::string spelling;
  if (value->sign < 0)
    spelling += '-';
  spelling += "0x";
  for (size_t index = 0; index < digits.size(); ++index) {
    if (index != 0 && (digits.size() - index) % 4 == 0)
      spelling += '_';
    spelling += digits[index];
  }
  return makeString(std::string_view(spelling));
}

auto compareMagnitude(const Limbs& lhs, const MulberryBigInt* rhs) -> int32_t {
  auto lhsCount = canonicalLimbCount(lhs);
  auto rhsCount = checkedLimbCount(rhs);
  if (lhsCount != rhsCount)
    return lhsCount < rhsCount ? -1 : 1;

  for (size_t index = lhsCount; index != 0; --index) {
    auto lhsLimb = lhs[index - 1];
    auto rhsLimb = rhs->limbs[index - 1];
    if (lhsLimb != rhsLimb)
      return lhsLimb < rhsLimb ? -1 : 1;
  }
  return 0;
}

auto shiftMagnitudeLeftOne(Limbs& limbs) -> void {
  uint64_t carry = 0;
  for (auto& limb : limbs) {
    auto shifted = (uint64_t{limb} << 1) | carry;
    limb = static_cast<Limb>(shifted);
    carry = shifted >> 32;
  }
  if (carry != 0)
    limbs.push_back(static_cast<Limb>(carry));
}

auto subtractMagnitudeInPlace(Limbs& larger,
                              const MulberryBigInt* smaller) -> void {
  auto smallerCount = checkedLimbCount(smaller);
  uint64_t borrow = 0;
  for (size_t index = 0; index < larger.size(); ++index) {
    auto smallerLimb = index < smallerCount ? smaller->limbs[index] : 0;
    auto subtrahend = uint64_t{smallerLimb} + borrow;
    auto largerLimb = larger[index];
    larger[index] = static_cast<Limb>(uint64_t{largerLimb} - subtrahend);
    borrow = uint64_t{largerLimb} < subtrahend ? 1 : 0;
  }
  if (borrow != 0)
    std::abort();
  trimMagnitude(larger);
}

auto magnitudeBitLength(const MulberryBigInt* value) -> size_t {
  auto limbCount = checkedLimbCount(value);
  if (limbCount == 0)
    return 0;
  if (limbCount > std::numeric_limits<size_t>::max() / 32)
    std::abort();

  auto topLimb = value->limbs[limbCount - 1];
  size_t topBits = 0;
  while (topLimb != 0) {
    ++topBits;
    topLimb >>= 1;
  }
  return (limbCount - 1) * 32 + topBits;
}

struct MagnitudeDivision {
  Limbs quotient;
  Limbs remainder;
};

auto divideMagnitude(const MulberryBigInt* dividend,
                     const MulberryBigInt* divisor) -> MagnitudeDivision {
  if (divisor->sign == 0)
    std::abort();

  auto dividendLimbCount = checkedLimbCount(dividend);
  Limbs quotient(dividendLimbCount);
  Limbs remainder;
  auto bitCount = magnitudeBitLength(dividend);
  // Restoring binary long division consumes one dividend bit at a time.
  for (size_t bitsRemaining = bitCount; bitsRemaining != 0;
       --bitsRemaining) {
    shiftMagnitudeLeftOne(remainder);

    auto bitIndex = bitsRemaining - 1;
    auto bit = Limb{1} << (bitIndex % 32);
    if ((dividend->limbs[bitIndex / 32] & bit) != 0) {
      if (remainder.empty())
        remainder.push_back(1);
      else
        remainder[0] |= 1;
    }

    if (compareMagnitude(remainder, divisor) >= 0) {
      subtractMagnitudeInPlace(remainder, divisor);
      quotient[bitIndex / 32] |= bit;
    }
  }

  trimMagnitude(quotient);
  return {std::move(quotient), std::move(remainder)};
}

auto incrementMagnitude(Limbs& limbs) -> void {
  for (auto& limb : limbs) {
    ++limb;
    if (limb != 0)
      return;
  }
  limbs.push_back(1);
}

auto shiftMagnitudeLeft(const MulberryBigInt* value, uint64_t count)
    -> Limbs {
  auto limbCount = checkedLimbCount(value);
  if (limbCount == 0)
    return {};

  auto wordCount = count / 32;
  if (wordCount > std::numeric_limits<size_t>::max())
    std::abort();
  auto wordShift = static_cast<size_t>(wordCount);
  if (wordShift > std::numeric_limits<size_t>::max() - limbCount)
    std::abort();

  auto resultCount = wordShift + limbCount;
  auto bitShift = static_cast<uint32_t>(count % 32);
  if (bitShift != 0) {
    if (resultCount == std::numeric_limits<size_t>::max())
      std::abort();
    ++resultCount;
  }

  Limbs result(resultCount);
  uint64_t carry = 0;
  for (size_t index = 0; index < limbCount; ++index) {
    auto shifted = (uint64_t{value->limbs[index]} << bitShift) | carry;
    result[index + wordShift] = static_cast<Limb>(shifted);
    carry = shifted >> 32;
  }
  if (bitShift != 0)
    result[limbCount + wordShift] = static_cast<Limb>(carry);
  return result;
}

struct RightShiftMagnitude {
  Limbs limbs;
  bool discardedBits = false;
};

auto shiftMagnitudeRight(const MulberryBigInt* value, uint64_t count)
    -> RightShiftMagnitude {
  auto limbCount = checkedLimbCount(value);
  auto wordCount = count / 32;
  if (wordCount >= value->limbCount)
    return {{}, limbCount != 0};

  auto wordShift = static_cast<size_t>(wordCount);
  auto bitShift = static_cast<uint32_t>(count % 32);
  bool discardedBits = false;
  for (size_t index = 0; index < wordShift; ++index) {
    if (value->limbs[index] != 0) {
      discardedBits = true;
      break;
    }
  }
  if (bitShift != 0) {
    auto mask = (uint32_t{1} << bitShift) - 1;
    discardedBits = discardedBits ||
                    (value->limbs[wordShift] & mask) != 0;
  }

  Limbs result(limbCount - wordShift);
  for (size_t index = wordShift; index < limbCount; ++index) {
    auto shifted = uint64_t{value->limbs[index]} >> bitShift;
    if (bitShift != 0 && index + 1 < limbCount)
      shifted |= uint64_t{value->limbs[index + 1]} << (32 - bitShift);
    result[index - wordShift] = static_cast<Limb>(shifted);
  }
  return {std::move(result), discardedBits};
}

auto addOneModulo(Limbs& words) -> void {
  for (auto& word : words) {
    ++word;
    if (word != 0)
      return;
  }
}

auto makeTwosComplementWords(const MulberryBigInt* value, size_t wordCount)
    -> Limbs {
  Limbs words(wordCount);
  auto limbCount = checkedLimbCount(value);
  for (size_t index = 0; index < limbCount; ++index)
    words[index] = value->limbs[index];

  if (value->sign < 0) {
    for (auto& word : words)
      word = ~word;
    addOneModulo(words);
  }
  return words;
}

auto makeValueFromTwosComplement(Limbs words) -> MulberryBigInt* {
  if ((words.back() & (uint32_t{1} << 31)) == 0)
    return makeValue(1, words);

  for (auto& word : words)
    word = ~word;
  addOneModulo(words);
  return makeValue(-1, words);
}

enum class BitwiseOperation {
  And,
  Or,
  Xor,
};

auto applyBitwise(const MulberryBigInt* lhs, const MulberryBigInt* rhs,
                  BitwiseOperation operation) -> MulberryBigInt* {
  auto limbCount = lhs->limbCount > rhs->limbCount ? lhs->limbCount
                                                    : rhs->limbCount;
  if (limbCount >= std::numeric_limits<size_t>::max())
    std::abort();
  auto wordCount = static_cast<size_t>(limbCount) + 1;

  // One extra sign limb makes this finite representation equivalent to the
  // infinite two's-complement sign extension used by signed bit operations.
  auto leftWords = makeTwosComplementWords(lhs, wordCount);
  auto rightWords = makeTwosComplementWords(rhs, wordCount);
  for (size_t index = 0; index < wordCount; ++index) {
    switch (operation) {
    case BitwiseOperation::And:
      leftWords[index] &= rightWords[index];
      break;
    case BitwiseOperation::Or:
      leftWords[index] |= rightWords[index];
      break;
    case BitwiseOperation::Xor:
      leftWords[index] ^= rightWords[index];
      break;
    }
  }
  return makeValueFromTwosComplement(std::move(leftWords));
}

auto digitValue(char digit) -> int32_t {
  if (digit >= '0' && digit <= '9')
    return digit - '0';
  if (digit >= 'a' && digit <= 'f')
    return digit - 'a' + 10;
  if (digit >= 'A' && digit <= 'F')
    return digit - 'A' + 10;
  return -1;
}

auto parseLiteral(std::string_view spelling) -> MulberryBigInt* {
  if (spelling.empty())
    return nullptr;

  uint32_t base = 10;
  size_t index = 0;
  if (spelling.size() >= 2 && spelling[0] == '0' && spelling[1] == 'x') {
    base = 16;
    index += 2;
    if (index == spelling.size())
      return nullptr;

    // Keep the runtime constant ABI aligned with the source literal grammar:
    // decimal literals have no separator, while hexadecimal separators group
    // exactly four digits after the first group.
    bool hasSeparator = false;
    bool isFirstGroup = true;
    size_t groupLength = 0;
    for (size_t groupIndex = index; groupIndex < spelling.size();
         ++groupIndex) {
      auto character = spelling[groupIndex];
      if (character == '_') {
        if (groupLength == 0 || groupLength > 4 ||
            (!isFirstGroup && groupLength != 4))
          return nullptr;
        hasSeparator = true;
        isFirstGroup = false;
        groupLength = 0;
        continue;
      }
      if (digitValue(character) < 0 || digitValue(character) >= 16)
        return nullptr;
      ++groupLength;
    }
    if (groupLength == 0 ||
        (hasSeparator && !isFirstGroup && groupLength != 4))
      return nullptr;
  } else {
    for (char character : spelling) {
      if (character < '0' || character > '9')
        return nullptr;
    }
  }

  Limbs limbs;
  bool sawDigit = false;
  for (; index < spelling.size(); ++index) {
    auto character = spelling[index];
    if (character == '_')
      continue;
    auto digit = digitValue(character);
    if (digit < 0 || static_cast<uint32_t>(digit) >= base)
      return nullptr;
    sawDigit = true;

    uint64_t carry = static_cast<uint32_t>(digit);
    for (auto &limb : limbs) {
      auto product = uint64_t{limb} * base + carry;
      limb = static_cast<Limb>(product);
      carry = product >> 32;
    }
    if (carry != 0)
      limbs.push_back(static_cast<Limb>(carry));
  }
  if (!sawDigit)
    return nullptr;

  return makeValue(1, limbs);
}

} // namespace

extern "C" MulberryBigInt* mulberry_bigint_from_uint64(uint64_t value) {
  Limbs limbs;
  if (value != 0) {
    limbs.push_back(static_cast<Limb>(value));
    auto high = static_cast<Limb>(value >> 32);
    if (high != 0)
      limbs.push_back(high);
  }
  return makeValue(1, limbs);
}

extern "C" MulberryBigInt* mulberry_bigint_from_literal(const char* spelling,
                                                         uint64_t length) {
  if (!spelling)
    return nullptr;
  return parseLiteral(std::string_view(spelling, length));
}

extern "C" MulberryBigInt* mulberry_bigint_neg(const MulberryBigInt* value) {
  value = requireValue(value);
  Limbs limbs(value->limbs, value->limbs + value->limbCount);
  return makeValue(-value->sign, limbs);
}

extern "C" MulberryBigInt* mulberry_bigint_add(const MulberryBigInt* lhs,
                                                const MulberryBigInt* rhs) {
  lhs = requireValue(lhs);
  rhs = requireValue(rhs);
  if (lhs->sign == 0)
    return copyValue(rhs);
  if (rhs->sign == 0)
    return copyValue(lhs);

  if (lhs->sign == rhs->sign)
    return makeValue(lhs->sign, addMagnitude(lhs, rhs));

  auto magnitude = compareMagnitude(lhs, rhs);
  if (magnitude == 0)
    return makeValue(1, {});
  if (magnitude > 0)
    return makeValue(lhs->sign, subtractMagnitude(lhs, rhs));
  return makeValue(rhs->sign, subtractMagnitude(rhs, lhs));
}

extern "C" MulberryBigInt* mulberry_bigint_sub(const MulberryBigInt* lhs,
                                                const MulberryBigInt* rhs) {
  lhs = requireValue(lhs);
  rhs = requireValue(rhs);
  if (rhs->sign == 0)
    return copyValue(lhs);

  Limbs rhsLimbs(rhs->limbs, rhs->limbs + rhs->limbCount);
  auto negatedRhs = makeValue(-rhs->sign, rhsLimbs);
  return mulberry_bigint_add(lhs, negatedRhs);
}

extern "C" MulberryBigInt* mulberry_bigint_mul(const MulberryBigInt* lhs,
                                                const MulberryBigInt* rhs) {
  lhs = requireValue(lhs);
  rhs = requireValue(rhs);
  auto sign = lhs->sign == rhs->sign ? 1 : -1;
  return makeValue(sign, multiplyMagnitude(lhs, rhs));
}

extern "C" MulberryBigInt* mulberry_bigint_div(const MulberryBigInt* lhs,
                                                 const MulberryBigInt* rhs) {
  lhs = requireValue(lhs);
  rhs = requireValue(rhs);
  // Public std.bigint checks the divisor before calling this unchecked ABI.
  if (rhs->sign == 0)
    std::abort();

  auto division = divideMagnitude(lhs, rhs);
  auto sign = lhs->sign == rhs->sign ? 1 : -1;
  return makeValue(sign, division.quotient);
}

extern "C" MulberryBigInt* mulberry_bigint_rem(const MulberryBigInt* lhs,
                                                 const MulberryBigInt* rhs) {
  lhs = requireValue(lhs);
  rhs = requireValue(rhs);
  // Public std.bigint checks the divisor before calling this unchecked ABI.
  if (rhs->sign == 0)
    std::abort();

  auto division = divideMagnitude(lhs, rhs);
  return makeValue(lhs->sign, division.remainder);
}

// Source extern calls use MLIR's C interface wrapper names at the JIT
// boundary. Keep the ordinary C ABI available for runtime unit tests.
extern "C" MulberryBigInt* _mlir_ciface_mulberry_bigint_div(
    const MulberryBigInt* lhs, const MulberryBigInt* rhs) {
  return mulberry_bigint_div(lhs, rhs);
}

extern "C" MulberryBigInt* _mlir_ciface_mulberry_bigint_rem(
    const MulberryBigInt* lhs, const MulberryBigInt* rhs) {
  return mulberry_bigint_rem(lhs, rhs);
}

extern "C" MulberryBigInt* mulberry_bigint_bit_and(
    const MulberryBigInt* lhs, const MulberryBigInt* rhs) {
  return applyBitwise(requireValue(lhs), requireValue(rhs),
                      BitwiseOperation::And);
}

extern "C" MulberryBigInt* mulberry_bigint_bit_or(
    const MulberryBigInt* lhs, const MulberryBigInt* rhs) {
  return applyBitwise(requireValue(lhs), requireValue(rhs),
                      BitwiseOperation::Or);
}

extern "C" MulberryBigInt* mulberry_bigint_bit_xor(
    const MulberryBigInt* lhs, const MulberryBigInt* rhs) {
  return applyBitwise(requireValue(lhs), requireValue(rhs),
                      BitwiseOperation::Xor);
}

extern "C" MulberryBigInt* mulberry_bigint_shift_left(
    const MulberryBigInt* value, uint64_t count) {
  value = requireValue(value);
  return makeValue(value->sign, shiftMagnitudeLeft(value, count));
}

extern "C" MulberryBigInt* mulberry_bigint_shift_right(
    const MulberryBigInt* value, uint64_t count) {
  value = requireValue(value);
  auto shifted = shiftMagnitudeRight(value, count);
  if (value->sign >= 0)
    return makeValue(value->sign, shifted.limbs);
  if (shifted.discardedBits)
    incrementMagnitude(shifted.limbs);
  return makeValue(-1, shifted.limbs);
}

extern "C" int32_t mulberry_bigint_compare(const MulberryBigInt* lhs,
                                            const MulberryBigInt* rhs) {
  lhs = requireValue(lhs);
  rhs = requireValue(rhs);
  if (lhs->sign != rhs->sign)
    return lhs->sign < rhs->sign ? -1 : 1;
  if (lhs->sign == 0)
    return 0;
  auto magnitude = compareMagnitude(lhs, rhs);
  return lhs->sign > 0 ? magnitude : -magnitude;
}

extern "C" uint64_t mulberry_bigint_to_uint64(const MulberryBigInt* value) {
  value = requireValue(value);
  auto limbCount = checkedLimbCount(value);
  if (value->sign < 0 || limbCount > 2)
    std::abort();

  uint64_t result = 0;
  if (limbCount > 0)
    result = value->limbs[0];
  if (limbCount > 1)
    result |= uint64_t{value->limbs[1]} << 32;
  return result;
}

extern "C" uint64_t _mlir_ciface_mulberry_bigint_to_uint64(
    const MulberryBigInt* value) {
  return mulberry_bigint_to_uint64(value);
}

extern "C" void _mlir_ciface_mulberry_bigint_to_string(
    MulberryString* result, const MulberryBigInt* value) {
  *result = formatDecimal(value);
}

extern "C" void _mlir_ciface_mulberry_bigint_to_hex_string(
    MulberryString* result, const MulberryBigInt* value) {
  *result = formatHex(value);
}
