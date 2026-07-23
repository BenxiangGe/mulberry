//===--- BigIntRuntimeTest.cpp - BigInt runtime tests --------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "mulberry/Runtime/BigInt.h"

#include "gtest/gtest.h"

#include <string_view>

namespace {

TEST(BigIntRuntimeTest, CreatesCanonicalUnsignedValues) {
  auto *zero = mulberry_bigint_from_uint64(0);
  EXPECT_EQ(zero->sign, 0);
  EXPECT_EQ(zero->limbCount, 0u);

  auto *value = mulberry_bigint_from_uint64(0x100000001ULL);
  EXPECT_EQ(value->sign, 1);
  ASSERT_EQ(value->limbCount, 2u);
  EXPECT_EQ(value->limbs[0], 1u);
  EXPECT_EQ(value->limbs[1], 1u);
}

TEST(BigIntRuntimeTest, ParsesDecimalAndHexLiterals) {
  auto *decimal = mulberry_bigint_from_literal("4294967296", 10);
  ASSERT_EQ(decimal->limbCount, 2u);
  EXPECT_EQ(decimal->limbs[0], 0u);
  EXPECT_EQ(decimal->limbs[1], 1u);

  auto *hex = mulberry_bigint_from_literal("0x1_0000_0000", 13);
  ASSERT_EQ(hex->limbCount, 2u);
  EXPECT_EQ(hex->limbs[0], 0u);
  EXPECT_EQ(hex->limbs[1], 1u);

  auto *negative = mulberry_bigint_neg(hex);
  EXPECT_EQ(negative->sign, -1);
  ASSERT_EQ(negative->limbCount, 2u);
  EXPECT_EQ(negative->limbs[0], 0u);
  EXPECT_EQ(negative->limbs[1], 1u);
}

TEST(BigIntRuntimeTest, AddsSubtractsAndNegates) {
  auto *left = mulberry_bigint_from_literal("0xffffffff", 10);
  auto *one = mulberry_bigint_from_uint64(1);
  auto *sum = mulberry_bigint_add(left, one);
  ASSERT_EQ(sum->limbCount, 2u);
  EXPECT_EQ(sum->limbs[0], 0u);
  EXPECT_EQ(sum->limbs[1], 1u);

  auto *difference = mulberry_bigint_sub(sum, one);
  ASSERT_EQ(difference->limbCount, 1u);
  EXPECT_EQ(difference->limbs[0], 0xffffffffu);

  auto *negative = mulberry_bigint_neg(difference);
  EXPECT_EQ(negative->sign, -1);
  EXPECT_EQ(mulberry_bigint_compare(negative, difference), -1);

  auto *zero = mulberry_bigint_add(negative, difference);
  EXPECT_EQ(zero->sign, 0);
  EXPECT_EQ(zero->limbCount, 0u);
}

TEST(BigIntRuntimeTest, MultipliesAndComparesSignedValues) {
  auto *value = mulberry_bigint_from_literal("0xffffffff", 10);
  auto *product = mulberry_bigint_mul(value, value);
  ASSERT_EQ(product->limbCount, 2u);
  EXPECT_EQ(product->limbs[0], 1u);
  EXPECT_EQ(product->limbs[1], 0xfffffffeu);

  auto *negative = mulberry_bigint_neg(value);
  auto *negativeProduct = mulberry_bigint_mul(negative, value);
  EXPECT_EQ(negativeProduct->sign, -1);
  EXPECT_EQ(mulberry_bigint_compare(negativeProduct, product), -1);
  EXPECT_EQ(mulberry_bigint_compare(product, product), 0);

  auto *wide = mulberry_bigint_from_literal("0xffff_ffff_ffff_ffff", 21);
  auto *wideProduct = mulberry_bigint_mul(wide, wide);
  ASSERT_EQ(wideProduct->limbCount, 4u);
  EXPECT_EQ(wideProduct->limbs[0], 1u);
  EXPECT_EQ(wideProduct->limbs[1], 0u);
  EXPECT_EQ(wideProduct->limbs[2], 0xfffffffeu);
  EXPECT_EQ(wideProduct->limbs[3], 0xffffffffu);
}

TEST(BigIntRuntimeTest, DividesSignedValuesAndNarrowsToUInt64) {
  auto *seven = mulberry_bigint_from_uint64(7);
  auto *three = mulberry_bigint_from_uint64(3);
  auto *two = mulberry_bigint_from_uint64(2);
  auto *one = mulberry_bigint_from_uint64(1);
  auto *negativeSeven = mulberry_bigint_neg(seven);
  auto *negativeThree = mulberry_bigint_neg(three);
  auto *negativeTwo = mulberry_bigint_neg(two);
  auto *negativeOne = mulberry_bigint_neg(one);

  EXPECT_EQ(mulberry_bigint_compare(mulberry_bigint_div(seven, three), two),
            0);
  EXPECT_EQ(mulberry_bigint_compare(mulberry_bigint_rem(seven, three), one),
            0);
  EXPECT_EQ(mulberry_bigint_compare(
                mulberry_bigint_div(negativeSeven, three), negativeTwo),
            0);
  EXPECT_EQ(mulberry_bigint_compare(
                mulberry_bigint_rem(negativeSeven, three), negativeOne),
            0);
  EXPECT_EQ(mulberry_bigint_compare(
                mulberry_bigint_div(seven, negativeThree), negativeTwo),
            0);
  EXPECT_EQ(mulberry_bigint_compare(
                mulberry_bigint_rem(negativeSeven, negativeThree),
                negativeOne),
            0);

  auto *wide = mulberry_bigint_from_literal("0x1_0000_0000_0000_0000", 23);
  auto *word = mulberry_bigint_from_literal("0x1_0000_0000", 13);
  EXPECT_EQ(mulberry_bigint_compare(mulberry_bigint_div(wide, word), word),
            0);
  EXPECT_EQ(mulberry_bigint_compare(mulberry_bigint_rem(wide, word),
                                    mulberry_bigint_from_uint64(0)),
            0);

  auto *max = mulberry_bigint_from_literal("18446744073709551615", 20);
  EXPECT_EQ(mulberry_bigint_to_uint64(max), 18446744073709551615ULL);
}

TEST(BigIntRuntimeTest, ShiftsAndUsesSignedBitwiseSemantics) {
  auto *one = mulberry_bigint_from_uint64(1);
  auto *wide = mulberry_bigint_shift_left(one, 64);
  ASSERT_EQ(wide->limbCount, 3u);
  EXPECT_EQ(wide->limbs[0], 0u);
  EXPECT_EQ(wide->limbs[1], 0u);
  EXPECT_EQ(wide->limbs[2], 1u);
  EXPECT_EQ(mulberry_bigint_compare(mulberry_bigint_shift_right(wide, 64),
                                    one),
            0);

  auto *negativeThree =
      mulberry_bigint_neg(mulberry_bigint_from_uint64(3));
  auto *negativeTwo = mulberry_bigint_neg(mulberry_bigint_from_uint64(2));
  auto *negativeOne = mulberry_bigint_neg(one);
  EXPECT_EQ(mulberry_bigint_compare(
                mulberry_bigint_shift_right(negativeThree, 1), negativeTwo),
            0);
  EXPECT_EQ(mulberry_bigint_compare(
                mulberry_bigint_shift_right(negativeOne, 4096), negativeOne),
            0);

  auto *mask = mulberry_bigint_from_uint64(0xff);
  auto *two = mulberry_bigint_from_uint64(2);
  auto *expectedAnd = mulberry_bigint_from_uint64(253);
  auto *expectedOr = negativeOne;
  auto *expectedXor =
      mulberry_bigint_neg(mulberry_bigint_from_uint64(4));
  EXPECT_EQ(mulberry_bigint_compare(
                mulberry_bigint_bit_and(negativeThree, mask), expectedAnd),
            0);
  EXPECT_EQ(mulberry_bigint_compare(
                mulberry_bigint_bit_or(negativeThree, two), expectedOr),
            0);
  EXPECT_EQ(mulberry_bigint_compare(
                mulberry_bigint_bit_xor(negativeThree, one), expectedXor),
            0);
}

TEST(BigIntRuntimeTest, FormatsDecimalAndGroupedHex) {
  auto *wide = mulberry_bigint_from_literal("0x1_0000_0000_0000_0000", 23);
  auto *negativeWide = mulberry_bigint_neg(wide);

  MulberryString decimal;
  _mlir_ciface_mulberry_bigint_to_string(&decimal, wide);
  EXPECT_EQ(std::string_view(reinterpret_cast<const char*>(decimal.data),
                             decimal.length),
            "18446744073709551616");

  MulberryString negativeDecimal;
  _mlir_ciface_mulberry_bigint_to_string(&negativeDecimal, negativeWide);
  EXPECT_EQ(std::string_view(
                reinterpret_cast<const char*>(negativeDecimal.data),
                negativeDecimal.length),
            "-18446744073709551616");

  MulberryString hex;
  _mlir_ciface_mulberry_bigint_to_hex_string(&hex, wide);
  EXPECT_EQ(std::string_view(reinterpret_cast<const char*>(hex.data),
                             hex.length),
            "0x1_0000_0000_0000_0000");

  MulberryString negativeHex;
  _mlir_ciface_mulberry_bigint_to_hex_string(&negativeHex, negativeWide);
  EXPECT_EQ(std::string_view(reinterpret_cast<const char*>(negativeHex.data),
                             negativeHex.length),
            "-0x1_0000_0000_0000_0000");
}

} // namespace
