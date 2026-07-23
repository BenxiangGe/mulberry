#include "mulberry/Parse/Lexer.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace mulberry;

namespace {

TEST(LexerTest, firstTest) {
  auto input = R"(fn ; , => == = { } ( ) | & ^ << >> 01 0x1_2345 0x1234_5678_0123 a0 a0a)";
  auto inputBuffer = llvm::MemoryBuffer::getMemBuffer(input, "main.mulberry");

  llvm::SourceMgr sourceManager;
  sourceManager.AddNewSourceBuffer(std::move(inputBuffer),
                                   /*IncludeLoc*/ llvm::SMLoc());

  auto lexer = std::make_unique<Lexer>(sourceManager);
  ASSERT_TRUE(lexer->lexToken().is(Token::kw_fn));
  ASSERT_TRUE(lexer->lexToken().is(Token::semi));
  ASSERT_TRUE(lexer->lexToken().is(Token::comma));
  ASSERT_TRUE(lexer->lexToken().is(Token::fat_arrow));
  ASSERT_TRUE(lexer->lexToken().is(Token::eq));
  ASSERT_TRUE(lexer->lexToken().is(Token::assign));
  ASSERT_TRUE(lexer->lexToken().is(Token::l_brace));
  ASSERT_TRUE(lexer->lexToken().is(Token::r_brace));
  ASSERT_TRUE(lexer->lexToken().is(Token::l_paren));
  ASSERT_TRUE(lexer->lexToken().is(Token::r_paren));
  ASSERT_TRUE(lexer->lexToken().is(Token::pipe));
  ASSERT_TRUE(lexer->lexToken().is(Token::amp));
  ASSERT_TRUE(lexer->lexToken().is(Token::caret));
  ASSERT_TRUE(lexer->lexToken().is(Token::shift_left));
  ASSERT_TRUE(lexer->lexToken().is(Token::shift_right));
  {
    auto token = lexer->lexToken();
    ASSERT_TRUE(token.is(Token::integer_literal));
    ASSERT_EQ(token.getSpelling(), "01");
    auto uint64 = token.getUInt64IntegerLiteralValue();
    ASSERT_TRUE(uint64.has_value());
    ASSERT_EQ(uint64.value(), 1);
  }
  {
    auto token = lexer->lexToken();
    ASSERT_TRUE(token.is(Token::integer_literal));
    ASSERT_EQ(token.getSpelling(), "0x1_2345");
    auto uint64 = token.getUInt64IntegerLiteralValue();
    ASSERT_TRUE(uint64.has_value());
    ASSERT_EQ(uint64.value(), 74565u);
  }
  {
    auto token = lexer->lexToken();
    ASSERT_TRUE(token.is(Token::integer_literal));
    ASSERT_EQ(token.getSpelling(), "0x1234_5678_0123");
    auto uint64 = token.getUInt64IntegerLiteralValue();
    ASSERT_TRUE(uint64.has_value());
    ASSERT_EQ(uint64.value(), 20015998304547u);
  }
  {
    auto token = lexer->lexToken();
    ASSERT_TRUE(token.is(Token::identifier));
    ASSERT_EQ(token.getSpelling(), "a0");
  }
  {
    auto token = lexer->lexToken();
    ASSERT_TRUE(token.is(Token::identifier));
    ASSERT_EQ(token.getSpelling(), "a0a");
  }
  ASSERT_TRUE(lexer->lexToken().is(Token::eof));
}

TEST(LexerTest, boundedSource) {
  std::string input = "items[index].name";
  Lexer lexer(input, Lexer::Mode::StringInterpolation);

  ASSERT_TRUE(lexer.lexToken().is(Token::identifier));
  ASSERT_TRUE(lexer.lexToken().is(Token::l_square));
  ASSERT_TRUE(lexer.lexToken().is(Token::identifier));
  ASSERT_TRUE(lexer.lexToken().is(Token::r_square));
  ASSERT_TRUE(lexer.lexToken().is(Token::dot));
  ASSERT_TRUE(lexer.lexToken().is(Token::identifier));
  ASSERT_TRUE(lexer.lexToken().is(Token::eof));
}

TEST(LexerTest, stringLiteralSegments) {
  auto input = R"("left \{ {$value} \} \\ right" "\{$value}")";
  auto inputBuffer = llvm::MemoryBuffer::getMemBuffer(input, "main.mulberry");

  llvm::SourceMgr sourceManager;
  sourceManager.AddNewSourceBuffer(std::move(inputBuffer), llvm::SMLoc());
  Lexer lexer(sourceManager);

  auto interpolated = lexer.lexToken();
  auto segments = interpolated.getStringLiteralSegments();
  ASSERT_TRUE(segments.has_value());
  ASSERT_EQ(segments->size(), 3u);
  EXPECT_EQ((*segments)[0].kind, StringLiteralSegment::Kind::Text);
  EXPECT_EQ((*segments)[0].value, "left { ");
  EXPECT_EQ((*segments)[1].kind,
            StringLiteralSegment::Kind::Interpolation);
  EXPECT_EQ(interpolated.getSpelling().substr((*segments)[1].sourceOffset,
                                               (*segments)[1].sourceLength),
            "value");
  EXPECT_EQ((*segments)[2].kind, StringLiteralSegment::Kind::Text);
  EXPECT_EQ((*segments)[2].value, " } \\ right");

  auto escaped = lexer.lexToken().getStringLiteralSegments();
  ASSERT_TRUE(escaped.has_value());
  ASSERT_EQ(escaped->size(), 1u);
  EXPECT_EQ(escaped->front().kind, StringLiteralSegment::Kind::Text);
  EXPECT_EQ(escaped->front().value, "{$value}");
}

} // end namespace
