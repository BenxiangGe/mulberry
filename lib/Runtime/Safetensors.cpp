//===--- Safetensors.cpp --------------------------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct TensorInfo {
  std::vector<int64_t> shape;
  uint64_t headerLength = 0;
  int64_t begin = 0;
  int64_t end = 0;
};

struct String {
  uint64_t length;
  uint8_t* data;
};

struct ListUInt64 {
  uint64_t length;
  uint64_t capacity;
  uint64_t* data;
};

[[noreturn]] void fail(const char* message) {
  std::fprintf(stderr, "safetensors error: %s\n", message);
  std::abort();
}

void expect(bool condition, const char* message) {
  if (!condition)
    fail(message);
}

void skipSpaces(const std::string& text, size_t& pos) {
  while (pos < text.size() &&
         std::isspace(static_cast<unsigned char>(text[pos])))
    ++pos;
}

void expectChar(const std::string& text, size_t& pos, char ch) {
  skipSpaces(text, pos);
  expect(pos < text.size() && text[pos] == ch, "unexpected JSON character");
  ++pos;
}

std::string parseString(const std::string& text, size_t& pos) {
  skipSpaces(text, pos);
  expect(pos < text.size() && text[pos] == '"', "expected JSON string");
  ++pos;

  std::string value;
  while (pos < text.size() && text[pos] != '"') {
    expect(text[pos] != '\\', "escaped JSON strings are not supported yet");
    value.push_back(text[pos]);
    ++pos;
  }

  expect(pos < text.size(), "unterminated JSON string");
  ++pos;
  return value;
}

int64_t parseInteger(const std::string& text, size_t& pos) {
  skipSpaces(text, pos);
  expect(pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos])),
         "expected JSON integer");

  int64_t value = 0;
  while (pos < text.size() &&
         std::isdigit(static_cast<unsigned char>(text[pos]))) {
    value = value * 10 + (text[pos] - '0');
    ++pos;
  }
  return value;
}

std::vector<int64_t> parseIntegerArray(const std::string& text, size_t& pos) {
  std::vector<int64_t> values;
  expectChar(text, pos, '[');
  skipSpaces(text, pos);
  if (pos < text.size() && text[pos] == ']') {
    ++pos;
    return values;
  }

  for (;;) {
    values.push_back(parseInteger(text, pos));
    skipSpaces(text, pos);
    if (pos < text.size() && text[pos] == ']') {
      ++pos;
      return values;
    }
    expectChar(text, pos, ',');
  }
}

void skipValue(const std::string& text, size_t& pos) {
  skipSpaces(text, pos);
  expect(pos < text.size(), "unexpected end of JSON value");

  if (text[pos] == '"') {
    parseString(text, pos);
    return;
  }

  if (text[pos] == '[') {
    int depth = 0;
    do {
      if (text[pos] == '"') {
        parseString(text, pos);
        continue;
      }
      if (text[pos] == '[')
        ++depth;
      if (text[pos] == ']')
        --depth;
      ++pos;
      expect(pos <= text.size(), "unterminated JSON array");
    } while (depth > 0);
    return;
  }

  if (text[pos] == '{') {
    int depth = 0;
    do {
      if (text[pos] == '"') {
        parseString(text, pos);
        continue;
      }
      if (text[pos] == '{')
        ++depth;
      if (text[pos] == '}')
        --depth;
      ++pos;
      expect(pos <= text.size(), "unterminated JSON object");
    } while (depth > 0);
    return;
  }

  while (pos < text.size() && text[pos] != ',' && text[pos] != '}' &&
         text[pos] != ']')
    ++pos;
}

TensorInfo parseTensorObject(const std::string& header, size_t& pos) {
  TensorInfo info;
  bool seenDtype = false;
  bool seenShape = false;
  bool seenOffsets = false;

  expectChar(header, pos, '{');
  skipSpaces(header, pos);
  if (pos < header.size() && header[pos] == '}')
    fail("empty tensor metadata");

  for (;;) {
    auto field = parseString(header, pos);
    expectChar(header, pos, ':');

    if (field == "dtype") {
      auto dtype = parseString(header, pos);
      expect(dtype == "F32", "only F32 safetensors are supported");
      seenDtype = true;
    } else if (field == "shape") {
      info.shape = parseIntegerArray(header, pos);
      seenShape = true;
    } else if (field == "data_offsets") {
      auto offsets = parseIntegerArray(header, pos);
      expect(offsets.size() == 2, "data_offsets must contain two integers");
      info.begin = offsets[0];
      info.end = offsets[1];
      expect(info.begin >= 0 && info.end >= info.begin,
             "invalid data_offsets");
      seenOffsets = true;
    } else {
      skipValue(header, pos);
    }

    skipSpaces(header, pos);
    if (pos < header.size() && header[pos] == '}') {
      ++pos;
      break;
    }
    expectChar(header, pos, ',');
  }

  expect(seenDtype && seenShape && seenOffsets, "incomplete tensor metadata");
  return info;
}

std::string readHeader(FILE* file, uint64_t& headerLength) {
  expect(file != nullptr, "file is null");
  expect(std::fseek(file, 0, SEEK_SET) == 0, "failed to seek safetensors file");

  unsigned char lengthBytes[8] = {};
  expect(std::fread(lengthBytes, 1, sizeof(lengthBytes), file) ==
             sizeof(lengthBytes),
         "failed to read safetensors header length");

  headerLength = 0;
  for (int i = 7; i >= 0; --i)
    headerLength = (headerLength << 8) | lengthBytes[i];
  expect(headerLength <= 64 * 1024 * 1024, "safetensors header is too large");

  std::string header(headerLength, '\0');
  expect(std::fread(header.data(), 1, header.size(), file) == header.size(),
         "failed to read safetensors header");
  return header;
}

TensorInfo findTensor(FILE* file, String name) {
  expect(name.data != nullptr, "tensor name is null");
  uint64_t headerLength = 0;
  auto header = readHeader(file, headerLength);

  size_t pos = 0;
  expectChar(header, pos, '{');
  skipSpaces(header, pos);
  while (pos < header.size() && header[pos] != '}') {
    auto key = parseString(header, pos);
    expectChar(header, pos, ':');
    if (key.size() == name.length &&
        std::memcmp(key.data(), name.data, name.length) == 0) {
      auto info = parseTensorObject(header, pos);
      info.headerLength = headerLength;
      return info;
    }

    skipValue(header, pos);
    skipSpaces(header, pos);
    if (pos < header.size() && header[pos] == ',') {
      ++pos;
      continue;
    }
  }

  fail("tensor name not found");
}

} // namespace

struct TensorFloat32 {
  uint8_t* data;
  uint64_t rank;
  uint64_t numel;
  ListUInt64 sizes;
  ListUInt64 strides;
};

extern "C" TensorFloat32 mulberry_safetensor_read_tensor_f32(FILE* file,
                                                             String name) {
  auto info = findTensor(file, name);
  expect(!info.shape.empty(), "tensor rank must be non-zero");

  auto payloadOffset = static_cast<long>(8 + info.headerLength + info.begin);
  expect(std::fseek(file, payloadOffset, SEEK_SET) == 0,
         "failed to seek tensor payload");

  auto numel = static_cast<uint64_t>(1);
  for (auto dim : info.shape) {
    expect(dim >= 0, "negative runtime tensor dimension");
    numel *= static_cast<uint64_t>(dim);
  }
  auto bytes = numel * static_cast<uint64_t>(sizeof(float));
  expect(info.end - info.begin == static_cast<int64_t>(bytes),
         "tensor byte size mismatch");

  auto* data = static_cast<uint8_t*>(std::malloc(bytes));
  expect(data != nullptr, "failed to allocate tensor payload");
  expect(std::fread(data, 1, bytes, file) == static_cast<size_t>(bytes),
         "failed to read tensor payload");

  auto* sizes = static_cast<uint64_t*>(
      std::malloc(info.shape.size() * sizeof(uint64_t)));
  expect(sizes != nullptr, "failed to allocate tensor sizes");
  auto* strides = static_cast<uint64_t*>(
      std::malloc(info.shape.size() * sizeof(uint64_t)));
  expect(strides != nullptr, "failed to allocate tensor strides");

  uint64_t stride = 1;
  for (size_t i = info.shape.size(); i > 0; --i) {
    auto index = i - 1;
    sizes[index] = static_cast<uint64_t>(info.shape[index]);
    strides[index] = stride;
    stride *= sizes[index];
  }

  auto length = static_cast<uint64_t>(info.shape.size());
  return TensorFloat32{
      data,
      length,
      numel,
      ListUInt64{length, length, sizes},
      ListUInt64{length, length, strides},
  };
}

// MLIR C-interface lowering passes aggregate returns through an explicit
// result pointer. Keep the runtime wrapper in that ABI so JIT symbol
// resolution matches the lowered call sites.
extern "C" void _mlir_ciface_mulberry_safetensor_read_tensor_f32(
    TensorFloat32* out, FILE* file, String name) {
  expect(out != nullptr, "tensor result slot is null");
  *out = mulberry_safetensor_read_tensor_f32(file, name);
}
