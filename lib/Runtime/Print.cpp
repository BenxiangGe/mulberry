//===--- Print.cpp --------------------------------------------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <string_view>

namespace {

struct String {
  uint64_t length;
  uint8_t* data;
};

extern "C" void* mulberry_boehm_malloc(uint64_t size);

auto makeString(std::string_view value) -> String {
  auto size = static_cast<uint64_t>(value.size());
  auto* data = static_cast<uint8_t*>(mulberry_boehm_malloc(size + 1));
  if (!value.empty())
    std::memcpy(data, value.data(), value.size());
  data[size] = 0;
  return {size, data};
}

template <typename T>
auto formatInteger(T value) -> String {
  char buffer[128];
  auto conversion = std::to_chars(std::begin(buffer), std::end(buffer), value);
  if (conversion.ec != std::errc{})
    std::abort();
  auto length = static_cast<size_t>(conversion.ptr - buffer);
  return makeString(std::string_view(buffer, length));
}

auto formatFloat(float value) -> String {
  char buffer[64];
  auto length = std::snprintf(buffer, sizeof(buffer), "%.9g",
                              static_cast<double>(value));
  if (length < 0 || static_cast<size_t>(length) >= sizeof(buffer))
    std::abort();
  return makeString(
      std::string_view(buffer, static_cast<size_t>(length)));
}

auto formatObjectIdentity(String typeName, const void* object) -> String {
  char addressBuffer[2 * sizeof(uintptr_t)];
  auto address = reinterpret_cast<uintptr_t>(object);
  auto conversion = std::to_chars(std::begin(addressBuffer),
                                  std::end(addressBuffer), address, 16);
  if (conversion.ec != std::errc{})
    std::abort();

  auto addressLength = static_cast<size_t>(conversion.ptr - addressBuffer);
  auto length = typeName.length + 3 + addressLength;
  auto* data = static_cast<uint8_t*>(mulberry_boehm_malloc(length + 1));
  if (typeName.length != 0)
    std::memcpy(data, typeName.data, typeName.length);
  std::memcpy(data + typeName.length, "@0x", 3);
  std::memcpy(data + typeName.length + 3, addressBuffer, addressLength);
  data[length] = 0;
  return {static_cast<uint64_t>(length), data};
}

} // namespace

extern "C" void _mlir_ciface_mulberry_string_from_bool(String* result,
                                                         bool value) {
  *result = makeString(value ? "true" : "false");
}

extern "C" void _mlir_ciface_mulberry_string_from_u8(String* result,
                                                       uint8_t value) {
  *result = formatInteger(static_cast<uint64_t>(value));
}

extern "C" void _mlir_ciface_mulberry_string_from_u64(String* result,
                                                        uint64_t value) {
  *result = formatInteger(value);
}

extern "C" void _mlir_ciface_mulberry_string_from_f32(String* result,
                                                        float value) {
  *result = formatFloat(value);
}

extern "C" void _mlir_ciface_mulberry_string_object_identity(
    String* result, String typeName, uint8_t* object) {
  *result = formatObjectIdentity(typeName, object);
}

extern "C" void _mlir_ciface_writeString(String value) {
  std::fwrite(value.data, 1, value.length, stdout);
}
