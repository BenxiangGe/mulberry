//===--- Builtins.h -------------------------------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_BUILTINS_H
#define CHERRY_BUILTINS_H

#include <string_view>

namespace cherry {
namespace builtins {

// Functions
const std::string_view print = "print";
const std::string_view boolToUInt64 = "boolToUInt64";
const std::string_view size = "size";
const std::string_view open = "open";
const std::string_view read = "read";
const std::string_view write = "write";
const std::string_view close = "close";

} // end namespace builtins

namespace nn {
const std::string_view matmul = "matmul";
const std::string_view matadd = "matadd";
const std::string_view matsub = "matsub";
const std::string_view hadamard = "hadamard";
const std::string_view matscale = "matscale";
const std::string_view transpose = "transpose";
const std::string_view exp = "exp";
const std::string_view sigmoid = "sigmoid";
const std::string_view sigmoidPrime = "sigmoidPrime";
const std::string_view argmax = "argmax";
} // end namespace nn

} // end namespace cherry

#endif // CHERRY_BUILTINS_H
