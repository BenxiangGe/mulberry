//===--- Builtins.h -------------------------------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_BUILTINS_H
#define CHERRY_BUILTINS_H

#include <string>
#include <string_view>

namespace cherry {
namespace builtins {

// Functions
const std::string_view builtinPrint = "builtinPrint";
const std::string_view print = "print";
const std::string_view boolToUInt64 = "boolToUInt64";
const std::string_view sizeOf = "sizeof";
const std::string_view alignOf = "alignof";
const std::string_view zeros = "zeros";
const std::string_view read = "std.io.read";
const std::string_view readTensor = "std.io.readTensor";
const std::string_view tensorPack = "std.tensor.pack";
const std::string_view tensorView = "std.tensor.view";
const std::string_view write = "std.io.write";
const std::string_view ptrAsUInt8 = "std.ptr.asUInt8";

} // end namespace builtins

} // end namespace cherry

#endif // CHERRY_BUILTINS_H
