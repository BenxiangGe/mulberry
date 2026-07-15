//===--- Builtins.h -------------------------------------------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_BUILTINS_H
#define MULBERRY_BUILTINS_H

#include <string>
#include <string_view>

namespace mulberry {
namespace builtins {

// Functions
const std::string_view sizeOf = "sizeof";
const std::string_view alignOf = "alignof";
const std::string_view typeInfo = "typeInfo";
const std::string_view typeOf = "typeOf";
const std::string_view objectIdentity = "__objectIdentity";

} // end namespace builtins

} // end namespace mulberry

#endif // MULBERRY_BUILTINS_H
