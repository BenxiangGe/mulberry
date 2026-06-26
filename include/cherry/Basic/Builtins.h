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
const std::string_view sizeOf = "sizeof";
const std::string_view alignOf = "alignof";
const std::string_view zeros = "zeros";

} // end namespace builtins

} // end namespace cherry

#endif // CHERRY_BUILTINS_H
