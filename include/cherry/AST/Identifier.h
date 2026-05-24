//===--- Identifier.h - Cherry Language Identifier ASTs ---------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_IDENTIFIER_H
#define CHERRY_IDENTIFIER_H

#include "cherry/AST/Node.h"
#include <string>
#include <string_view>

namespace cherry {

class Identifier : public Node {
public:
  explicit Identifier(llvm::SMLoc location, std::string_view name)
      : Node{location}, _name(name){};

  auto name() const -> std::string_view { return _name; }

private:
  std::string _name;
};

} // end namespace cherry

#endif // CHERRY_IDENTIFIER_H
