//===--- Identifier.h - Mulberry Language Identifier ASTs ---------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_IDENTIFIER_H
#define MULBERRY_IDENTIFIER_H

#include "mulberry/AST/Node.h"
#include <string>
#include <string_view>

namespace mulberry {

class Identifier : public Node {
public:
  explicit Identifier(llvm::SMLoc location, std::string_view name)
      : Node{location}, _name(name){};

  auto name() const -> std::string_view { return _name; }

  auto setName(std::string_view name) -> void { _name = name; }

private:
  std::string _name;
};

} // end namespace mulberry

#endif // MULBERRY_IDENTIFIER_H
