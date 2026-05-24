//===--- Name.h - Cherry Language Name ASTs ---------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_NAME_H
#define CHERRY_NAME_H

#include "cherry/AST/Identifier.h"

namespace cherry {

class Type;

class FunctionName final : public Identifier {
public:
  using Identifier::Identifier;
};

class StructName final : public Identifier {
public:
  using Identifier::Identifier;

  auto setType(const Type *type) -> void { _type = type; }

  auto type() const -> const Type * { return _type; }

private:
  const Type *_type = nullptr;
};

} // end namespace cherry

#endif // CHERRY_NAME_H
