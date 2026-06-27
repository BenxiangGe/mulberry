//===--- Name.h - Mulberry Language Name ASTs ---------------------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_NAME_H
#define MULBERRY_NAME_H

#include "mulberry/AST/Identifier.h"

namespace mulberry {

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

} // end namespace mulberry

#endif // MULBERRY_NAME_H
