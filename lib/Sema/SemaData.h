//===--- SemaData.h - Shared semantic-analysis data -----------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_SEMA_DATA_H
#define MULBERRY_SEMA_DATA_H

#include "mulberry/AST/AST.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <string>

namespace mulberry {

using NameSet = std::set<std::string, std::less<>>;

struct TypeSubstitution {
  std::string parameterName;
  const TypeNode *argumentTypeNode = nullptr;
  std::optional<uint64_t> uint64Value;
};

struct ComptimeArgument {
  ComptimeArg::Kind kind = ComptimeArg::Kind::Type;
  const Type *type = nullptr;
  std::unique_ptr<TypeNode> typeNode;
  uint64_t uint64Value = 0;
};

struct InferredComptimeArgument {
  ComptimeParam::Kind kind = ComptimeParam::Kind::Type;
  const Type *type = nullptr;
  std::unique_ptr<TypeNode> typeNode;
  std::optional<uint64_t> uint64Value;

  auto isResolved() const -> bool {
    if (kind == ComptimeParam::Kind::Type)
      return type != nullptr;
    return uint64Value.has_value();
  }
};

} // namespace mulberry

#endif // MULBERRY_SEMA_DATA_H
