//===--- Node.h - Mulberry Language Node AST ----------------------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_NODE_H
#define MULBERRY_NODE_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SMLoc.h"

namespace mulberry {

template <typename T>
using VectorUniquePtr = llvm::SmallVector<std::unique_ptr<T>, 2>;

class Node {
public:
  explicit Node(llvm::SMLoc location) : _location{location} {};

  virtual ~Node() = default;

  auto location() const -> const llvm::SMLoc & { return _location; }

private:
  llvm::SMLoc _location;
};

} // namespace mulberry

#endif // MULBERRY_NODE_H
