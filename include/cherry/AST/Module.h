//===--- Module.h - Cherry Language Module AST ------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_MODULE_H
#define CHERRY_MODULE_H

#include "cherry/AST/Node.h"
#include <string>
#include <string_view>

namespace cherry {
class Decl;

class Module : public Node {
public:
  explicit Module(llvm::SMLoc location, VectorUniquePtr<Decl> declarations)
      : Node{location}, _declarations{std::move(declarations)} {};

  auto packageName() const -> std::string_view { return _packageName; }

  auto setPackageName(std::string_view packageName) -> void {
    _packageName = packageName;
  }

  auto declarations() const -> const VectorUniquePtr<Decl> & {
    return _declarations;
  }

  auto takeDeclarations() -> VectorUniquePtr<Decl> {
    return std::move(_declarations);
  }

  auto setDeclarations(VectorUniquePtr<Decl> declarations) -> void {
    _declarations = std::move(declarations);
  }

private:
  std::string _packageName;
  VectorUniquePtr<Decl> _declarations;

public:
  auto begin() const -> decltype(_declarations.begin()) {
    return _declarations.begin();
  }
  auto end() const -> decltype(_declarations.end()) {
    return _declarations.end();
  }
};

} // end namespace cherry

#endif // CHERRY_MODULE_H
