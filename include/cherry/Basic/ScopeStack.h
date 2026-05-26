#ifndef CHERRY_SCOPESTACK_H
#define CHERRY_SCOPESTACK_H

#include <string_view>
#include <vector>

namespace cherry {

template <typename Scope>
class ScopeStack {
public:
  auto reset() -> void { _scopes.clear(); }

  auto enterScope() -> void { _scopes.emplace_back(); }

  auto leaveScope() -> void { _scopes.pop_back(); }

  auto currentScope() -> Scope& { return _scopes.back(); }

  auto empty() const -> bool { return _scopes.empty(); }

  auto lookup(std::string_view name) -> typename Scope::mapped_type * {
    for (auto scope = _scopes.rbegin(); scope != _scopes.rend(); ++scope) {
      auto symbol = scope->find(name);
      if (symbol != scope->end())
        return &symbol->second;
    }
    return nullptr;
  }

private:
  std::vector<Scope> _scopes;
};

} // end namespace cherry

#endif // CHERRY_SCOPESTACK_H
