#ifndef MULBERRY_SCOPESTACK_H
#define MULBERRY_SCOPESTACK_H

#include <string_view>
#include <vector>

namespace mulberry {

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

  auto lookupCurrent(std::string_view name) -> typename Scope::mapped_type * {
    if (_scopes.empty())
      return nullptr;
    auto symbol = _scopes.back().find(name);
    if (symbol == _scopes.back().end())
      return nullptr;
    return &symbol->second;
  }

private:
  std::vector<Scope> _scopes;
};

} // end namespace mulberry

#endif // MULBERRY_SCOPESTACK_H
