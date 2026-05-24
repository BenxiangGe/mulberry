#include "cherry/LLVMGen/TypeConverter.h"

#include "llvm/IR/DerivedTypes.h"

namespace cherry {

auto LLVMTypeConverter::convert(const BuiltinType& type) const -> llvm::Type* {
  switch (type.builtinKind()) {
  case BuiltinTypeKind::Unit:
    return llvm::Type::getVoidTy(_context);
  case BuiltinTypeKind::UInt64:
    return llvm::Type::getInt64Ty(_context);
  case BuiltinTypeKind::Float32:
    return llvm::Type::getFloatTy(_context);
  case BuiltinTypeKind::Bool:
    return llvm::Type::getInt1Ty(_context);
  }
}

} // namespace cherry
