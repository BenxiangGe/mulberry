#ifndef CHERRY_LLVMGEN_TYPECONVERTER_H
#define CHERRY_LLVMGEN_TYPECONVERTER_H

#include "cherry/Basic/Types.h"
#include "llvm/IR/Type.h"

namespace cherry {

class LLVMTypeConverter {
public:
  explicit LLVMTypeConverter(llvm::LLVMContext& context) : _context(context) {}

  auto convert(const BuiltinType& type) const -> llvm::Type*;

private:
  llvm::LLVMContext& _context;
};

} // namespace cherry

#endif // CHERRY_LLVMGEN_TYPECONVERTER_H
