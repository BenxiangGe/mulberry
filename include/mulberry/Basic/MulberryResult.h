//===--- MulberryResult.h - Mulberry subclass of LogicalResult ---*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_MULBERRYRESULT_H
#define MULBERRY_MULBERRYRESULT_H

#include "llvm/Support/LogicalResult.h"

namespace mulberry {

class MulberryResult : public llvm::LogicalResult {
public:
  MulberryResult(LogicalResult result = llvm::success())
      : llvm::LogicalResult(result) {}

  explicit operator bool() const { return llvm::failed(*this); }
};

} // end namespace mulberry

#endif // MULBERRY_MULBERRYRESULT_H
