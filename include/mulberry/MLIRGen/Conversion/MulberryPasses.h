//===--- MulberryPasses.h - Mulberry passes ---------------------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_PASSES_H
#define MULBERRY_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace mulberry {
#define GEN_PASS_DECL
#include "mulberry/MLIRGen/Conversion/MulberryPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "mulberry/MLIRGen/Conversion/MulberryPasses.h.inc"
} // namespace mulberry
} // namespace mlir

#endif // MULBERRY_PASSES_H
