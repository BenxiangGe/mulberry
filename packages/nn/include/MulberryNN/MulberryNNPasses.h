//===--- MulberryNNPasses.h - Mulberry NN package passes -------*- C++ -*-===//

#ifndef MULBERRY_NN_PASSES_H
#define MULBERRY_NN_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace mulberry_nn {
#define GEN_PASS_DECL
#include "MulberryNN/MulberryNNPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "MulberryNN/MulberryNNPasses.h.inc"
} // namespace mulberry_nn
} // namespace mlir

#endif // MULBERRY_NN_PASSES_H

