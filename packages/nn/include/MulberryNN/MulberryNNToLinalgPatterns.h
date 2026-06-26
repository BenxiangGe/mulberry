//===--- MulberryNNToLinalgPatterns.h --------------------------*- C++ -*-===//

#ifndef MULBERRY_NN_TO_LINALG_PATTERNS_H
#define MULBERRY_NN_TO_LINALG_PATTERNS_H

namespace mlir {
class RewritePatternSet;
class TypeConverter;

namespace mulberry_nn {

auto populateMulberryNNToLinalgPatterns(const TypeConverter& typeConverter,
                                        RewritePatternSet& patterns) -> void;

} // namespace mulberry_nn
} // namespace mlir

#endif // MULBERRY_NN_TO_LINALG_PATTERNS_H

