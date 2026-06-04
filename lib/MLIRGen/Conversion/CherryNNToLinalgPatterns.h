//===--- CherryNNToLinalgPatterns.h -----------------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_NN_TO_LINALG_PATTERNS_H
#define CHERRY_NN_TO_LINALG_PATTERNS_H

namespace mlir {
class RewritePatternSet;
class TypeConverter;

namespace cherry {

auto populateCherryNNToLinalgPatterns(const TypeConverter& typeConverter,
                                      RewritePatternSet& patterns) -> void;

} // namespace cherry
} // namespace mlir

#endif // CHERRY_NN_TO_LINALG_PATTERNS_H
