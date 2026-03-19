//===--- ConvertCherryToSCF.cpp -------------------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/Conversion/CherryPasses.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::cherry {

#define GEN_PASS_DEF_CONVERTCHERRYTOSCF
#include "cherry/MLIRGen/Conversion/CherryPasses.h.inc"

namespace {

struct ConvertCherryToSCF
    : public impl::ConvertCherryToSCFBase<ConvertCherryToSCF> {

  using impl::ConvertCherryToSCFBase<
      ConvertCherryToSCF>::ConvertCherryToSCFBase;

  auto runOnOperation() -> void final {
    ConversionTarget target(getContext());
    target.addLegalDialect<cherry::CherryDialect, scf::SCFDialect>();

    RewritePatternSet patterns(&getContext());

    auto f = getOperation();
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPartialConversion(f, target, patternSet)))
      signalPassFailure();
  }
};

} // end namespace
} // namespace mlir::cherry
