//===--- LowerCherryBridgeCasts.cpp --------------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/Conversion/CherryPasses.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"

#include <vector>

namespace mlir::cherry {

#define GEN_PASS_DEF_LOWERCHERRYBRIDGECASTS
#include "cherry/MLIRGen/Conversion/CherryPasses.h.inc"

namespace {

auto getPrintfType(MLIRContext *context) -> LLVM::LLVMFunctionType {
  auto llvmI32Ty = IntegerType::get(context, 32);
  auto llvmPtrTy = LLVM::LLVMPointerType::get(context);
  return LLVM::LLVMFunctionType::get(llvmI32Ty, llvmPtrTy,
                                     /*isVarArg=*/true);
}

auto getOrInsertPrintf(OpBuilder &builder, ModuleOp module)
    -> FlatSymbolRefAttr {
  auto *context = module.getContext();
  if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
    return SymbolRefAttr::get(context, "printf");

  OpBuilder::InsertionGuard insertGuard(builder);
  builder.setInsertionPointToStart(module.getBody());
  LLVM::LLVMFuncOp::create(builder, module.getLoc(), "printf",
                           getPrintfType(context));
  return SymbolRefAttr::get(context, "printf");
}

auto getOrCreateGlobalString(Location loc, OpBuilder &builder, StringRef name,
                             StringRef value, ModuleOp module) -> Value {
  LLVM::GlobalOp global;
  if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto type = LLVM::LLVMArrayType::get(IntegerType::get(builder.getContext(),
                                                          8),
                                         value.size());
    global = LLVM::GlobalOp::create(builder, loc, type,
                                    /*isConstant=*/true,
                                    LLVM::Linkage::Internal, name,
                                    builder.getStringAttr(value));
  }

  auto globalPtr = LLVM::AddressOfOp::create(builder, loc, global);
  auto cst0 = LLVM::ConstantOp::create(
      builder, loc, IntegerType::get(builder.getContext(), 64),
      builder.getIntegerAttr(builder.getIndexType(), 0));
  return LLVM::GEPOp::create(
      builder, loc, LLVM::LLVMPointerType::get(builder.getContext()),
      global.getType(), globalPtr, ArrayRef<Value>({cst0, cst0}));
}

struct LowerCherryBridgeCasts
    : public impl::LowerCherryBridgeCastsBase<LowerCherryBridgeCasts> {
  using impl::LowerCherryBridgeCastsBase<
      LowerCherryBridgeCasts>::LowerCherryBridgeCastsBase;

  auto runOnOperation() -> void final {
    auto module = cast<ModuleOp>(getOperation());
    OpBuilder builder(module.getContext());

    std::vector<PrintOp> prints;
    module.walk([&](PrintOp op) { prints.push_back(op); });

    for (auto op : prints) {
      builder.setInsertionPoint(op);
      auto printfRef = getOrInsertPrintf(builder, module);
      auto formatSpecifier = getOrCreateGlobalString(
          op.getLoc(), builder, "frmt_spec", StringRef("%llu\n\0", 6),
          module);
      if (!LLVM::isCompatibleType(op.getInput().getType())) {
        op.emitError("print input is not LLVM-compatible after CIR lowering");
        signalPassFailure();
        return;
      }
      LLVM::CallOp::create(builder, op.getLoc(),
                           getPrintfType(module.getContext()), printfRef,
                           ArrayRef<Value>({formatSpecifier, op.getInput()}));
      op.erase();
    }

    std::vector<BridgeCastOp> bridgeCasts;
    module.walk([&](BridgeCastOp op) { bridgeCasts.push_back(op); });

    for (auto op : bridgeCasts) {
      auto input = op.getInput();
      if (auto unrealizedCast =
              input.getDefiningOp<UnrealizedConversionCastOp>()) {
        auto inputs = unrealizedCast.getInputs();
        if (inputs.size() == 1 &&
            inputs.front().getType() == op.getResult().getType()) {
          op.replaceAllUsesWith(inputs.front());
          op.erase();
          continue;
        }
      }

      auto users = llvm::to_vector(op.getResult().getUsers());
      if (users.size() == 1) {
        if (auto unrealizedCast =
                dyn_cast<UnrealizedConversionCastOp>(users.front())) {
          if (unrealizedCast.getInputs().size() == 1 &&
              unrealizedCast.getInputs().front() == op.getResult() &&
              unrealizedCast.getNumResults() == 1 &&
              unrealizedCast.getResult(0).getType() == input.getType()) {
            unrealizedCast.getResult(0).replaceAllUsesWith(input);
            unrealizedCast.erase();
            op.erase();
            continue;
          }
        }
      }

      // CIR lowering may leave a temporary unrealized cast before a bridge
      // cast. Do not replace an LLVM-compatible result with the pre-lowering
      // CIR input; that would make later LLVM ops ill-typed.
      if (input.getType() != op.getResult().getType()) {
        op.emitError("cannot erase bridge cast before source/result match");
        signalPassFailure();
        return;
      }

      op.replaceAllUsesWith(input);
      op.erase();
    }
  }
};

} // namespace
} // namespace mlir::cherry
