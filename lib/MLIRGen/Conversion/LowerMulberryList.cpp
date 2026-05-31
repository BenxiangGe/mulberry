//===--- LowerMulberryList.cpp - Lower Mulberry list ops -----------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/Basic/Logging.h"
#include "cherry/MLIRGen/Conversion/CherryPasses.h"
#include "cherry/MLIRGen/Conversion/TensorDescriptorLowering.h"
#include "cherry/MLIRGen/IR/MulberryOps.h"
#include "cherry/MLIRGen/IR/MulberryTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/DialectConversion.h"

#include <map>
#include <vector>

#define DEBUG_TYPE "LowerMulberryList"

namespace mlir::cherry {

#define GEN_PASS_DEF_LOWERMULBERRYLIST
#include "cherry/MLIRGen/Conversion/CherryPasses.h.inc"

namespace {

auto getListCreate(Value list) -> mulberry::ListCreateOp {
  if (!list)
    return {};
  return list.getDefiningOp<mulberry::ListCreateOp>();
}

auto createLengthStorage(Location location, size_t length,
                         PatternRewriter& rewriter) -> Value {
  auto i64Type = rewriter.getI64Type();
  auto storageType = MemRefType::get({1}, i64Type);
  auto storage = memref::AllocOp::create(rewriter, location, storageType);
  auto index = arith::ConstantIndexOp::create(rewriter, location, 0);
  auto value = arith::ConstantIntOp::create(rewriter, location, length, 64);
  memref::StoreOp::create(rewriter, location, value, storage.getResult(),
                          ValueRange{index});
  return storage.getResult();
}

auto loadListLength(Location location, Value lengthStorage,
                    PatternRewriter& rewriter) -> Value {
  auto index = arith::ConstantIndexOp::create(rewriter, location, 0);
  auto value = memref::LoadOp::create(rewriter, location, lengthStorage,
                                      ValueRange{index});
  return value.getResult();
}

struct ListDescriptorStorage {
  Value length;
  Value data;
};

auto createListDescriptorStorage(Location location, size_t length,
                                 MemRefType dataType,
                                 PatternRewriter& rewriter)
    -> ListDescriptorStorage {
  // Lowering-only mirror of the intended runtime List<T> descriptor:
  // {length, data}. This keeps the current static-list rewrite close to the
  // final ABI shape without inventing a first-class MLIR aggregate value yet.
  return {
      createLengthStorage(location, length, rewriter),
      memref::AllocOp::create(rewriter, location, dataType).getResult(),
  };
}

class StaticValueListStorage {
public:
  auto load(mulberry::ListGetOp op, mulberry::ListCreateOp createOp,
            PatternRewriter& rewriter) -> FailureOr<Value> {
    auto elementType = op.getResult().getType();
    auto descriptor = getOrCreateDescriptor(createOp, elementType, rewriter);
    if (failed(descriptor))
      return failure();

    auto value = memref::LoadOp::create(rewriter, op.getLoc(),
                                        descriptor->data, op.getIndex());
    return value.getResult();
  }

  auto loadLength(mulberry::ListSizeOp op, mulberry::ListCreateOp createOp,
                  PatternRewriter& rewriter) -> FailureOr<Value> {
    auto listType =
        llvm::cast<mulberry::ListType>(createOp.getResult().getType());
    auto descriptor = getOrCreateDescriptor(createOp, listType.getElementType(),
                                            rewriter);
    if (failed(descriptor))
      return failure();

    return loadListLength(op.getLoc(), descriptor->length, rewriter);
  }

private:
  struct ListStorage {
    Type elementType;
    ListDescriptorStorage descriptor;
  };

  auto getStorage(mulberry::ListCreateOp createOp, Type elementType)
      -> ListStorage * {
    auto elements = createOp.getElements();
    if (elements.empty())
      return nullptr;
    if (!BaseMemRefType::isValidElementType(elementType))
      return nullptr;

    auto it = _storageByCreateOp.find(createOp.getOperation());
    if (it != _storageByCreateOp.end()) {
      if (it->second.elementType != elementType)
        return nullptr;
      return &it->second;
    }

    _storageByCreateOp[createOp.getOperation()] =
        ListStorage{elementType, {}};
    return &_storageByCreateOp[createOp.getOperation()];
  }

  auto getOrCreateDescriptor(mulberry::ListCreateOp createOp, Type elementType,
                             PatternRewriter& rewriter)
      -> FailureOr<ListDescriptorStorage> {
    auto *storage = getStorage(createOp, elementType);
    if (!storage)
      return failure();
    if (storage->descriptor.length && storage->descriptor.data)
      return storage->descriptor;

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(createOp);
    auto elements = createOp.getElements();
    auto storageType = MemRefType::get(
        {static_cast<int64_t>(elements.size())}, elementType);
    auto descriptor = createListDescriptorStorage(createOp.getLoc(),
                                                 elements.size(),
                                                 storageType, rewriter);

    for (const auto& element : llvm::enumerate(elements)) {
      auto index =
          arith::ConstantIndexOp::create(rewriter, createOp.getLoc(),
                                         element.index());
      memref::StoreOp::create(rewriter, createOp.getLoc(), element.value(),
                              descriptor.data, ValueRange{index});
    }

    storage->descriptor = descriptor;
    return storage->descriptor;
  }

  std::map<Operation *, ListStorage> _storageByCreateOp;
};

class StaticTensorListStorage {
public:
  auto loadLength(mulberry::ListSizeOp op, mulberry::ListCreateOp createOp,
                  PatternRewriter& rewriter) -> FailureOr<Value> {
    auto memRefType = getElementMemRefType(createOp);
    auto *storage = getStorage(createOp, memRefType);
    if (!storage)
      return failure();

    auto length = getOrCreateLength(createOp, *storage, rewriter);
    return loadListLength(op.getLoc(), length, rewriter);
  }

  auto load(mulberry::TensorUnpackOp op, mulberry::ListGetOp listGetOp,
            mulberry::ListCreateOp createOp, PatternRewriter& rewriter)
      -> FailureOr<Value> {
    auto resultType = llvm::cast<MemRefType>(op.getResult().getType());
    auto storage = getOrCreate(createOp, resultType, rewriter);
    if (failed(storage))
      return failure();

    return loadTensorFromDescriptorTables(op.getLoc(), resultType,
                                          storage->tables,
                                          listGetOp.getIndex(), rewriter);
  }

private:
  struct TensorListStorage {
    MemRefType tensorType;
    Value length;
    TensorDescriptorTables tables;
  };

  auto getElementMemRefType(mulberry::ListCreateOp createOp) -> MemRefType {
    auto listType =
        llvm::cast<mulberry::ListType>(createOp.getResult().getType());
    auto descriptorType =
        llvm::dyn_cast<mulberry::TensorDescriptorType>(
            listType.getElementType());
    if (!descriptorType)
      return {};
    return descriptorType.getMemrefType();
  }

  auto getStorage(mulberry::ListCreateOp createOp, MemRefType resultType)
      -> TensorListStorage * {
    if (!resultType)
      return nullptr;

    auto it = _storageByCreateOp.find(createOp.getOperation());
    if (it != _storageByCreateOp.end()) {
      if (it->second.tensorType != resultType)
        return nullptr;
      return &it->second;
    }

    _storageByCreateOp[createOp.getOperation()] =
        TensorListStorage{resultType, {}, {}};
    return &_storageByCreateOp[createOp.getOperation()];
  }

  auto getOrCreateLength(mulberry::ListCreateOp createOp,
                         TensorListStorage& storage,
                         PatternRewriter& rewriter) -> Value {
    if (storage.length)
      return storage.length;

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(createOp);
    storage.length = createLengthStorage(createOp.getLoc(),
                                         createOp.getElements().size(),
                                         rewriter);
    return storage.length;
  }

  auto getOrCreate(mulberry::ListCreateOp createOp, MemRefType resultType,
                   PatternRewriter& rewriter) -> FailureOr<TensorListStorage> {
    auto *storage = getStorage(createOp, resultType);
    if (!storage)
      return failure();
    if (storage->tables.baseBuffers)
      return *storage;

    // Medium-term List<Tensor> lowering: store tensor descriptor metadata
    // instead of raw ranked memrefs. The storage is still static-list
    // lowering, but its shape mirrors the intended runtime list descriptor:
    // {length, tensor descriptor tables}. Each descriptor table stores one
    // memref descriptor field for every list element.
    // See: https://mlir.llvm.org/docs/Dialects/MemRef/
    //
    // Materialize once after the list literal and cache it for all list[i]
    // users. Otherwise multiple dynamic indexes of the same list would rebuild
    // identical storage.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(createOp);
    auto descriptors = createOp.getElements();
    if (descriptors.empty())
      return failure();
    if (!storage->length)
      storage->length = createLengthStorage(createOp.getLoc(),
                                            descriptors.size(), rewriter);
    storage->tables = createTensorDescriptorTables(
        createOp.getLoc(), descriptors.size(), resultType, rewriter);

    for (const auto& descriptor : llvm::enumerate(descriptors)) {
      auto descriptorPackOp =
          descriptor.value().getDefiningOp<mulberry::TensorPackOp>();
      if (!descriptorPackOp)
        return failure();

      auto tensor = descriptorPackOp.getTensor();
      if (tensor.getType() != resultType)
        return failure();

      storeTensorDescriptor(createOp.getLoc(), descriptor.index(), tensor,
                            storage->tables, rewriter);
    }

    return *storage;
  }

  std::map<Operation *, TensorListStorage> _storageByCreateOp;
};

class ListSizeOpLowering : public OpRewritePattern<mulberry::ListSizeOp> {
public:
  ListSizeOpLowering(MLIRContext *context,
                     StaticValueListStorage& staticValueListStorage,
                     StaticTensorListStorage& staticTensorListStorage)
      : OpRewritePattern<mulberry::ListSizeOp>(context),
        _staticValueListStorage(staticValueListStorage),
        _staticTensorListStorage(staticTensorListStorage) {}

  auto matchAndRewrite(mulberry::ListSizeOp op,
                       PatternRewriter& rewriter) const
      -> LogicalResult final {
    auto createOp = getListCreate(op.getList());
    if (!createOp)
      return rewriter.notifyMatchFailure(
          op, "list.size currently requires list.create source");

    auto listType =
        llvm::cast<mulberry::ListType>(createOp.getResult().getType());
    if (llvm::isa<mulberry::TensorDescriptorType>(listType.getElementType())) {
      auto size = _staticTensorListStorage.loadLength(op, createOp, rewriter);
      if (failed(size))
        return rewriter.notifyMatchFailure(
            op, "cannot materialize static tensor list descriptor storage");

      rewriter.replaceOp(op, *size);
      return success();
    }

    auto size = _staticValueListStorage.loadLength(op, createOp, rewriter);
    if (failed(size))
      return rewriter.notifyMatchFailure(
          op, "cannot materialize static list descriptor storage");

    rewriter.replaceOp(op, *size);
    return success();
  }

private:
  StaticValueListStorage& _staticValueListStorage;
  StaticTensorListStorage& _staticTensorListStorage;
};

class TensorUnpackOpLowering
    : public OpRewritePattern<mulberry::TensorUnpackOp> {
public:
  TensorUnpackOpLowering(MLIRContext *context,
                         StaticTensorListStorage& staticTensorListStorage)
      : OpRewritePattern<mulberry::TensorUnpackOp>(context),
        _staticTensorListStorage(staticTensorListStorage) {}

  auto matchAndRewrite(mulberry::TensorUnpackOp op,
                       PatternRewriter& rewriter) const
      -> LogicalResult final {
    auto listGetOp = op.getTensor().getDefiningOp<mulberry::ListGetOp>();
    if (!listGetOp)
      return rewriter.notifyMatchFailure(
          op, "tensor.unpack of list value currently requires list.get source");

    auto createOp = getListCreate(listGetOp.getList());
    if (!createOp)
      return rewriter.notifyMatchFailure(
          op, "tensor.unpack of list.get currently requires list.create");

    auto descriptors = createOp.getElements();
    if (descriptors.empty())
      return rewriter.notifyMatchFailure(op, "cannot lower empty list.get");

    auto tensor =
        _staticTensorListStorage.load(op, listGetOp, createOp, rewriter);
    if (failed(tensor))
      return rewriter.notifyMatchFailure(
          op, "cannot materialize static tensor list storage");

    rewriter.replaceOp(op, *tensor);
    return success();
  }

private:
  StaticTensorListStorage& _staticTensorListStorage;
};

class ListGetOpLowering : public OpRewritePattern<mulberry::ListGetOp> {
public:
  ListGetOpLowering(MLIRContext *context,
                    StaticValueListStorage& staticValueListStorage,
                    StaticTensorListStorage& staticTensorListStorage)
      : OpRewritePattern<mulberry::ListGetOp>(context),
        _staticValueListStorage(staticValueListStorage),
        _staticTensorListStorage(staticTensorListStorage) {}

  auto matchAndRewrite(mulberry::ListGetOp op, PatternRewriter& rewriter) const
      -> LogicalResult final {
    auto createOp = getListCreate(op.getList());
    if (!createOp)
      return rewriter.notifyMatchFailure(
          op, "list.get currently requires list.create source");

    if (op->use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    if (llvm::isa<mulberry::TensorDescriptorType>(op.getResult().getType())) {
      std::vector<mulberry::TensorUnpackOp> unpackUsers;
      for (auto *user : op->getUsers()) {
        auto unpackOp = llvm::dyn_cast<mulberry::TensorUnpackOp>(user);
        if (!unpackOp)
          return rewriter.notifyMatchFailure(
              op, "tensor list.get currently requires tensor.unpack users");
        unpackUsers.push_back(unpackOp);
      }

      for (auto unpackOp : unpackUsers) {
        auto tensor = _staticTensorListStorage.load(unpackOp, op, createOp,
                                                    rewriter);
        if (failed(tensor))
          return rewriter.notifyMatchFailure(
              op, "cannot materialize static tensor list storage");
        rewriter.replaceOp(unpackOp, *tensor);
      }

      rewriter.eraseOp(op);
      return success();
    }

    auto value = _staticValueListStorage.load(op, createOp, rewriter);
    if (failed(value))
      return rewriter.notifyMatchFailure(
          op, "cannot materialize static list element storage");

    rewriter.replaceOp(op, *value);
    return success();
  }

private:
  StaticValueListStorage& _staticValueListStorage;
  StaticTensorListStorage& _staticTensorListStorage;
};

class ListCreateOpLowering
    : public OpRewritePattern<mulberry::ListCreateOp> {
public:
  using OpRewritePattern<mulberry::ListCreateOp>::OpRewritePattern;

  auto matchAndRewrite(mulberry::ListCreateOp op,
                       PatternRewriter& rewriter) const
      -> LogicalResult final {
    if (!op->use_empty())
      return rewriter.notifyMatchFailure(op, "list.create still has users");

    rewriter.eraseOp(op);
    return success();
  }
};

struct LowerMulberryList
    : public impl::LowerMulberryListBase<LowerMulberryList> {
  using impl::LowerMulberryListBase<LowerMulberryList>::LowerMulberryListBase;

  auto runOnOperation() -> void final {
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect>();
    target.addIllegalOp<mulberry::ListGetOp, mulberry::ListSizeOp,
                        mulberry::TensorUnpackOp>();
    target.addLegalOp<mulberry::ListCreateOp, mulberry::TensorPackOp>();

    StaticValueListStorage staticValueListStorage;
    StaticTensorListStorage staticTensorListStorage;
    RewritePatternSet patterns(&getContext());
    patterns.add<ListSizeOpLowering>(&getContext(), staticValueListStorage,
                                     staticTensorListStorage);
    patterns.add<ListGetOpLowering>(&getContext(), staticValueListStorage,
                                    staticTensorListStorage);
    patterns.add<TensorUnpackOpLowering>(&getContext(),
                                         staticTensorListStorage);

    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPartialConversion(getOperation(), target, patternSet)))
      signalPassFailure();

    RewritePatternSet cleanupPatterns(&getContext());
    cleanupPatterns.add<ListCreateOpLowering>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(),
                                     std::move(cleanupPatterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::cherry
