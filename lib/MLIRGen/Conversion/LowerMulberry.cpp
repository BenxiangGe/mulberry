//===--- LowerMulberry.cpp -----------------------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/Conversion/CherryPasses.h"
#include "CherryNNToLinalgPatterns.h"
#include "cherry/MLIRGen/IR/CherryNNOps.h"
#include "cherry/MLIRGen/IR/MulberryDialect.h"
#include "cherry/MLIRGen/IR/MulberryOps.h"
#include "cherry/MLIRGen/IR/MulberryTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <optional>

namespace mlir::cherry {

#define GEN_PASS_DEF_LOWERMULBERRY
#include "cherry/MLIRGen/Conversion/CherryPasses.h.inc"

namespace {

static auto getPtrType(MLIRContext* context) -> LLVM::LLVMPointerType {
  return LLVM::LLVMPointerType::get(context);
}

static auto getI64Type(MLIRContext* context) -> IntegerType {
  return IntegerType::get(context, 64);
}

static auto isScalarStorageType(Type type) -> bool {
  return type.isIndex() || llvm::isa<IntegerType, FloatType>(type);
}

static auto convertRecordToBackendType(mulberry::RecordType type)
    -> std::optional<Type>;

static auto convertMemRefShape(ArrayRef<int64_t> shape)
    -> std::vector<int64_t> {
  std::vector<int64_t> memrefShape;
  for (auto dim : shape) {
    memrefShape.push_back(dim < 0 ? ShapedType::kDynamic : dim);
  }
  return memrefShape;
}

static auto convertTensorToMemRefType(mulberry::TensorType type)
    -> std::optional<Type> {
  return MemRefType::get(convertMemRefShape(type.getShape()),
                         type.getElementType());
}

// This pass is a transitional storage lowering, not the final
// Mulberry-to-LLVM ABI lowering. Tensor values become memrefs so cherry_nn can
// lower to linalg, while scalar/record stack storage still uses LLVM dialect.
static auto convertToBackendType(Type type) -> std::optional<Type> {
  if (auto recordType = llvm::dyn_cast<mulberry::RecordType>(type))
    return convertRecordToBackendType(recordType);

  if (isScalarStorageType(type))
    return type;

  return std::nullopt;
}

static auto convertAllocaElementType(Type type) -> std::optional<Type> {
  if (auto tensorType = llvm::dyn_cast<mulberry::TensorType>(type))
    return convertTensorToMemRefType(tensorType);

  return convertToBackendType(type);
}

static auto convertListElementType(Type type) -> std::optional<Type> {
  if (isScalarStorageType(type))
    return type;

  // Local List<Tensor> still stores the lowered memref handle so inference
  // loops can lower to linalg. This is not a function-boundary ABI: memref
  // storage cannot use Tensor ABI descriptor structs as element types.
  if (auto tensorType = llvm::dyn_cast<mulberry::TensorType>(type))
    return convertTensorToMemRefType(tensorType);

  return std::nullopt;
}

static auto convertToListStorageType(mulberry::ListType type)
    -> std::optional<mulberry::ListStorageType> {
  auto elementType = convertListElementType(type.getElementType());
  if (!elementType)
    return std::nullopt;

  return mulberry::ListStorageType::get(type.getContext(), *elementType);
}

static auto convertToMemRefType(mulberry::ListStorageType type)
    -> std::optional<MemRefType> {
  auto elementType = type.getElementType();
  // List<Tensor> stores lowered tensor handles. This is still a memref-level
  // storage lowering, not the final Mulberry-to-LLVM ABI representation.
  if (!isScalarStorageType(elementType) && !llvm::isa<MemRefType>(elementType))
    return std::nullopt;

  return MemRefType::get({ShapedType::kDynamic}, elementType);
}

static auto isDescriptorBackedStorage(Type type) -> bool {
  auto storageType = llvm::dyn_cast<mulberry::ListStorageType>(type);
  return storageType &&
         llvm::isa<mulberry::TensorDescType>(storageType.getElementType());
}

using TensorABIDescriptorType = LLVM::LLVMStructType;

// Tensor ABI descriptor layout:
//
//   { data: ptr, sizes: array<rank x i64>, strides: array<rank x i64> }
//
// The MLIRContext used below is only needed to create the MLIR Type objects.
// It is not stored in the descriptor. The descriptor fields are exactly the
// Types pushed into `fields`.
struct TensorABILayout {
  TensorABIDescriptorType descriptorType;
};

static auto convertToTensorABILayout(mulberry::TensorDescType type)
    -> TensorABILayout {
  auto context = type.getContext();
  auto rank = type.getShape().size();
  auto indexArrayType = LLVM::LLVMArrayType::get(getI64Type(context), rank);

  std::vector<Type> fields;
  fields.push_back(getPtrType(context)); // data pointer
  fields.push_back(indexArrayType);      // sizes[rank]
  fields.push_back(indexArrayType);      // strides[rank]
  auto descriptorType = LLVM::LLVMStructType::getLiteral(context, fields);

  return TensorABILayout{descriptorType};
}

using ListABIDescriptorType = LLVM::LLVMStructType;

// List ABI descriptor layout:
//
//   { length: i64, data: ptr }
//
// `length` is the runtime element count. `data` is an opaque backend pointer to
// contiguous element ABI storage; with LLVM opaque pointers, the element ABI
// type is not encoded in the printed pointer type.
struct ListABILayout {
  ListABIDescriptorType descriptorType;
};

static auto convertToListABIElementType(Type type) -> std::optional<Type> {
  if (isScalarStorageType(type))
    return type;

  if (auto tensorDescType = llvm::dyn_cast<mulberry::TensorDescType>(type))
    return convertToTensorABILayout(tensorDescType).descriptorType;

  return std::nullopt;
}

// This helper only describes the backend descriptor type. Function boundaries
// still fail-fast before conversion, because type conversion alone cannot
// materialize list storage into a backend pointer.
static auto convertToListABILayout(mulberry::ListDescType type)
    -> std::optional<ListABILayout> {
  auto elementType = convertToListABIElementType(type.getElementType());
  if (!elementType)
    return std::nullopt;

  auto context = type.getContext();

  std::vector<Type> fields;
  fields.push_back(getI64Type(context)); // runtime length
  fields.push_back(getPtrType(context)); // data pointer
  auto descriptorType = LLVM::LLVMStructType::getLiteral(context, fields);

  return ListABILayout{descriptorType};
}

static auto createMemRefDataPointer(
    Location location, ConversionPatternRewriter& rewriter, Value storage)
    -> Value {
  auto pointerAsIndex = memref::ExtractAlignedPointerAsIndexOp::create(
      rewriter, location, storage);
  auto pointerAsInt = arith::IndexCastOp::create(
      rewriter, location, getI64Type(rewriter.getContext()),
      pointerAsIndex.getResult());
  auto pointer = LLVM::IntToPtrOp::create(
      rewriter, location, getPtrType(rewriter.getContext()),
      ValueRange{pointerAsInt.getResult()});
  return pointer.getResult();
}

// This only builds the backend ABI record value after both fields are already
// legal backend values. It does not decide when List<T> may cross a function
// boundary.
static auto createListABIDesc(
    Location location, ConversionPatternRewriter& rewriter,
    const ListABILayout& layout, Value length, Value dataPointer) -> Value {
  auto listAbiDesc = LLVM::UndefOp::create(rewriter, location,
                                           layout.descriptorType);
  auto withLength = LLVM::InsertValueOp::create(
      rewriter, location, listAbiDesc.getResult(), length,
      ArrayRef<int64_t>{0});
  auto withData = LLVM::InsertValueOp::create(
      rewriter, location, withLength.getResult(), dataPointer,
      ArrayRef<int64_t>{1});
  return withData.getResult();
}

static auto createTensorABIDesc(
    Location location, ConversionPatternRewriter& rewriter,
    const TensorABILayout& layout, Value tensor, mulberry::TensorDescType type)
    -> Value {
  auto desc = LLVM::UndefOp::create(rewriter, location,
                                    layout.descriptorType);
  auto dataPointer = createMemRefDataPointer(location, rewriter, tensor);
  Value current = LLVM::InsertValueOp::create(
      rewriter, location, desc.getResult(), dataPointer,
      ArrayRef<int64_t>{0}).getResult();

  Value stride = arith::ConstantIntOp::create(
      rewriter, location, 1, /*width=*/64).getResult();
  std::vector<Value> sizes;
  for (size_t dim = 0; dim < type.getShape().size(); ++dim) {
    auto index = arith::ConstantIndexOp::create(rewriter, location, dim);
    auto size = memref::DimOp::create(rewriter, location, tensor, index);
    auto sizeI64 = arith::IndexCastOp::create(
        rewriter, location, getI64Type(rewriter.getContext()),
        size.getResult());
    sizes.push_back(sizeI64.getResult());
  }

  for (int64_t dim = static_cast<int64_t>(sizes.size()) - 1; dim >= 0; --dim) {
    // Tensor ABI stores nested LLVM arrays inside the descriptor. InsertValue
    // uses `[field, dim]` to write one element of those array fields.
    current = LLVM::InsertValueOp::create(
        rewriter, location, current, sizes[dim],
        ArrayRef<int64_t>{1, dim}).getResult();
    current = LLVM::InsertValueOp::create(
        rewriter, location, current, stride,
        ArrayRef<int64_t>{2, dim}).getResult();
    if (dim != 0)
      stride = arith::MulIOp::create(
          rewriter, location, stride, sizes[dim]).getResult();
  }

  return current;
}

static auto containsListType(Type type) -> bool {
  if (llvm::isa<mulberry::ListType>(type))
    return true;

  if (auto recordType = llvm::dyn_cast<mulberry::RecordType>(type))
    for (auto field : recordType.getFields())
      if (containsListType(field.type))
        return true;

  return false;
}

static auto containsListType(TypeRange types) -> bool {
  for (auto type : types)
    if (containsListType(type))
      return true;

  return false;
}

static auto containsListStorageType(Type type) -> bool {
  if (llvm::isa<mulberry::ListStorageType>(type))
    return true;

  if (auto recordType = llvm::dyn_cast<mulberry::RecordType>(type))
    for (auto field : recordType.getFields())
      if (containsListStorageType(field.type))
        return true;

  return false;
}

static auto containsListStorageType(TypeRange types) -> bool {
  for (auto type : types)
    if (containsListStorageType(type))
      return true;

  return false;
}

static auto containsListDescType(Type type) -> bool {
  if (llvm::isa<mulberry::ListDescType>(type))
    return true;

  if (auto recordType = llvm::dyn_cast<mulberry::RecordType>(type))
    for (auto field : recordType.getFields())
      if (containsListDescType(field.type))
        return true;

  return false;
}

static auto containsListDescType(TypeRange types) -> bool {
  for (auto type : types)
    if (containsListDescType(type))
      return true;

  return false;
}

static auto containsTensorDescType(Type type) -> bool {
  if (llvm::isa<mulberry::TensorDescType>(type))
    return true;

  if (auto recordType = llvm::dyn_cast<mulberry::RecordType>(type))
    for (auto field : recordType.getFields())
      if (containsTensorDescType(field.type))
        return true;

  return false;
}

static auto containsTensorDescType(TypeRange types) -> bool {
  for (auto type : types)
    if (containsTensorDescType(type))
      return true;

  return false;
}

static auto containsTensorDescType(FunctionType type) -> bool {
  return containsTensorDescType(type.getInputs()) ||
         containsTensorDescType(type.getResults());
}

static auto containsTensorHandleType(Type type) -> bool {
  if (llvm::isa<mulberry::TensorHandleType>(type))
    return true;

  if (auto recordType = llvm::dyn_cast<mulberry::RecordType>(type))
    for (auto field : recordType.getFields())
      if (containsTensorHandleType(field.type))
        return true;

  return false;
}

static auto containsTensorHandleType(TypeRange types) -> bool {
  for (auto type : types)
    if (containsTensorHandleType(type))
      return true;

  return false;
}

static auto containsTensorHandleType(FunctionType type) -> bool {
  return containsTensorHandleType(type.getInputs()) ||
         containsTensorHandleType(type.getResults());
}

static auto containsListBoundaryType(TypeRange types) -> bool {
  return containsListType(types) || containsListStorageType(types) ||
         containsListDescType(types);
}

static auto isTensorDescListDesc(Type type) -> bool {
  auto listDescType = llvm::dyn_cast<mulberry::ListDescType>(type);
  return listDescType &&
         llvm::isa<mulberry::TensorDescType>(listDescType.getElementType());
}

static auto containsUnsupportedListArg(Type type) -> bool {
  if (containsListType(type) || containsListStorageType(type))
    return true;

  if (containsListDescType(type) && !isTensorDescListDesc(type))
    return true;

  return false;
}

static auto containsUnsupportedListArg(TypeRange types) -> bool {
  for (auto type : types)
    if (containsUnsupportedListArg(type))
      return true;

  return false;
}

static auto rejectListBoundaries(Operation* root) -> LogicalResult {
  auto result = success();
  root->walk([&](Operation* op) {
    if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {
      auto funcType = funcOp.getFunctionType();
      if (containsUnsupportedListArg(funcType.getInputs()) ||
          containsListBoundaryType(funcType.getResults())) {
        funcOp.emitError("failed to legalize operation 'func.func': "
                         "List function boundaries are not supported yet");
        result = failure();
        return WalkResult::interrupt();
      }
    }

    if (llvm::isa<func::CallOp>(op)) {
      if (containsUnsupportedListArg(op->getOperandTypes()) ||
          containsListBoundaryType(op->getResultTypes())) {
        op->emitError()
            << "failed to legalize operation '" << op->getName()
            << "': List function boundaries are not supported yet";
        result = failure();
        return WalkResult::interrupt();
      }
    }

    if (llvm::isa<func::ReturnOp>(op)) {
      if (containsListBoundaryType(op->getOperandTypes())) {
        op->emitError()
            << "failed to legalize operation '" << op->getName()
            << "': List function boundaries are not supported yet";
        result = failure();
        return WalkResult::interrupt();
      }
    }

    return WalkResult::advance();
  });

  return result;
}

static auto rejectTensorDescBoundaries(Operation* root) -> LogicalResult {
  auto result = success();
  root->walk([&](Operation* op) {
    if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {
      if (containsTensorDescType(funcOp.getFunctionType())) {
        funcOp.emitError("failed to legalize operation 'func.func': "
                         "Tensor descriptor function boundaries are not "
                         "supported yet");
        result = failure();
        return WalkResult::interrupt();
      }
    }

    if (llvm::isa<func::CallOp, func::ReturnOp>(op)) {
      if (containsTensorDescType(op->getOperandTypes()) ||
          containsTensorDescType(op->getResultTypes())) {
        op->emitError()
            << "failed to legalize operation '" << op->getName()
            << "': Tensor descriptor function boundaries are not supported yet";
        result = failure();
        return WalkResult::interrupt();
      }
    }

    return WalkResult::advance();
  });

  return result;
}

static auto rejectTensorDescUnpack(Operation* root) -> LogicalResult {
  auto result = success();
  root->walk([&](mulberry::TensorDescUnpackOp op) {
    // This is a real boundary reconstruction problem, not a local cast:
    // MLIR cannot rebuild a memref from a raw LLVM pointer without an explicit
    // Tensor handle/runtime ABI design.
    op.emitError("failed to legalize operation 'mulberry.tensor.desc_unpack': "
                 "Tensor descriptor unpack lowering needs explicit Tensor ABI "
                 "reconstruction support");
    result = failure();
    return WalkResult::interrupt();
  });
  return result;
}

static auto rejectTensorHandleOps(Operation* root) -> LogicalResult {
  auto result = success();
  root->walk([&](Operation* op) {
    if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {
      if (containsTensorHandleType(funcOp.getFunctionType())) {
        funcOp.emitError("failed to legalize operation 'func.func': "
                         "Tensor handle function boundaries are not supported "
                         "yet");
        result = failure();
        return WalkResult::interrupt();
      }
    }

    if (llvm::isa<func::CallOp, func::ReturnOp>(op)) {
      if (containsTensorHandleType(op->getOperandTypes()) ||
          containsTensorHandleType(op->getResultTypes())) {
        op->emitError()
            << "failed to legalize operation '" << op->getName()
            << "': Tensor handle function boundaries are not supported yet";
        result = failure();
        return WalkResult::interrupt();
      }
    }

    if (auto fromDesc = llvm::dyn_cast<mulberry::TensorHandleFromDescOp>(op)) {
      // TensorHandle is only a reconstruction-stage marker today. Lowering it
      // needs a real runtime handle or memref reconstruction ABI first.
      fromDesc.emitError(
          "failed to legalize operation 'mulberry.tensor.handle_from_desc': "
          "Tensor handle lowering needs explicit reconstruction support");
      result = failure();
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  return result;
}

static auto containsListDescOp(Operation* op) -> bool {
  auto found = false;
  op->walk([&](Operation* nestedOp) {
    if (llvm::isa<mulberry::ListDescPackOp, mulberry::ListDescLengthOp,
                  mulberry::ListDescDataOp, mulberry::ListDescGetOp,
                  mulberry::ListDescToABIOp, mulberry::ListToDescOp>(
            nestedOp)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static auto convertRecordToBackendType(mulberry::RecordType type)
    -> std::optional<Type> {
  std::vector<Type> fieldTypes;
  for (auto field : type.getFields()) {
    // Keep tensor/list fields illegal until record storage can contain real
    // Mulberry value descriptors instead of backend ABI workaround values.
    auto fieldType = convertToBackendType(field.type);
    if (!fieldType)
      return std::nullopt;
    fieldTypes.push_back(*fieldType);
  }

  return LLVM::LLVMStructType::getLiteral(type.getContext(), fieldTypes);
}

static auto convertPtrType(mulberry::PtrType type) -> std::optional<Type> {
  auto pointeeType = type.getPointeeType();
  if (auto tensorType = llvm::dyn_cast<mulberry::TensorType>(pointeeType)) {
    auto tensorStorageType = convertTensorToMemRefType(tensorType);
    // ptr<Tensor> is a local tensor handle slot. After Tensor lowers to memref,
    // the slot becomes a 0-D memref storing that memref value; this is not a
    // function-boundary pointer bridge.
    if (tensorStorageType)
      return MemRefType::get({}, *tensorStorageType);
  }

  if (convertToBackendType(pointeeType))
    return getPtrType(type.getContext());

  return std::nullopt;
}

// Safety net for Mulberry types that should not fall through to the identity
// conversion after their specific lowering conversion fails.
static auto rejectUnloweredMulberryType(Type type, SmallVectorImpl<Type>&)
    -> std::optional<LogicalResult> {
  if (auto recordType = llvm::dyn_cast<mulberry::RecordType>(type))
    if (!convertRecordToBackendType(recordType))
      return failure();

  if (auto ptrType = llvm::dyn_cast<mulberry::PtrType>(type))
    if (!convertPtrType(ptrType))
      return failure();

  if (llvm::isa<mulberry::ListType>(type))
    return failure();

  if (llvm::isa<mulberry::ListStorageType>(type))
    return failure();

  if (llvm::isa<mulberry::ListDescType>(type))
    return failure();

  if (llvm::isa<mulberry::TensorDescType>(type))
    return failure();

  return std::nullopt;
}

class MulberryTypeConverter : public TypeConverter {
public:
  MulberryTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion(rejectUnloweredMulberryType);
    addConversion([](mulberry::ListType type) -> std::optional<Type> {
      return convertToListStorageType(type);
    });
    addConversion([](mulberry::ListStorageType type) -> std::optional<Type> {
      if (auto memRefType = convertToMemRefType(type))
        return *memRefType;
      if (llvm::isa<mulberry::TensorDescType>(type.getElementType()))
        return getPtrType(type.getContext());
      return std::nullopt;
    });
    addConversion([](mulberry::ListDescType type) -> std::optional<Type> {
      auto layout = convertToListABILayout(type);
      if (!layout)
        return std::nullopt;
      return layout->descriptorType;
    });
    addConversion([](mulberry::TensorDescType type) -> Type {
      return convertToTensorABILayout(type).descriptorType;
    });
    addConversion(convertRecordToBackendType);
    addConversion(convertPtrType);
    addConversion(convertTensorToMemRefType);
    // Keep unsupported Mulberry types illegal until each one has a real
    // lowering. The identity conversion above is only for non-Mulberry types.
  }
};

static auto getRecordType(Type type) -> mulberry::RecordType {
  auto ptrType = llvm::dyn_cast<mulberry::PtrType>(type);
  if (!ptrType)
    return {};

  return llvm::dyn_cast<mulberry::RecordType>(ptrType.getPointeeType());
}

static auto getRecordFieldType(mulberry::RecordType recordType,
                               StringRef field) -> Type {
  return convertToBackendType(recordType.getFieldType(field)).value_or(Type{});
}

static auto getPtrPointeeType(Type type) -> Type {
  auto ptrType = llvm::dyn_cast<mulberry::PtrType>(type);
  if (!ptrType)
    return {};

  return ptrType.getPointeeType();
}

class AllocaOpLowering : public OpConversionPattern<mulberry::AllocaOp> {
public:
  using OpConversionPattern<mulberry::AllocaOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::AllocaOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto elementType = convertAllocaElementType(op.getElementType());
    if (!elementType)
      return rewriter.notifyMatchFailure(
          op, "alloca needs a lowerable storage type");

    if (llvm::isa<MemRefType>(*elementType)) {
      auto slotType = MemRefType::get({}, *elementType);
      auto alloca = memref::AllocaOp::create(rewriter, op.getLoc(), slotType);
      rewriter.replaceOp(op, alloca.getResult());
      return success();
    }

    auto elementCount = arith::ConstantIntOp::create(
        rewriter, op.getLoc(), 1, /*width=*/64);
    auto alloca = LLVM::AllocaOp::create(
        rewriter, op.getLoc(), getPtrType(op.getContext()), *elementType,
        elementCount.getResult(), /*alignment=*/0);
    rewriter.replaceOp(op, alloca.getResult());
    return success();
  }
};

class LoadOpLowering : public OpConversionPattern<mulberry::LoadOp> {
public:
  using OpConversionPattern<mulberry::LoadOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::LoadOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(
          op, "load needs a lowerable result type");

    if (llvm::isa<MemRefType>(resultType)) {
      auto load = memref::LoadOp::create(rewriter, op.getLoc(),
                                         adaptor.getPtr(), ValueRange{});
      rewriter.replaceOp(op, load.getResult());
      return success();
    }

    auto load = LLVM::LoadOp::create(rewriter, op.getLoc(), resultType,
                                     adaptor.getPtr());
    rewriter.replaceOp(op, load.getResult());
    return success();
  }
};

class StoreOpLowering : public OpConversionPattern<mulberry::StoreOp> {
public:
  using OpConversionPattern<mulberry::StoreOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::StoreOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto elementType = convertAllocaElementType(getPtrPointeeType(
        op.getPtr().getType()));
    if (!elementType)
      return rewriter.notifyMatchFailure(
          op, "store target must be a lowered storage slot");

    if (llvm::isa<MemRefType>(*elementType)) {
      memref::StoreOp::create(rewriter, op.getLoc(), adaptor.getValue(),
                              adaptor.getPtr(), ValueRange{});
      rewriter.eraseOp(op);
      return success();
    }

    LLVM::StoreOp::create(rewriter, op.getLoc(), adaptor.getValue(),
                          adaptor.getPtr());
    rewriter.eraseOp(op);
    return success();
  }
};

class ListCreateOpLowering
    : public OpConversionPattern<mulberry::ListCreateOp> {
public:
  using OpConversionPattern<mulberry::ListCreateOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::ListCreateOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto listType =
        llvm::cast<mulberry::ListType>(op.getResult().getType());
    auto storageType = convertToListStorageType(listType);
    if (!storageType)
      return rewriter.notifyMatchFailure(
          op, "list storage needs a lowerable element type");

    auto length = arith::ConstantIndexOp::create(
        rewriter, op.getLoc(), op.getElements().size());
    auto storage = mulberry::ListAllocOp::create(
        rewriter, op.getLoc(), *storageType, length);

    for (auto element : llvm::enumerate(adaptor.getElements())) {
      auto index = arith::ConstantIndexOp::create(
          rewriter, op.getLoc(), element.index());
      mulberry::ListStoreOp::create(rewriter, op.getLoc(), element.value(),
                                    storage, index);
    }

    rewriter.replaceOp(op, storage.getResult());
    return success();
  }
};

class ListAllocOpLowering : public OpConversionPattern<mulberry::ListAllocOp> {
public:
  using OpConversionPattern<mulberry::ListAllocOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::ListAllocOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto storageType = llvm::dyn_cast_or_null<MemRefType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    if (storageType) {
      auto alloc = memref::AllocOp::create(rewriter, op.getLoc(), storageType,
                                           adaptor.getLength());
      rewriter.replaceOp(op, alloc.getResult());
      return success();
    }

    auto listStorageType = llvm::cast<mulberry::ListStorageType>(
        op.getResult().getType());
    auto tensorDescType = llvm::dyn_cast<mulberry::TensorDescType>(
        listStorageType.getElementType());
    if (!tensorDescType)
      return rewriter.notifyMatchFailure(
          op, "list storage needs a lowerable storage element type");
    auto elementType = convertToTensorABILayout(tensorDescType).descriptorType;

    auto length = arith::IndexCastOp::create(
        rewriter, op.getLoc(), getI64Type(op.getContext()),
        adaptor.getLength());
    // Descriptor elements cannot live in memref storage, so descriptor-backed
    // list storage is an explicit pointer to a contiguous LLVM element array.
    auto alloc = LLVM::AllocaOp::create(
        rewriter, op.getLoc(), getPtrType(op.getContext()), elementType,
        length.getResult(), /*alignment=*/0);
    rewriter.replaceOp(op, alloc.getResult());
    return success();
  }
};

class ListGetOpLowering : public OpConversionPattern<mulberry::ListGetOp> {
public:
  using OpConversionPattern<mulberry::ListGetOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::ListGetOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(
          op, "list load needs a lowerable result type");

    auto loaded = mulberry::ListLoadOp::create(
        rewriter, op.getLoc(), resultType, adaptor.getList(),
        adaptor.getIndex());
    rewriter.replaceOp(op, loaded.getResult());
    return success();
  }
};

class ListLoadOpLowering : public OpConversionPattern<mulberry::ListLoadOp> {
public:
  using OpConversionPattern<mulberry::ListLoadOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::ListLoadOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    if (auto tensorDescType = llvm::dyn_cast<mulberry::TensorDescType>(
            op.getResult().getType())) {
      auto resultType = convertToTensorABILayout(tensorDescType).descriptorType;
      auto index = arith::IndexCastOp::create(
          rewriter, op.getLoc(), getI64Type(op.getContext()),
          adaptor.getIndex());
      auto elementPtr = LLVM::GEPOp::create(
          rewriter, op.getLoc(), getPtrType(op.getContext()), resultType,
          adaptor.getStorage(), ArrayRef<Value>{index.getResult()});
      auto load = LLVM::LoadOp::create(rewriter, op.getLoc(), resultType,
                                       elementPtr.getResult());
      rewriter.replaceOp(op, load.getResult());
      return success();
    }

    auto load = memref::LoadOp::create(rewriter, op.getLoc(),
                                       adaptor.getStorage(),
                                       adaptor.getIndex());
    rewriter.replaceOp(op, load.getResult());
    return success();
  }
};

class ListSizeOpLowering : public OpConversionPattern<mulberry::ListSizeOp> {
public:
  using OpConversionPattern<mulberry::ListSizeOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::ListSizeOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto length = mulberry::ListLengthOp::create(
        rewriter, op.getLoc(), rewriter.getIndexType(), adaptor.getList());
    auto result = arith::IndexCastOp::create(
        rewriter, op.getLoc(), op.getResult().getType(), length);
    rewriter.replaceOp(op, result.getResult());
    return success();
  }
};

class ListToDescOpLowering
    : public OpRewritePattern<mulberry::ListToDescOp> {
public:
  using OpRewritePattern<mulberry::ListToDescOp>::OpRewritePattern;

  auto matchAndRewrite(mulberry::ListToDescOp op,
                       PatternRewriter& rewriter) const
      -> LogicalResult final {
    auto createOp = op.getList().getDefiningOp<mulberry::ListCreateOp>();
    if (!createOp)
      return rewriter.notifyMatchFailure(
          op, "list descriptor bridge needs a local list.create");

    auto descType = llvm::cast<mulberry::ListDescType>(
        op.getResult().getType());
    auto tensorDescType = llvm::dyn_cast<mulberry::TensorDescType>(
        descType.getElementType());
    if (!tensorDescType)
      return rewriter.notifyMatchFailure(
          op, "list descriptor bridge only supports TensorDesc elements");

    auto storageType = mulberry::ListStorageType::get(
        op.getContext(), tensorDescType);
    auto length = arith::ConstantIndexOp::create(
        rewriter, op.getLoc(), createOp.getElements().size());
    auto storage = mulberry::ListAllocOp::create(
        rewriter, op.getLoc(), storageType, length);

    // Bridge high-level List<Tensor> to ABI-ready List<TensorDesc> explicitly:
    // each tensor element becomes a Tensor ABI descriptor before entering the
    // pointer-backed list storage used across function arguments.
    for (auto element : llvm::enumerate(createOp.getElements())) {
      auto index = arith::ConstantIndexOp::create(
          rewriter, op.getLoc(), element.index());
      auto desc = mulberry::TensorDescPackOp::create(
          rewriter, op.getLoc(), tensorDescType, element.value());
      mulberry::ListStoreOp::create(rewriter, op.getLoc(), desc.getResult(),
                                    storage, index);
    }

    auto listDesc = mulberry::ListDescPackOp::create(
        rewriter, op.getLoc(), descType, length, storage);
    rewriter.replaceOp(op, listDesc.getResult());
    if (createOp->use_empty())
      rewriter.eraseOp(createOp);
    return success();
  }
};

class ListStoreOpLowering : public OpConversionPattern<mulberry::ListStoreOp> {
public:
  using OpConversionPattern<mulberry::ListStoreOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::ListStoreOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    if (auto tensorDescType = llvm::dyn_cast<mulberry::TensorDescType>(
            op.getValue().getType())) {
      auto valueType = convertToTensorABILayout(tensorDescType).descriptorType;
      auto index = arith::IndexCastOp::create(
          rewriter, op.getLoc(), getI64Type(op.getContext()),
          adaptor.getIndex());
      auto elementPtr = LLVM::GEPOp::create(
          rewriter, op.getLoc(), getPtrType(op.getContext()), valueType,
          adaptor.getStorage(), ArrayRef<Value>{index.getResult()});
      LLVM::StoreOp::create(rewriter, op.getLoc(), adaptor.getValue(),
                            elementPtr.getResult());
      rewriter.eraseOp(op);
      return success();
    }

    memref::StoreOp::create(rewriter, op.getLoc(), adaptor.getValue(),
                            adaptor.getStorage(), adaptor.getIndex());
    rewriter.eraseOp(op);
    return success();
  }
};

class ListLengthOpLowering
    : public OpConversionPattern<mulberry::ListLengthOp> {
public:
  using OpConversionPattern<mulberry::ListLengthOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::ListLengthOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto zero = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0);
    auto length = memref::DimOp::create(rewriter, op.getLoc(),
                                        adaptor.getStorage(), zero);
    rewriter.replaceOp(op, length.getResult());
    return success();
  }
};

class ListDescPackOpLowering
    : public OpRewritePattern<mulberry::ListDescPackOp> {
public:
  using OpRewritePattern<mulberry::ListDescPackOp>::OpRewritePattern;

  auto matchAndRewrite(mulberry::ListDescPackOp op,
                       PatternRewriter& rewriter) const
      -> LogicalResult final {
    if (!op->use_empty())
      return rewriter.notifyMatchFailure(
          op, "list descriptor still has projection users");

    rewriter.eraseOp(op);
    return success();
  }
};

class ListDescLengthOpLowering
    : public OpRewritePattern<mulberry::ListDescLengthOp> {
public:
  using OpRewritePattern<mulberry::ListDescLengthOp>::OpRewritePattern;

  auto matchAndRewrite(mulberry::ListDescLengthOp op,
                       PatternRewriter& rewriter) const
      -> LogicalResult final {
    auto pack = op.getDesc().getDefiningOp<mulberry::ListDescPackOp>();
    if (!pack)
      return rewriter.notifyMatchFailure(
          op, "list descriptor length needs a local desc_pack");

    // Descriptor-backed list storage is lowered to `{length, data}` first, then
    // projected with llvm.extractvalue. Folding it here would bypass that ABI
    // projection path and hide boundary-lowering bugs.
    if (isDescriptorBackedStorage(pack.getData().getType()))
      return rewriter.notifyMatchFailure(
          op, "descriptor-backed list length needs ABI projection");

    rewriter.replaceOp(op, pack.getLength());
    if (pack->use_empty())
      rewriter.eraseOp(pack);
    return success();
  }
};

class ListDescDataOpLowering
    : public OpRewritePattern<mulberry::ListDescDataOp> {
public:
  using OpRewritePattern<mulberry::ListDescDataOp>::OpRewritePattern;

  auto matchAndRewrite(mulberry::ListDescDataOp op,
                       PatternRewriter& rewriter) const
      -> LogicalResult final {
    auto pack = op.getDesc().getDefiningOp<mulberry::ListDescPackOp>();
    if (!pack)
      return rewriter.notifyMatchFailure(
          op, "list descriptor data needs a local desc_pack");

    // Descriptor-backed list storage lowers to a backend pointer, not a memref.
    // Leave it for conversion so desc_data becomes extractvalue field 1.
    if (isDescriptorBackedStorage(pack.getData().getType()))
      return rewriter.notifyMatchFailure(
          op, "descriptor-backed list data needs ABI projection");

    rewriter.replaceOp(op, pack.getData());
    if (pack->use_empty())
      rewriter.eraseOp(pack);
    return success();
  }
};

class ListDescGetOpLowering
    : public OpRewritePattern<mulberry::ListDescGetOp> {
public:
  using OpRewritePattern<mulberry::ListDescGetOp>::OpRewritePattern;

  auto matchAndRewrite(mulberry::ListDescGetOp op,
                       PatternRewriter& rewriter) const
      -> LogicalResult final {
    auto descType = llvm::cast<mulberry::ListDescType>(
        op.getDesc().getType());
    auto storageType = mulberry::ListStorageType::get(
        op.getContext(), descType.getElementType());
    auto data = mulberry::ListDescDataOp::create(
        rewriter, op.getLoc(), storageType, op.getDesc());
    auto loaded = mulberry::ListLoadOp::create(
        rewriter, op.getLoc(), op.getResult().getType(), data.getResult(),
        op.getIndex());
    rewriter.replaceOp(op, loaded.getResult());
    return success();
  }
};

class ListDescPackOpConversion
    : public OpConversionPattern<mulberry::ListDescPackOp> {
public:
  using OpConversionPattern<mulberry::ListDescPackOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::ListDescPackOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto descType = llvm::cast<mulberry::ListDescType>(
        op.getResult().getType());
    auto layout = convertToListABILayout(descType);
    if (!layout)
      return rewriter.notifyMatchFailure(
          op, "list descriptor ABI needs a lowerable element type");

    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (resultType != layout->descriptorType)
      return rewriter.notifyMatchFailure(
          op, "list descriptor result type does not match its element ABI");

    auto dataType = getTypeConverter()->convertType(op.getData().getType());
    if (!dataType)
      return rewriter.notifyMatchFailure(
          op, "list descriptor ABI needs lowerable list storage");
    if (!llvm::isa<MemRefType, LLVM::LLVMPointerType>(dataType))
      return rewriter.notifyMatchFailure(
          op, "list descriptor ABI needs lowered list storage");

    // desc_pack owns the actual `{length, data}` materialization. Keeping this
    // here prevents desc_to_abi from becoming a hidden function-boundary bridge.
    auto length = arith::IndexCastOp::create(
        rewriter, op.getLoc(), getI64Type(op.getContext()),
        adaptor.getLength());
    auto dataPointer = adaptor.getData();
    if (llvm::isa<MemRefType>(dataType))
      dataPointer = createMemRefDataPointer(op.getLoc(), rewriter,
                                            adaptor.getData());
    auto desc = createListABIDesc(op.getLoc(), rewriter, *layout,
                                  length.getResult(), dataPointer);

    rewriter.replaceOp(op, desc);
    return success();
  }
};

class ListDescLengthOpConversion
    : public OpConversionPattern<mulberry::ListDescLengthOp> {
public:
  using OpConversionPattern<mulberry::ListDescLengthOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::ListDescLengthOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto descType = llvm::cast<mulberry::ListDescType>(
        op.getDesc().getType());
    auto layout = convertToListABILayout(descType);
    if (!layout)
      return rewriter.notifyMatchFailure(
          op, "list descriptor length needs a lowerable ABI descriptor");

    // List ABI field 0 is the runtime length stored as i64. The Mulberry op
    // returns index because length is used by indexing and loop operations.
    auto length = LLVM::ExtractValueOp::create(
        rewriter, op.getLoc(), getI64Type(op.getContext()),
        adaptor.getDesc(), ArrayRef<int64_t>{0});
    auto indexLength = arith::IndexCastOp::create(
        rewriter, op.getLoc(), rewriter.getIndexType(), length.getResult());
    rewriter.replaceOp(op, indexLength.getResult());
    return success();
  }
};

class ListDescDataOpConversion
    : public OpConversionPattern<mulberry::ListDescDataOp> {
public:
  using OpConversionPattern<mulberry::ListDescDataOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::ListDescDataOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!llvm::isa_and_nonnull<LLVM::LLVMPointerType>(resultType))
      return rewriter.notifyMatchFailure(
          op, "list descriptor data needs pointer-backed storage");

    // List ABI field 1 is the backend data pointer. Only pointer-backed
    // descriptor storage can be reconstructed from this field today.
    auto data = LLVM::ExtractValueOp::create(
        rewriter, op.getLoc(), resultType, adaptor.getDesc(),
        ArrayRef<int64_t>{1});
    rewriter.replaceOp(op, data.getResult());
    return success();
  }
};

class ListDescToABIOpLowering
    : public OpConversionPattern<mulberry::ListDescToABIOp> {
public:
  using OpConversionPattern<mulberry::ListDescToABIOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::ListDescToABIOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto descType = llvm::cast<mulberry::ListDescType>(
        op.getDesc().getType());
    auto layout = convertToListABILayout(descType);
    if (!layout)
      return rewriter.notifyMatchFailure(
          op, "list descriptor ABI needs a lowerable element type");

    if (op.getResult().getType() != layout->descriptorType)
      return rewriter.notifyMatchFailure(
          op, "list descriptor ABI result type does not match its element ABI");

    // This op is intentionally only a local marker. desc_pack lowering already
    // materialized the backend ABI record for the descriptor operand.
    rewriter.replaceOp(op, adaptor.getDesc());
    return success();
  }
};

class RecordGetFieldOpLowering
    : public OpConversionPattern<mulberry::RecordGetFieldOp> {
public:
  using OpConversionPattern<mulberry::RecordGetFieldOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::RecordGetFieldOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto recordType = getRecordType(op.getRecord().getType());
    if (!recordType)
      return rewriter.notifyMatchFailure(
          op, "field address needs a pointer to a record");

    auto llvmRecordType = getTypeConverter()->convertType(recordType);
    auto fieldType = getRecordFieldType(recordType, op.getField());
    if (!llvmRecordType || !fieldType)
      return rewriter.notifyMatchFailure(
          op, "field address needs a lowerable record field");

    auto fieldIndex = static_cast<int32_t>(
        recordType.getFieldIndex(op.getField()));
    auto fieldPtr = LLVM::GEPOp::create(
        rewriter, op.getLoc(), getPtrType(op.getContext()), llvmRecordType,
        adaptor.getRecord(), ArrayRef<LLVM::GEPArg>{0, fieldIndex});
    rewriter.replaceOp(op, fieldPtr.getResult());
    return success();
  }
};

class RecordExtractOpLowering
    : public OpConversionPattern<mulberry::RecordExtractOp> {
public:
  using OpConversionPattern<mulberry::RecordExtractOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::RecordExtractOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto recordType = llvm::dyn_cast<mulberry::RecordType>(
        op.getRecord().getType());
    auto fieldType = getRecordFieldType(recordType, op.getField());
    if (!fieldType)
      return rewriter.notifyMatchFailure(
          op, "record extract needs a lowerable field");

    auto fieldIndex = static_cast<int64_t>(
        recordType.getFieldIndex(op.getField()));
    auto extracted = LLVM::ExtractValueOp::create(
        rewriter, op.getLoc(), fieldType, adaptor.getRecord(), fieldIndex);
    rewriter.replaceOp(op, extracted.getResult());
    return success();
  }
};

class TensorAllocOpLowering
    : public OpConversionPattern<mulberry::TensorAllocOp> {
public:
  using OpConversionPattern<mulberry::TensorAllocOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::TensorAllocOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto resultType = llvm::dyn_cast_or_null<MemRefType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType)
      return rewriter.notifyMatchFailure(
          op, "tensor alloc needs a lowerable memref result type");

    auto alloc = memref::AllocOp::create(rewriter, op.getLoc(), resultType,
                                         adaptor.getDynamicSizes());
    rewriter.replaceOp(op, alloc.getResult());
    return success();
  }
};

class TensorDimOpLowering : public OpConversionPattern<mulberry::TensorDimOp> {
public:
  using OpConversionPattern<mulberry::TensorDimOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::TensorDimOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto dim = memref::DimOp::create(rewriter, op.getLoc(),
                                     adaptor.getTensor(), adaptor.getIndex());
    rewriter.replaceOp(op, dim.getResult());
    return success();
  }
};

class TensorCastOpLowering
    : public OpConversionPattern<mulberry::TensorCastOp> {
public:
  using OpConversionPattern<mulberry::TensorCastOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::TensorCastOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto destType = getTypeConverter()->convertType(op.getDest().getType());
    if (!destType)
      return rewriter.notifyMatchFailure(
          op, "tensor cast needs a lowerable memref result type");

    auto cast = memref::CastOp::create(rewriter, op.getLoc(), destType,
                                       adaptor.getSource());
    rewriter.replaceOp(op, cast.getResult());
    return success();
  }
};

class TensorLoadOpLowering
    : public OpConversionPattern<mulberry::TensorLoadOp> {
public:
  using OpConversionPattern<mulberry::TensorLoadOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::TensorLoadOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto load = memref::LoadOp::create(rewriter, op.getLoc(),
                                       adaptor.getTensor(),
                                       adaptor.getIndices());
    rewriter.replaceOp(op, load.getResult());
    return success();
  }
};

class TensorStoreOpLowering
    : public OpConversionPattern<mulberry::TensorStoreOp> {
public:
  using OpConversionPattern<mulberry::TensorStoreOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::TensorStoreOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    memref::StoreOp::create(rewriter, op.getLoc(), adaptor.getValue(),
                            adaptor.getTensor(), adaptor.getIndices());
    rewriter.eraseOp(op);
    return success();
  }
};

class TensorDescPackOpConversion
    : public OpConversionPattern<mulberry::TensorDescPackOp> {
public:
  using OpConversionPattern<mulberry::TensorDescPackOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::TensorDescPackOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto descType = llvm::cast<mulberry::TensorDescType>(
        op.getResult().getType());
    auto layout = convertToTensorABILayout(descType);
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (resultType != layout.descriptorType)
      return rewriter.notifyMatchFailure(
          op, "tensor descriptor result type does not match its ABI");

    if (!llvm::isa<MemRefType>(adaptor.getTensor().getType()))
      return rewriter.notifyMatchFailure(
          op, "tensor descriptor ABI needs lowered tensor storage");

    // desc_pack owns `{data, sizes, strides}` materialization. desc_to_abi is
    // only a local marker and must not become a hidden function-boundary bridge.
    auto desc = createTensorABIDesc(op.getLoc(), rewriter, layout,
                                    adaptor.getTensor(), descType);
    rewriter.replaceOp(op, desc);
    return success();
  }
};

class TensorDescToABIOpLowering
    : public OpConversionPattern<mulberry::TensorDescToABIOp> {
public:
  using OpConversionPattern<mulberry::TensorDescToABIOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::TensorDescToABIOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto descType = llvm::cast<mulberry::TensorDescType>(
        op.getDesc().getType());
    auto layout = convertToTensorABILayout(descType);
    if (op.getResult().getType() != layout.descriptorType)
      return rewriter.notifyMatchFailure(
          op, "tensor descriptor ABI result type does not match its ABI");

    rewriter.replaceOp(op, adaptor.getDesc());
    return success();
  }
};

struct LowerMulberry : public impl::LowerMulberryBase<LowerMulberry> {
  using impl::LowerMulberryBase<LowerMulberry>::LowerMulberryBase;

  auto runOnOperation() -> void final {
    if (failed(rejectListBoundaries(getOperation())))
      return signalPassFailure();

    if (failed(rejectTensorDescBoundaries(getOperation())))
      return signalPassFailure();

    if (failed(rejectTensorDescUnpack(getOperation())))
      return signalPassFailure();

    if (failed(rejectTensorHandleOps(getOperation())))
      return signalPassFailure();

    // list_desc is only a descriptor-stage local value today. Folding local
    // pack/projection pairs is safe, but function-boundary descriptors remain
    // illegal until list storage is converted through MLIR's memref descriptor
    // finalization. LLVM structs cannot directly contain memref-typed fields.
    if (containsListDescOp(getOperation())) {
      RewritePatternSet descPatterns(&getContext());
      descPatterns.add<ListDescDataOpLowering, ListDescGetOpLowering,
                       ListDescLengthOpLowering, ListDescPackOpLowering,
                       ListToDescOpLowering>(
          &getContext());
      if (failed(
              applyPatternsGreedily(getOperation(), std::move(descPatterns))))
        return signalPassFailure();
    }

    MulberryTypeConverter typeConverter;

    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect,
                           linalg::LinalgDialect, LLVM::LLVMDialect,
                           math::MathDialect, memref::MemRefDialect,
                           scf::SCFDialect>();
    target.addIllegalDialect<cherry_nn::CherryNNDialect>();
    target.addIllegalDialect<mulberry::MulberryDialect>();
    target.addDynamicallyLegalOp<func::FuncOp>(
        [&](func::FuncOp op) {
          // Tensor function boundaries lower to memref today. Only explicit
          // list_desc<TensorDesc> arguments may use the List ABI descriptor;
          // returns stay illegal until ownership/lifetime rules are clear.
          if (containsTensorDescType(op.getFunctionType()) ||
              containsUnsupportedListArg(op.getFunctionType().getInputs()) ||
              containsListBoundaryType(op.getFunctionType().getResults()))
            return false;
          return typeConverter.isSignatureLegal(op.getFunctionType());
        });
    target.addDynamicallyLegalOp<func::CallOp, func::ReturnOp>(
        [&](Operation* op) {
          if (containsTensorDescType(op->getOperandTypes()) ||
              containsTensorDescType(op->getResultTypes()) ||
              containsUnsupportedListArg(op->getOperandTypes()) ||
              containsListBoundaryType(op->getResultTypes()))
            return false;
          if (llvm::isa<func::ReturnOp>(op) &&
              containsListBoundaryType(op->getOperandTypes()))
            return false;
          return typeConverter.isLegal(op);
        });

    RewritePatternSet patterns(&getContext());
    patterns.add<AllocaOpLowering, ListAllocOpLowering,
                 ListCreateOpLowering, ListGetOpLowering, ListLengthOpLowering,
                 ListLoadOpLowering, ListDescDataOpConversion,
                 ListDescLengthOpConversion, ListDescPackOpConversion,
                 ListDescToABIOpLowering, ListSizeOpLowering,
                 ListStoreOpLowering, LoadOpLowering, RecordExtractOpLowering,
                 RecordGetFieldOpLowering, StoreOpLowering, TensorAllocOpLowering,
                 TensorCastOpLowering, TensorDimOpLowering,
                 TensorDescPackOpConversion, TensorDescToABIOpLowering,
                 TensorLoadOpLowering, TensorStoreOpLowering>(
        typeConverter, &getContext());
    populateCherryNNToLinalgPatterns(typeConverter, patterns);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    // Keep unresolved Mulberry ops fail-fast until each one has an explicit
    // lowering. A silent no-op pass would make backend tests look lower than
    // they are.
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPartialConversion(getOperation(), target, patternSet)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::cherry
