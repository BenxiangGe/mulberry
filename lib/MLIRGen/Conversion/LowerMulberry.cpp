//===--- LowerMulberry.cpp -----------------------------------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "mulberry/MLIRGen/Conversion/MulberryPasses.h"
#include "mulberry/MLIRGen/IR/MulberryDialect.h"
#include "mulberry/MLIRGen/IR/MulberryOps.h"
#include "mulberry/MLIRGen/IR/MulberryTypes.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <optional>
#include <string>

namespace mlir::mulberry_core {

#define GEN_PASS_DEF_LOWERMULBERRY
#include "mulberry/MLIRGen/Conversion/MulberryPasses.h.inc"

namespace {

static auto getPtrType(MLIRContext* context) -> LLVM::LLVMPointerType {
  return LLVM::LLVMPointerType::get(context);
}

static auto getI64Type(MLIRContext* context) -> IntegerType {
  return IntegerType::get(context, 64);
}

static auto getTensorDataAddressSpace(MLIRContext* context) -> Attribute {
  return LLVM::AddressSpaceAttr::get(context, 0);
}

static auto isScalarStorageType(Type type) -> bool {
  return type.isIndex() || llvm::isa<IntegerType, FloatType>(type);
}

static auto convertRecordToBackendType(mulberry_core::RecordType type)
    -> std::optional<Type>;
static auto convertPtrType(mulberry_core::PtrType type) -> std::optional<Type>;

static auto convertMemRefShape(ArrayRef<int64_t> shape)
    -> std::vector<int64_t> {
  std::vector<int64_t> memrefShape;
  for (auto dim : shape) {
    memrefShape.push_back(dim < 0 ? ShapedType::kDynamic : dim);
  }
  return memrefShape;
}

// mulberry_core.tensor is the compiler-owned internal tensor IR. It lowers to
// memref; source-level Tensor<T> is a stdlib record header.
static auto convertTensorToMemRefType(mulberry_core::TensorType type)
    -> MemRefType {
  return MemRefType::get(convertMemRefShape(type.getShape()),
                         type.getElementType());
}

static auto convertTensorToDataMemRefType(mulberry_core::TensorType type)
    -> MemRefType {
  auto layout = MemRefLayoutAttrInterface{};
  return MemRefType::get(convertMemRefShape(type.getShape()),
                         type.getElementType(), layout,
                         getTensorDataAddressSpace(type.getContext()));
}

// This pass is a transitional storage lowering, not the final
// Mulberry-to-LLVM ABI lowering. Internal tensor values become memrefs while
// scalar/record stack storage still uses LLVM dialect. Domain packages such as
// mulberry.nn are intentionally not lowered here; they live outside core.
static auto convertToBackendType(Type type) -> std::optional<Type> {
  if (auto recordType = llvm::dyn_cast<mulberry_core::RecordType>(type))
    return convertRecordToBackendType(recordType);

  if (auto ptrType = llvm::dyn_cast<mulberry_core::PtrType>(type))
    return convertPtrType(ptrType);

  if (isScalarStorageType(type))
    return type;

  return std::nullopt;
}

static auto convertAllocaElementType(Type type) -> std::optional<Type> {
  if (auto tensorType = llvm::dyn_cast<mulberry_core::TensorType>(type))
    return convertTensorToMemRefType(tensorType);

  return convertToBackendType(type);
}

using TensorABIDescriptorType = LLVM::LLVMStructType;

// Lowering-only descriptor for moving internal tensor/memref values across
// temporary boundaries. This is not the source-level Tensor<T> header layout.
//
// Tensor ABI descriptor layout:
//
//   { data: ptr,
//     sizes: array<rank x i64>,
//     strides: array<rank x i64> }
//
// The MLIRContext used below is only needed to create the MLIR Type objects.
// It is not stored in the descriptor. The descriptor fields are exactly the
// Types pushed into `fields`.
struct TensorABILayout {
  TensorABIDescriptorType descriptorType;
};

static auto convertToTensorABILayout(MLIRContext* context, size_t rank)
    -> TensorABILayout {
  auto indexArrayType = LLVM::LLVMArrayType::get(getI64Type(context), rank);

  std::vector<Type> fields;
  fields.push_back(getPtrType(context)); // data pointer
  fields.push_back(indexArrayType);      // sizes[rank]
  fields.push_back(indexArrayType);      // strides[rank]
  auto descriptorType = LLVM::LLVMStructType::getLiteral(context, fields);

  return TensorABILayout{descriptorType};
}

static auto convertToTensorABILayout(mulberry_core::TensorType type)
    -> TensorABILayout {
  return convertToTensorABILayout(type.getContext(), type.getShape().size());
}

static auto convertTensorToABIDescriptorType(mulberry_core::TensorType type)
    -> TensorABIDescriptorType {
  return convertToTensorABILayout(type).descriptorType;
}

static auto convertPointerElementType(Type type) -> std::optional<Type> {
  if (auto tensorType = llvm::dyn_cast<mulberry_core::TensorType>(type))
    return convertTensorToABIDescriptorType(tensorType);

  return convertToBackendType(type);
}

static auto createMemRefDataPointer(
    Location location, ConversionPatternRewriter& rewriter, Value storage)
    -> Value {
  // Runtime calls and ABI descriptors need an opaque LLVM pointer, not the full
  // memref descriptor. Extract the memref's aligned data pointer and cast that
  // raw address to ptr.
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

static auto createSizeOf(Location location, OpBuilder& builder, Type type)
    -> Value {
  // This is the standard LLVM IR sizeof trick:
  //   ptrtoint (getelementptr T, ptr null, 1) to i64
  // It lets LLVM's datalayout decide the real ABI size instead of hardcoding
  // pointer and struct field sizes in Mulberry lowering.
  auto ptrType = getPtrType(builder.getContext());
  auto nullPtr = LLVM::ZeroOp::create(builder, location, ptrType);
  auto nextPtr = LLVM::GEPOp::create(builder, location, ptrType, type,
                                     nullPtr, ArrayRef<LLVM::GEPArg>{1});
  return LLVM::PtrToIntOp::create(builder, location,
                                  getI64Type(builder.getContext()), nextPtr);
}

static auto createTensorByteSize(Location location,
                                 ConversionPatternRewriter& rewriter,
                                 Value tensor) -> FailureOr<Value> {
  auto memRefType = llvm::dyn_cast<MemRefType>(tensor.getType());
  if (!memRefType)
    return failure();

  auto i64Type = getI64Type(rewriter.getContext());
  Value elementCount = LLVM::ConstantOp::create(
      rewriter, location, i64Type, rewriter.getI64IntegerAttr(1));

  for (int64_t dim = 0; dim < memRefType.getRank(); ++dim) {
    auto index = arith::ConstantIndexOp::create(rewriter, location, dim);
    auto size = memref::DimOp::create(rewriter, location, tensor, index);
    auto sizeI64 = arith::IndexCastOp::create(
        rewriter, location, i64Type, size.getResult());
    elementCount = arith::MulIOp::create(rewriter, location, elementCount,
                                         sizeI64.getResult());
  }

  auto elementBytes = createSizeOf(location, rewriter,
                                   memRefType.getElementType());
  auto byteSize = arith::MulIOp::create(rewriter, location, elementCount,
                                        elementBytes);
  return byteSize.getResult();
}

static auto callBoehmMalloc(Location location, OpBuilder& builder, Operation* op,
                            Value sizeInBytes) -> FailureOr<Value> {
  auto moduleOp = op->getParentOfType<ModuleOp>();
  if (!moduleOp)
    return failure();

  auto mallocFn = LLVM::lookupOrCreateFn(
      builder, moduleOp, "mulberry_boehm_malloc",
      {getI64Type(builder.getContext())}, getPtrType(builder.getContext()));
  if (failed(mallocFn))
    return failure();

  auto mallocCall = LLVM::CallOp::create(
      builder, location, *mallocFn, ValueRange{sizeInBytes});
  return mallocCall.getResult();
}

static auto extractRecordField(Location location,
                               ConversionPatternRewriter& rewriter,
                               Value record,
                               mulberry_core::RecordType recordType,
                               StringRef field) -> FailureOr<Value> {
  auto sourceFieldType = recordType.getFieldType(field);
  if (!sourceFieldType)
    return failure();

  auto fieldType = convertToBackendType(sourceFieldType);
  if (!fieldType)
    return failure();

  auto fieldIndex = static_cast<int64_t>(recordType.getFieldIndex(field));
  return LLVM::ExtractValueOp::create(rewriter, location, *fieldType, record,
                                      ArrayRef<int64_t>{fieldIndex})
      .getResult();
}

static auto loadI64PointerElement(Location location,
                                  ConversionPatternRewriter& rewriter,
                                  Value ptr, size_t index) -> Value {
  auto context = rewriter.getContext();
  auto i64Type = getI64Type(context);
  auto indexValue = LLVM::ConstantOp::create(
      rewriter, location, i64Type,
      rewriter.getI64IntegerAttr(static_cast<int64_t>(index)));
  auto slot = LLVM::GEPOp::create(
      rewriter, location, getPtrType(context), i64Type, ptr,
      ArrayRef<Value>{indexValue.getResult()});
  return LLVM::LoadOp::create(rewriter, location, i64Type,
                              slot.getResult()).getResult();
}

static auto storePointerElement(Location location,
                                ConversionPatternRewriter& rewriter,
                                Type elementType, Value ptr, size_t index,
                                Value value) -> void {
  auto context = rewriter.getContext();
  auto i64Type = getI64Type(context);
  auto indexValue = LLVM::ConstantOp::create(
      rewriter, location, i64Type,
      rewriter.getI64IntegerAttr(static_cast<int64_t>(index)));
  auto slot = LLVM::GEPOp::create(
      rewriter, location, getPtrType(context), elementType, ptr,
      ArrayRef<Value>{indexValue.getResult()});
  LLVM::StoreOp::create(rewriter, location, value, slot.getResult());
}

static auto createHeapArray(Location location,
                            ConversionPatternRewriter& rewriter,
                            Operation* op, Type elementType,
                            Value count) -> FailureOr<Value> {
  auto elementBytes = createSizeOf(location, rewriter, elementType);
  auto sizeInBytes = arith::MulIOp::create(rewriter, location, elementBytes,
                                           count);
  return callBoehmMalloc(location, rewriter, op, sizeInBytes.getResult());
}

static auto createListRecordValue(Location location,
                                  ConversionPatternRewriter& rewriter,
                                  mulberry_core::RecordType listType,
                                  Value length, Value data)
    -> FailureOr<Value> {
  auto listBackendType = convertRecordToBackendType(listType);
  if (!listBackendType)
    return failure();

  auto list = LLVM::UndefOp::create(rewriter, location, *listBackendType);
  Value current = list.getResult();
  auto lengthIndex = static_cast<int64_t>(listType.getFieldIndex("length"));
  auto capacityIndex =
      static_cast<int64_t>(listType.getFieldIndex("capacity"));
  auto dataIndex = static_cast<int64_t>(listType.getFieldIndex("data"));
  current = LLVM::InsertValueOp::create(
      rewriter, location, current, length, ArrayRef<int64_t>{lengthIndex});
  current = LLVM::InsertValueOp::create(
      rewriter, location, current, length, ArrayRef<int64_t>{capacityIndex});
  current = LLVM::InsertValueOp::create(
      rewriter, location, current, data, ArrayRef<int64_t>{dataIndex});
  return current;
}

static auto extractListDataPointer(Location location,
                                   ConversionPatternRewriter& rewriter,
                                   Value list,
                                   mulberry_core::RecordType listType)
    -> FailureOr<Value> {
  auto data = extractRecordField(location, rewriter, list, listType, "data");
  if (failed(data))
    return failure();
  return *data;
}

static auto createTensorABIDesc(
    Location location, ConversionPatternRewriter& rewriter,
    const TensorABILayout& layout, Value tensor, mulberry_core::TensorType type)
    -> Value {
  auto desc = LLVM::UndefOp::create(rewriter, location,
                                    layout.descriptorType);
  // Field 0 stores only the raw aligned data pointer. The memref metadata is
  // re-materialized below as explicit sizes/strides fields.
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

static auto createTensorABIDescFromStorage(
    Location location, ConversionPatternRewriter& rewriter,
    const TensorABILayout& layout, Value tensorRecord,
    mulberry_core::RecordType storageType, mulberry_core::TensorType tensorType)
    -> FailureOr<Value> {
  auto data =
      extractRecordField(location, rewriter, tensorRecord, storageType, "data");
  auto sizes =
      extractRecordField(location, rewriter, tensorRecord, storageType, "sizes");
  auto strides =
      extractRecordField(location, rewriter, tensorRecord, storageType,
                         "strides");
  if (failed(data) || failed(sizes) || failed(strides))
    return failure();

  auto sizesType = llvm::dyn_cast<mulberry_core::RecordType>(
      storageType.getFieldType("sizes"));
  auto stridesType = llvm::dyn_cast<mulberry_core::RecordType>(
      storageType.getFieldType("strides"));
  if (!sizesType || !stridesType)
    return failure();
  auto sizesData =
      extractListDataPointer(location, rewriter, *sizes, sizesType);
  auto stridesData =
      extractListDataPointer(location, rewriter, *strides, stridesType);
  if (failed(sizesData) || failed(stridesData))
    return failure();

  auto desc = LLVM::UndefOp::create(rewriter, location,
                                    layout.descriptorType);
  Value current = LLVM::InsertValueOp::create(
      rewriter, location, desc.getResult(), *data,
      ArrayRef<int64_t>{0}).getResult();

  for (size_t dim = 0; dim < tensorType.getShape().size(); ++dim) {
    auto size = loadI64PointerElement(location, rewriter, *sizesData, dim);
    auto stride = loadI64PointerElement(location, rewriter, *stridesData, dim);
    current = LLVM::InsertValueOp::create(
        rewriter, location, current, size,
        ArrayRef<int64_t>{1, static_cast<int64_t>(dim)}).getResult();
    current = LLVM::InsertValueOp::create(
        rewriter, location, current, stride,
        ArrayRef<int64_t>{2, static_cast<int64_t>(dim)}).getResult();
  }

  return current;
}

static auto createScalarMemRefFromDataPointer(
    Location location, ConversionPatternRewriter& rewriter,
    MemRefType memRefType, Value dataPointer) -> Value {
  auto context = rewriter.getContext();
  auto i64Type = getI64Type(context);
  auto descType = LLVM::LLVMStructType::getLiteral(
      context, {getPtrType(context), getPtrType(context), i64Type});
  auto desc = LLVM::UndefOp::create(rewriter, location, descType);
  auto offset = LLVM::ConstantOp::create(rewriter, location, i64Type,
                                         rewriter.getI64IntegerAttr(0));

  // This is the rank-0 memref descriptor expected by memref-to-LLVM:
  // {allocatedPtr, alignedPtr, offset}. Sizes and strides live in the Mulberry
  // Tensor ABI descriptor and are applied by memref.reinterpret_cast below.
  Value current = LLVM::InsertValueOp::create(
      rewriter, location, desc.getResult(), dataPointer,
      ArrayRef<int64_t>{0}).getResult();
  current = LLVM::InsertValueOp::create(
      rewriter, location, current, dataPointer,
      ArrayRef<int64_t>{1}).getResult();
  current = LLVM::InsertValueOp::create(
      rewriter, location, current, offset.getResult(),
      ArrayRef<int64_t>{2}).getResult();
  return UnrealizedConversionCastOp::create(
      rewriter, location, memRefType, current).getResult(0);
}

static auto createTensorFromABIDesc(
    Location location, ConversionPatternRewriter& rewriter, Value desc,
    mulberry_core::TensorType type) -> Value {
  auto context = rewriter.getContext();
  auto dataMemRefType = convertTensorToDataMemRefType(type);
  auto resultMemRefType = convertTensorToMemRefType(type);

  auto dataPointer = LLVM::ExtractValueOp::create(
      rewriter, location, getPtrType(context), desc, ArrayRef<int64_t>{0});
  auto layout = MemRefLayoutAttrInterface{};
  auto baseType = MemRefType::get({}, type.getElementType(), layout,
                                  getTensorDataAddressSpace(context));
  // The ABI data pointer has no memref metadata attached. Start from a scalar
  // base memref, then reinterpret it with the descriptor's sizes and strides.
  auto base = createScalarMemRefFromDataPointer(
      location, rewriter, baseType, dataPointer.getResult());

  Value offset = arith::ConstantIndexOp::create(rewriter, location, 0);
  std::vector<Value> sizes;
  std::vector<Value> strides;
  for (size_t dim = 0; dim < type.getShape().size(); ++dim) {
    auto size = LLVM::ExtractValueOp::create(
        rewriter, location, getI64Type(context), desc,
        ArrayRef<int64_t>{1, static_cast<int64_t>(dim)});
    auto stride = LLVM::ExtractValueOp::create(
        rewriter, location, getI64Type(context), desc,
        ArrayRef<int64_t>{2, static_cast<int64_t>(dim)});
    sizes.push_back(arith::IndexCastOp::create(
                        rewriter, location, rewriter.getIndexType(),
                        size.getResult())
                        .getResult());
    strides.push_back(arith::IndexCastOp::create(
                          rewriter, location, rewriter.getIndexType(),
                          stride.getResult())
                          .getResult());
  }

  // Tensor descriptor fields are the memref metadata. Reinterpret-cast uses
  // them to rebuild the ranked memref view from the base data pointer without
  // inventing opaque ptr_metadata or raw integer-pointer tricks.
  std::vector<int64_t> dynamicShape(dataMemRefType.getRank(),
                                    ShapedType::kDynamic);
  std::vector<int64_t> dynamicStrides(dataMemRefType.getRank(),
                                      ShapedType::kDynamic);
  auto stridedType = MemRefType::get(
      dynamicShape, dataMemRefType.getElementType(),
      StridedLayoutAttr::get(context, ShapedType::kDynamic, dynamicStrides),
      dataMemRefType.getMemorySpace());
  auto stridedTensor = memref::ReinterpretCastOp::create(
      rewriter, location, stridedType, base, offset, sizes, strides,
      ArrayRef<NamedAttribute>{});
  auto dataTensor = memref::CastOp::create(
      rewriter, location, dataMemRefType, stridedTensor.getResult());
  auto result = memref::MemorySpaceCastOp::create(
      rewriter, location, resultMemRefType, dataTensor.getResult());
  return result.getResult();
}

static auto convertRecordToBackendType(mulberry_core::RecordType type)
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

static auto convertPtrType(mulberry_core::PtrType type) -> std::optional<Type> {
  auto pointeeType = type.getPointeeType();
  if (auto tensorType = llvm::dyn_cast<mulberry_core::TensorType>(pointeeType)) {
    // Ptr<Tensor> follows the same C/C++ pointer model as scalar and record
    // pointers. The pointee storage is a Tensor ABI descriptor; load/store
    // pack and unpack that descriptor to/from the memref value used by linalg.
    (void)tensorType;
    return getPtrType(type.getContext());
  }

  if (convertToBackendType(pointeeType))
    return getPtrType(type.getContext());

  return std::nullopt;
}

// Safety net for Mulberry types that should not fall through to the identity
// conversion after their specific lowering conversion fails.
static auto rejectUnloweredMulberryType(Type type, SmallVectorImpl<Type>&)
    -> std::optional<LogicalResult> {
  if (auto recordType = llvm::dyn_cast<mulberry_core::RecordType>(type))
    if (!convertRecordToBackendType(recordType))
      return failure();

  if (auto ptrType = llvm::dyn_cast<mulberry_core::PtrType>(type))
    if (!convertPtrType(ptrType))
      return failure();

  return std::nullopt;
}

class MulberryTypeConverter : public TypeConverter {
public:
    MulberryTypeConverter() {
      addConversion([](Type type) { return type; });
      addConversion(rejectUnloweredMulberryType);
      addConversion(convertRecordToBackendType);
      addConversion(convertPtrType);
      addConversion(convertTensorToMemRefType);
    // Keep unsupported Mulberry types illegal until each one has a real
    // lowering. The identity conversion above is only for non-Mulberry types.
  }
};

class AllocaOpLowering : public OpConversionPattern<mulberry_core::AllocaOp> {
public:
  using OpConversionPattern<mulberry_core::AllocaOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry_core::AllocaOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    if (auto tensorType = llvm::dyn_cast<mulberry_core::TensorType>(
            op.getElementType())) {
      auto elementType = convertTensorToABIDescriptorType(tensorType);
      auto elementCount = arith::ConstantIntOp::create(
          rewriter, op.getLoc(), 1, /*width=*/64);
      auto alloca = LLVM::AllocaOp::create(
          rewriter, op.getLoc(), getPtrType(op.getContext()), elementType,
          elementCount.getResult(), /*alignment=*/0);
      rewriter.replaceOp(op, alloca.getResult());
      return success();
    }

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

class HeapAllocOpLowering
    : public OpConversionPattern<mulberry_core::HeapAllocOp> {
public:
  using OpConversionPattern<mulberry_core::HeapAllocOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry_core::HeapAllocOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto elementType = convertPointerElementType(op.getElementType());
    if (!elementType)
      return rewriter.notifyMatchFailure(
          op, "heap alloc needs a lowerable backend type");

    auto elementBytes = createSizeOf(op.getLoc(), rewriter, *elementType);
    auto count = arith::IndexCastOp::create(
        rewriter, op.getLoc(), getI64Type(op.getContext()),
        adaptor.getCount());
    auto sizeInBytes = arith::MulIOp::create(
        rewriter, op.getLoc(), elementBytes, count.getResult());
    auto heapPtr = callBoehmMalloc(op.getLoc(), rewriter, op,
                                   sizeInBytes.getResult());
    if (failed(heapPtr))
      return failure();

    rewriter.replaceOp(op, *heapPtr);
    return success();
  }
};

class PtrIndexOpLowering
    : public OpConversionPattern<mulberry_core::PtrIndexOp> {
public:
  using OpConversionPattern<mulberry_core::PtrIndexOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry_core::PtrIndexOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto ptrType = llvm::cast<mulberry_core::PtrType>(op.getPtr().getType());
    auto elementType = convertPointerElementType(ptrType.getPointeeType());
    if (!elementType)
      return rewriter.notifyMatchFailure(
          op, "pointer index needs a lowerable backend element type");

    auto index = arith::IndexCastOp::create(
        rewriter, op.getLoc(), getI64Type(op.getContext()),
        adaptor.getIndex());
    auto elementPtr = LLVM::GEPOp::create(
        rewriter, op.getLoc(), getPtrType(op.getContext()), *elementType,
        adaptor.getPtr(), ArrayRef<Value>{index.getResult()});
    rewriter.replaceOp(op, elementPtr.getResult());
    return success();
  }
};

class PtrCastOpLowering : public OpConversionPattern<mulberry_core::PtrCastOp> {
public:
  using OpConversionPattern<mulberry_core::PtrCastOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry_core::PtrCastOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    rewriter.replaceOp(op, adaptor.getPtr());
    return success();
  }
};

class LoadOpLowering : public OpConversionPattern<mulberry_core::LoadOp> {
public:
  using OpConversionPattern<mulberry_core::LoadOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry_core::LoadOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto ptrType = llvm::cast<mulberry_core::PtrType>(op.getPtr().getType());
    if (auto tensorType = llvm::dyn_cast<mulberry_core::TensorType>(
            ptrType.getPointeeType())) {
      auto layout = convertToTensorABILayout(tensorType);
      auto desc = LLVM::LoadOp::create(rewriter, op.getLoc(),
                                       layout.descriptorType,
                                       adaptor.getPtr());
      auto tensor = createTensorFromABIDesc(op.getLoc(), rewriter,
                                            desc.getResult(), tensorType);
      rewriter.replaceOp(op, tensor);
      return success();
    }

    auto elementType = convertAllocaElementType(ptrType.getPointeeType());
    if (!elementType)
      return rewriter.notifyMatchFailure(
          op, "load needs a lowerable result type");

    if (llvm::isa<MemRefType>(*elementType)) {
      auto load = memref::LoadOp::create(rewriter, op.getLoc(),
                                         adaptor.getPtr(), ValueRange{});
      rewriter.replaceOp(op, load.getResult());
      return success();
    }

    auto load = LLVM::LoadOp::create(rewriter, op.getLoc(), *elementType,
                                     adaptor.getPtr());
    rewriter.replaceOp(op, load.getResult());
    return success();
  }
};

class StoreOpLowering : public OpConversionPattern<mulberry_core::StoreOp> {
public:
  using OpConversionPattern<mulberry_core::StoreOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry_core::StoreOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto ptrType = llvm::dyn_cast<mulberry_core::PtrType>(op.getPtr().getType());
    if (!ptrType)
      return rewriter.notifyMatchFailure(
          op, "store target must be a Mulberry pointer");

    if (auto tensorType = llvm::dyn_cast<mulberry_core::TensorType>(
            ptrType.getPointeeType())) {
      if (!llvm::isa<MemRefType>(adaptor.getValue().getType()))
        return rewriter.notifyMatchFailure(
            op, "tensor store needs lowered tensor storage");

      auto layout = convertToTensorABILayout(tensorType);
      auto desc = createTensorABIDesc(op.getLoc(), rewriter, layout,
                                      adaptor.getValue(), tensorType);
      LLVM::StoreOp::create(rewriter, op.getLoc(), desc, adaptor.getPtr());
      rewriter.eraseOp(op);
      return success();
    }

    auto elementType = convertAllocaElementType(ptrType.getPointeeType());
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

class RecordGetFieldOpLowering
    : public OpConversionPattern<mulberry_core::RecordGetFieldOp> {
public:
  using OpConversionPattern<mulberry_core::RecordGetFieldOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry_core::RecordGetFieldOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto ptrType = llvm::dyn_cast<mulberry_core::PtrType>(op.getRecord().getType());
    if (!ptrType)
      return rewriter.notifyMatchFailure(
          op, "field address needs a pointer to a record");

    auto recordType =
        llvm::dyn_cast<mulberry_core::RecordType>(ptrType.getPointeeType());
    if (!recordType)
      return rewriter.notifyMatchFailure(
          op, "field address needs a pointer to a record");

    auto recordBackendType = convertRecordToBackendType(recordType);
    auto fieldType = convertToBackendType(recordType.getFieldType(
        op.getField()));
    if (!recordBackendType || !fieldType)
      return rewriter.notifyMatchFailure(
          op, "field address needs a lowerable record field");

    auto fieldIndex = static_cast<int32_t>(
        recordType.getFieldIndex(op.getField()));
    auto fieldPtr = LLVM::GEPOp::create(
        rewriter, op.getLoc(), getPtrType(op.getContext()), *recordBackendType,
        adaptor.getRecord(), ArrayRef<LLVM::GEPArg>{0, fieldIndex});
    rewriter.replaceOp(op, fieldPtr.getResult());
    return success();
  }
};

class RecordExtractOpLowering
    : public OpConversionPattern<mulberry_core::RecordExtractOp> {
public:
  using OpConversionPattern<mulberry_core::RecordExtractOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry_core::RecordExtractOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto recordType = llvm::dyn_cast<mulberry_core::RecordType>(
        op.getRecord().getType());
    if (!recordType)
      return rewriter.notifyMatchFailure(
          op, "record extract needs a record value");

    auto fieldType = convertToBackendType(recordType.getFieldType(
        op.getField()));
    if (!fieldType)
      return rewriter.notifyMatchFailure(
          op, "record extract needs a lowerable field");

    auto fieldIndex = static_cast<int64_t>(
        recordType.getFieldIndex(op.getField()));
    auto extracted = LLVM::ExtractValueOp::create(
        rewriter, op.getLoc(), *fieldType, adaptor.getRecord(), fieldIndex);
    rewriter.replaceOp(op, extracted.getResult());
    return success();
  }
};

class TensorViewOpLowering
    : public OpConversionPattern<mulberry_core::TensorViewOp> {
public:
  using OpConversionPattern<mulberry_core::TensorViewOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry_core::TensorViewOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto storageType = llvm::dyn_cast<mulberry_core::RecordType>(
        op.getTensor().getType());
    if (!storageType)
      return rewriter.notifyMatchFailure(
          op, "tensor view needs a Tensor record");

    auto tensorType = llvm::cast<mulberry_core::TensorType>(
        op.getResult().getType());
    auto layout = convertToTensorABILayout(tensorType);
    auto desc = createTensorABIDescFromStorage(
        op.getLoc(), rewriter, layout, adaptor.getTensor(), storageType,
        tensorType);
    if (failed(desc))
      return rewriter.notifyMatchFailure(
          op, "tensor view needs Tensor-compatible fields");

    auto tensor = createTensorFromABIDesc(op.getLoc(), rewriter, *desc,
                                          tensorType);
    rewriter.replaceOp(op, tensor);
    return success();
  }
};

class TensorPackOpLowering
    : public OpConversionPattern<mulberry_core::TensorPackOp> {
public:
  using OpConversionPattern<mulberry_core::TensorPackOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry_core::TensorPackOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto storageType = llvm::dyn_cast<mulberry_core::RecordType>(
        op.getResult().getType());
    if (!storageType)
      return rewriter.notifyMatchFailure(
          op, "tensor pack needs a Tensor record result");

    auto storageBackendType = convertRecordToBackendType(storageType);
    if (!storageBackendType)
      return rewriter.notifyMatchFailure(
          op, "tensor pack needs a lowerable Tensor record");

    auto tensorType = llvm::cast<mulberry_core::TensorType>(
        op.getTensor().getType());
    auto tensor = adaptor.getTensor();
    if (!llvm::isa<MemRefType>(tensor.getType()))
      return rewriter.notifyMatchFailure(
          op, "tensor pack needs lowered tensor storage");

    auto context = op.getContext();
    auto i64Type = getI64Type(context);
    auto rank = tensorType.getShape().size();
    auto rankValue = LLVM::ConstantOp::create(
        rewriter, op.getLoc(), i64Type,
        rewriter.getI64IntegerAttr(static_cast<int64_t>(rank)));

    auto sizes = createHeapArray(op.getLoc(), rewriter, op, i64Type,
                                 rankValue.getResult());
    auto strides = createHeapArray(op.getLoc(), rewriter, op, i64Type,
                                   rankValue.getResult());
    auto payloadBytes = createTensorByteSize(op.getLoc(), rewriter, tensor);
    if (failed(sizes) || failed(strides) || failed(payloadBytes))
      return failure();
    auto payload = callBoehmMalloc(op.getLoc(), rewriter, op, *payloadBytes);
    if (failed(payload))
      return failure();

    auto sizesType = llvm::dyn_cast<mulberry_core::RecordType>(
        storageType.getFieldType("sizes"));
    auto stridesType = llvm::dyn_cast<mulberry_core::RecordType>(
        storageType.getFieldType("strides"));
    if (!sizesType || !stridesType)
      return rewriter.notifyMatchFailure(
          op, "tensor pack needs List<i64> sizes and strides fields");

    auto layout = convertToTensorABILayout(tensorType);
    auto sourceDesc = createTensorABIDesc(op.getLoc(), rewriter, layout,
                                          tensor, tensorType);

    for (size_t dim = 0; dim < rank; ++dim) {
      auto size = LLVM::ExtractValueOp::create(
          rewriter, op.getLoc(), i64Type, sourceDesc,
          ArrayRef<int64_t>{1, static_cast<int64_t>(dim)});
      auto stride = LLVM::ExtractValueOp::create(
          rewriter, op.getLoc(), i64Type, sourceDesc,
          ArrayRef<int64_t>{2, static_cast<int64_t>(dim)});
      storePointerElement(op.getLoc(), rewriter, i64Type, *sizes, dim,
                          size.getResult());
      storePointerElement(op.getLoc(), rewriter, i64Type, *strides, dim,
                          stride.getResult());
    }

    // NN ops allocate temporary memrefs. The returned stdlib Tensor header must
    // own a Boehm heap payload, not point at temporary memref storage.
    auto heapDesc = LLVM::InsertValueOp::create(
        rewriter, op.getLoc(), sourceDesc, *payload, ArrayRef<int64_t>{0});
    auto heapTensor = createTensorFromABIDesc(op.getLoc(), rewriter,
                                              heapDesc.getResult(), tensorType);
    memref::CopyOp::create(rewriter, op.getLoc(), tensor, heapTensor);

    auto sizesList = createListRecordValue(
        op.getLoc(), rewriter, sizesType, rankValue.getResult(), *sizes);
    auto stridesList = createListRecordValue(
        op.getLoc(), rewriter, stridesType, rankValue.getResult(), *strides);
    if (failed(sizesList) || failed(stridesList))
      return failure();

    auto tensorRecord = LLVM::UndefOp::create(rewriter, op.getLoc(),
                                              *storageBackendType);
    Value current = tensorRecord.getResult();
    auto dataIndex = static_cast<int64_t>(storageType.getFieldIndex("data"));
    auto rankIndex = static_cast<int64_t>(storageType.getFieldIndex("rank"));
    auto numelIndex = static_cast<int64_t>(storageType.getFieldIndex("numel"));
    auto sizesIndex = static_cast<int64_t>(storageType.getFieldIndex("sizes"));
    auto stridesIndex =
        static_cast<int64_t>(storageType.getFieldIndex("strides"));
    current = LLVM::InsertValueOp::create(
        rewriter, op.getLoc(), current, *payload, ArrayRef<int64_t>{dataIndex});
    current = LLVM::InsertValueOp::create(
        rewriter, op.getLoc(), current, rankValue.getResult(),
        ArrayRef<int64_t>{rankIndex});
    auto elementCount = arith::DivUIOp::create(
        rewriter, op.getLoc(), *payloadBytes,
        createSizeOf(op.getLoc(), rewriter, tensorType.getElementType()));
    current = LLVM::InsertValueOp::create(
        rewriter, op.getLoc(), current, elementCount.getResult(),
        ArrayRef<int64_t>{numelIndex});
    current = LLVM::InsertValueOp::create(
        rewriter, op.getLoc(), current, *sizesList,
        ArrayRef<int64_t>{sizesIndex});
    current = LLVM::InsertValueOp::create(
        rewriter, op.getLoc(), current, *stridesList,
        ArrayRef<int64_t>{stridesIndex});

    if (!current)
      return rewriter.notifyMatchFailure(
          op, "tensor pack needs Tensor-compatible fields");

    rewriter.replaceOp(op, current);
    return success();
  }
};

struct LowerMulberry : public impl::LowerMulberryBase<LowerMulberry> {
  using impl::LowerMulberryBase<LowerMulberry>::LowerMulberryBase;

  auto runOnOperation() -> void final {
    MulberryTypeConverter typeConverter;

    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect,
                           linalg::LinalgDialect, LLVM::LLVMDialect,
                           math::MathDialect, memref::MemRefDialect,
                           scf::SCFDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addIllegalDialect<mulberry_core::MulberryDialect>();
    target.addDynamicallyLegalOp<func::FuncOp>(
        [&](func::FuncOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType());
        });
    target.addDynamicallyLegalOp<func::CallOp, func::ReturnOp>(
        [&](Operation* op) {
          return typeConverter.isLegal(op);
        });

    RewritePatternSet patterns(&getContext());
    patterns.add<AllocaOpLowering, HeapAllocOpLowering,
                 LoadOpLowering, PtrCastOpLowering, PtrIndexOpLowering,
                 RecordExtractOpLowering, RecordGetFieldOpLowering,
                 StoreOpLowering, TensorPackOpLowering, TensorViewOpLowering>(
        typeConverter, &getContext());
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
} // namespace mlir::mulberry_core
