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
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
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

#define GEN_PASS_DEF_FINALIZEMULBERRYTENSORSTORAGE
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

static auto convertRecordLayoutType(RecordType type)
    -> std::optional<Type>;
static auto convertPointerValueType(PtrType type)
    -> std::optional<Type>;

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
static auto convertTensorToMemRefType(TensorType type)
    -> MemRefType {
  return MemRefType::get(convertMemRefShape(type.getShape()),
                         type.getElementType());
}

static auto convertTensorToDataMemRefType(TensorType type)
    -> MemRefType {
  auto layout = MemRefLayoutAttrInterface{};
  return MemRefType::get(convertMemRefShape(type.getShape()),
                         type.getElementType(), layout,
                         getTensorDataAddressSpace(type.getContext()));
}

// This pass is a transitional storage lowering, not the final Mulberry-to-LLVM
// ABI lowering. Domain packages such as mulberry.nn are intentionally not
// lowered here; they live outside core.
static auto convertBackendValueType(Type type) -> std::optional<Type> {
  if (llvm::isa<FunctionType>(type))
    return getPtrType(type.getContext());

  if (auto tensorType = llvm::dyn_cast<TensorType>(type))
    return convertTensorToMemRefType(tensorType);

  if (auto recordType = llvm::dyn_cast<RecordType>(type))
    return convertRecordLayoutType(recordType);

  if (auto ptrType = llvm::dyn_cast<PtrType>(type))
    return convertPointerValueType(ptrType);

  if (isScalarStorageType(type))
    return type;

  return std::nullopt;
}

static auto convertRecordFieldStorageType(Type type) -> std::optional<Type> {
  // Internal tensors have a context-dependent representation and cannot be
  // embedded directly in an ordinary object record layout.
  if (llvm::isa<TensorType>(type))
    return std::nullopt;

  return convertBackendValueType(type);
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

static auto convertToTensorABILayout(TensorType type)
    -> TensorABILayout {
  return convertToTensorABILayout(type.getContext(), type.getShape().size());
}

static auto createMemRefDataPointer(Location location, OpBuilder& rewriter,
                                    Value storage)
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

static auto callRuntimeVoid(Location location, OpBuilder& builder,
                            Operation* op, StringRef name,
                            ValueRange arguments = {}) -> LogicalResult {
  auto moduleOp = op->getParentOfType<ModuleOp>();
  if (!moduleOp)
    return failure();

  std::vector<Type> argumentTypes;
  for (auto argument : arguments)
    argumentTypes.push_back(argument.getType());
  auto function = LLVM::lookupOrCreateFn(builder, moduleOp, name,
      argumentTypes, LLVM::LLVMVoidType::get(builder.getContext()));
  if (failed(function))
    return failure();

  LLVM::CallOp::create(builder, location, *function, arguments);
  return success();
}

static auto createTensorStorageOwner(Location location, OpBuilder& builder,
                                     Operation* op, Type storageLayout,
                                     Value allocated, Value data)
    -> FailureOr<Value> {
  auto storageBytes = createSizeOf(location, builder, storageLayout);
  auto storage = callBoehmMalloc(location, builder, op, storageBytes);
  if (failed(storage))
    return failure();

  auto i1Type = IntegerType::get(builder.getContext(), 1);
  auto falseValue = LLVM::ConstantOp::create(
      builder, location, i1Type, builder.getIntegerAttr(i1Type, 0));
  auto storageValue =
      LLVM::UndefOp::create(builder, location, storageLayout).getResult();
  storageValue = LLVM::InsertValueOp::create(
                     builder, location, storageValue, allocated,
                     ArrayRef<int64_t>{0})
                     .getResult();
  storageValue = LLVM::InsertValueOp::create(
                     builder, location, storageValue, data,
                     ArrayRef<int64_t>{1})
                     .getResult();
  storageValue = LLVM::InsertValueOp::create(
                     builder, location, storageValue, falseValue,
                     ArrayRef<int64_t>{2})
                     .getResult();
  LLVM::StoreOp::create(builder, location, storageValue, *storage);

  if (failed(callRuntimeVoid(location, builder, op,
                             "mulberry_tensor_storage_register",
                             ValueRange{*storage})))
    return failure();
  return *storage;
}

static auto extractRecordField(Location location,
                               ConversionPatternRewriter& rewriter,
                               Value record,
                               RecordType recordType,
                               StringRef field) -> FailureOr<Value> {
  auto sourceFieldType = recordType.getFieldType(field);
  if (!sourceFieldType)
    return failure();

  auto fieldType = convertRecordFieldStorageType(sourceFieldType);
  if (!fieldType)
    return failure();

  auto fieldIndex = static_cast<int64_t>(recordType.getFieldIndex(field));
  return LLVM::ExtractValueOp::create(rewriter, location, *fieldType, record,
                                      ArrayRef<int64_t>{fieldIndex})
      .getResult();
}

static auto getReferencedRecordType(Type type) -> RecordType {
  auto ptrType = llvm::dyn_cast<PtrType>(type);
  if (!ptrType)
    return {};
  return llvm::dyn_cast<RecordType>(ptrType.getPointeeType());
}

static auto createRecordFieldAddress(
    Location location, ConversionPatternRewriter& rewriter, Value record,
    RecordType recordType, StringRef field) -> FailureOr<Value> {
  auto backendType = convertRecordLayoutType(recordType);
  if (!backendType || !recordType.getFieldType(field))
    return failure();

  auto fieldIndex = static_cast<int32_t>(recordType.getFieldIndex(field));
  return LLVM::GEPOp::create(
             rewriter, location, getPtrType(rewriter.getContext()),
             *backendType, record, ArrayRef<LLVM::GEPArg>{0, fieldIndex})
      .getResult();
}

static auto loadTensorStorageReference(
    Location location, ConversionPatternRewriter& rewriter, Value tensor,
    RecordType tensorType) -> FailureOr<Value> {
  auto storageAddress = createRecordFieldAddress(
      location, rewriter, tensor, tensorType, "_storage");
  if (failed(storageAddress))
    return failure();
  return LLVM::LoadOp::create(rewriter, location,
                              getPtrType(rewriter.getContext()),
                              *storageAddress)
      .getResult();
}

static auto getTensorStorageType(RecordType tensorType)
    -> RecordType {
  return getReferencedRecordType(tensorType.getFieldType("_storage"));
}

static auto assertTensorStorageAlive(
    Location location, ConversionPatternRewriter& rewriter, Operation* op,
    Value storage, RecordType storageType) -> LogicalResult {
  auto disposedAddress = createRecordFieldAddress(
      location, rewriter, storage, storageType, "disposed");
  if (failed(disposedAddress))
    return failure();

  auto i1Type = IntegerType::get(rewriter.getContext(), 1);
  auto disposed = LLVM::LoadOp::create(rewriter, location, i1Type,
                                       *disposedAddress);
  auto ifOp = scf::IfOp::create(rewriter, location, disposed,
                                /*withElseRegion=*/false);
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(ifOp.thenBlock());
  if (failed(callRuntimeVoid(location, rewriter, op,
                             "mulberry_tensor_use_after_dispose")))
    return failure();
  return success();
}

static auto disposeTensorStorage(
    Location location, ConversionPatternRewriter& rewriter, Operation* op,
    Value storage, RecordType storageType) -> LogicalResult {
  if (!storageType.getFieldType("allocated") ||
      !storageType.getFieldType("data") ||
      !storageType.getFieldType("disposed"))
    return failure();
  return callRuntimeVoid(location, rewriter, op,
                         "mulberry_tensor_storage_dispose",
                         ValueRange{storage});
}

static auto loadRecordValue(Location location,
                            ConversionPatternRewriter& rewriter,
                            Value recordReference,
                            RecordType recordType)
    -> FailureOr<Value> {
  auto recordBackendType = convertRecordLayoutType(recordType);
  if (!recordBackendType)
    return failure();

  return LLVM::LoadOp::create(rewriter, location, *recordBackendType,
                              recordReference)
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
                                  RecordType listType,
                                  Value length, Value data)
    -> FailureOr<Value> {
  auto listBackendType = convertRecordLayoutType(listType);
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

static auto createHeapRecordReference(Location location,
                                      ConversionPatternRewriter& rewriter,
                                      Operation* op,
                                      RecordType recordType,
                                      Value record) -> FailureOr<Value> {
  auto recordBackendType = convertRecordLayoutType(recordType);
  if (!recordBackendType)
    return failure();

  auto one = LLVM::ConstantOp::create(
      rewriter, location, getI64Type(rewriter.getContext()),
      rewriter.getI64IntegerAttr(1));
  auto recordPtr = createHeapArray(location, rewriter, op, *recordBackendType,
                                   one.getResult());
  if (failed(recordPtr))
    return failure();

  LLVM::StoreOp::create(rewriter, location, record, *recordPtr);
  return *recordPtr;
}

static auto extractListDataPointer(Location location,
                                   ConversionPatternRewriter& rewriter,
                                   Value list,
                                   RecordType listType)
    -> FailureOr<Value> {
  auto data = extractRecordField(location, rewriter, list, listType, "data");
  if (failed(data))
    return failure();
  return *data;
}

static auto createTensorABIDesc(
    Location location, ConversionPatternRewriter& rewriter,
    const TensorABILayout& layout, Value tensor, TensorType type)
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

static auto createTensorABIDescFromRecord(
    Location location, ConversionPatternRewriter& rewriter,
    const TensorABILayout& layout, Value tensorRecord,
    RecordType tensorRecordType,
    TensorType tensorType)
    -> FailureOr<Value> {
  auto data = extractRecordField(location, rewriter, tensorRecord,
                                 tensorRecordType, "data");
  auto sizes = extractRecordField(location, rewriter, tensorRecord,
                                  tensorRecordType, "sizes");
  auto strides =
      extractRecordField(location, rewriter, tensorRecord, tensorRecordType,
                         "strides");
  if (failed(data) || failed(sizes) || failed(strides))
    return failure();

  auto sizesType =
      getReferencedRecordType(tensorRecordType.getFieldType("sizes"));
  auto stridesType =
      getReferencedRecordType(tensorRecordType.getFieldType("strides"));
  if (!sizesType || !stridesType)
    return failure();
  auto sizesValue = loadRecordValue(location, rewriter, *sizes, sizesType);
  auto stridesValue =
      loadRecordValue(location, rewriter, *strides, stridesType);
  if (failed(sizesValue) || failed(stridesValue))
    return failure();
  auto sizesData =
      extractListDataPointer(location, rewriter, *sizesValue, sizesType);
  auto stridesData =
      extractListDataPointer(location, rewriter, *stridesValue, stridesType);
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
    TensorType type) -> Value {
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

static auto createTensorFromFlatStorage(
    Location location, ConversionPatternRewriter& rewriter, Value storage,
    Value desc, TensorType type) -> Value {
  auto context = rewriter.getContext();
  auto resultMemRefType = convertTensorToMemRefType(type);

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

  std::vector<int64_t> dynamicShape(resultMemRefType.getRank(),
                                    ShapedType::kDynamic);
  std::vector<int64_t> dynamicStrides(resultMemRefType.getRank(),
                                      ShapedType::kDynamic);
  auto stridedType = MemRefType::get(
      dynamicShape, resultMemRefType.getElementType(),
      StridedLayoutAttr::get(context, ShapedType::kDynamic, dynamicStrides));
  auto stridedTensor = memref::ReinterpretCastOp::create(
      rewriter, location, stridedType, storage, offset, sizes, strides,
      ArrayRef<NamedAttribute>{});
  return memref::CastOp::create(rewriter, location, resultMemRefType,
                                stridedTensor.getResult());
}

static auto convertRecordLayoutType(RecordType type)
    -> std::optional<Type> {
  std::vector<Type> fieldTypes;
  for (auto field : type.getFields()) {
    auto fieldType = convertRecordFieldStorageType(field.type);
    if (!fieldType)
      return std::nullopt;
    fieldTypes.push_back(*fieldType);
  }

  return LLVM::LLVMStructType::getLiteral(type.getContext(), fieldTypes);
}

static auto convertPointerValueType(PtrType type)
    -> std::optional<Type> {
  auto pointeeType = type.getPointeeType();
  if (llvm::isa<TensorType>(pointeeType) ||
      !convertBackendValueType(pointeeType))
    return std::nullopt;

  return getPtrType(type.getContext());
}

// Safety net for Mulberry types that should not fall through to the identity
// conversion after their specific lowering conversion fails.
static auto rejectUnloweredMulberryType(Type type, SmallVectorImpl<Type>&)
    -> std::optional<LogicalResult> {
  if (auto recordType = llvm::dyn_cast<RecordType>(type))
    if (!convertRecordLayoutType(recordType))
      return failure();

  if (auto ptrType = llvm::dyn_cast<PtrType>(type))
    if (!convertPointerValueType(ptrType))
      return failure();

  return std::nullopt;
}

class MulberryTypeConverter : public TypeConverter {
public:
  MulberryTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion(rejectUnloweredMulberryType);
    addConversion(convertRecordLayoutType);
    addConversion(convertPointerValueType);
    addConversion(convertTensorToMemRefType);
    addConversion([this](FunctionType type) -> std::optional<Type> {
      SmallVector<Type> inputs;
      SmallVector<Type> results;
      if (failed(convertTypes(type.getInputs(), inputs)) ||
          failed(convertTypes(type.getResults(), results)))
        return std::nullopt;
      return FunctionType::get(type.getContext(), inputs, results);
    });
    // Keep unsupported Mulberry types illegal until each one has a real
    // lowering. The identity conversion above is only for non-Mulberry types.
  }
};

class FunctionConstantTypeConversion
    : public OpConversionPattern<func::ConstantOp> {
public:
  using OpConversionPattern<func::ConstantOp>::OpConversionPattern;

  auto matchAndRewrite(func::ConstantOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto convertedType = llvm::dyn_cast_or_null<FunctionType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    if (!convertedType)
      return rewriter.notifyMatchFailure(
          op, "function constant needs a converted function type");

    auto constant = func::ConstantOp::create(
        rewriter, op.getLoc(), convertedType, op.getValueAttr());
    rewriter.replaceOp(op, constant.getResult());
    return success();
  }
};

class CallIndirectTypeConversion
    : public OpConversionPattern<func::CallIndirectOp> {
public:
  using OpConversionPattern<func::CallIndirectOp>::OpConversionPattern;

  auto matchAndRewrite(func::CallIndirectOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    if (!llvm::isa<FunctionType>(adaptor.getCallee().getType()))
      return rewriter.notifyMatchFailure(
          op, "indirect callee needs a converted function type");

    auto call = func::CallIndirectOp::create(
        rewriter, op.getLoc(), adaptor.getCallee(),
        adaptor.getCalleeOperands());
    call->setAttrs(op->getAttrDictionary());
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

// MLIR's generic call conversion rebuilds func.call without copying extra
// attributes, so preserve the ownership contract before converting its type.
// https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Func/Transforms/FuncConversions.cpp
class TensorOwnershipCallOpLowering
    : public OpConversionPattern<func::CallOp> {
public:
  using OpConversionPattern<func::CallOp>::OpConversionPattern;

  auto matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    if (!op->hasAttr(kTransferTensorResultOwnershipAttr))
      return failure();

    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(),
                                                resultTypes)) ||
        resultTypes.size() != op.getNumResults())
      return rewriter.notifyMatchFailure(
          op, "Tensor ownership call needs one-to-one result conversion");

    auto call = func::CallOp::create(rewriter, op.getLoc(), op.getCallee(),
                                     resultTypes, adaptor.getOperands());
    call->setAttrs(op->getAttrDictionary());
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

class AllocaOpLowering : public OpConversionPattern<AllocaOp> {
public:
  using OpConversionPattern<AllocaOp>::OpConversionPattern;

  auto matchAndRewrite(AllocaOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    if (llvm::isa<TensorType>(op.getElementType()))
      return rewriter.notifyMatchFailure(
          op, "internal tensors cannot use pointer storage");

    auto storageType = convertBackendValueType(op.getElementType());
    if (!storageType)
      return rewriter.notifyMatchFailure(
          op, "alloca needs a lowerable storage type");

    auto elementCount = arith::ConstantIntOp::create(
        rewriter, op.getLoc(), 1, /*width=*/64);
    auto alloca = LLVM::AllocaOp::create(
        rewriter, op.getLoc(), getPtrType(op.getContext()), *storageType,
        elementCount.getResult(), /*alignment=*/0);
    rewriter.replaceOp(op, alloca.getResult());
    return success();
  }
};

class TensorStorageAllocOpLowering
    : public OpConversionPattern<TensorStorageAllocOp> {
public:
  using OpConversionPattern<TensorStorageAllocOp>::OpConversionPattern;

  auto matchAndRewrite(TensorStorageAllocOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto storagePtrType = llvm::dyn_cast<PtrType>(op.getStorage().getType());
    auto storageType = storagePtrType
                           ? llvm::dyn_cast<RecordType>(
                                 storagePtrType.getPointeeType())
                           : RecordType{};
    auto storageLayout = storageType
                             ? convertRecordLayoutType(storageType)
                             : std::optional<Type>{};
    auto payloadType = llvm::dyn_cast<TensorType>(op.getPayload().getType());
    if (!storageLayout || !payloadType)
      return rewriter.notifyMatchFailure(
          op, "Tensor storage allocation needs lowerable result types");

    auto memRefType = convertTensorToMemRefType(payloadType);
    auto allocation = TensorStorageAllocLoweredOp::create(
        rewriter, op.getLoc(),
        TypeRange{getPtrType(op.getContext()), memRefType}, *storageLayout,
        op.getElementType(), adaptor.getCount());
    allocation->setAttr(
        bufferization::BufferizationDialect::kManualDeallocation,
        rewriter.getUnitAttr());
    rewriter.replaceOp(op, allocation.getResults());
    return success();
  }
};

class HeapAllocOpLowering : public OpConversionPattern<HeapAllocOp> {
public:
  using OpConversionPattern<HeapAllocOp>::OpConversionPattern;

  auto matchAndRewrite(HeapAllocOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    // Internal tensors lower to memrefs and are never heap-allocated through
    // Mulberry pointer storage.
    if (llvm::isa<TensorType>(op.getElementType()))
      return rewriter.notifyMatchFailure(
          op, "internal tensors require memref allocation");

    auto storageType = convertBackendValueType(op.getElementType());
    if (!storageType)
      return rewriter.notifyMatchFailure(
          op, "heap alloc needs a lowerable backend type");

    auto elementBytes = createSizeOf(op.getLoc(), rewriter, *storageType);
    auto sizeInBytes = arith::MulIOp::create(
        rewriter, op.getLoc(), elementBytes, adaptor.getCount());
    auto heapPtr = callBoehmMalloc(op.getLoc(), rewriter, op,
                                   sizeInBytes.getResult());
    if (failed(heapPtr))
      return failure();

    rewriter.replaceOp(op, *heapPtr);
    return success();
  }
};

class PtrIndexOpLowering : public OpConversionPattern<PtrIndexOp> {
public:
  using OpConversionPattern<PtrIndexOp>::OpConversionPattern;

  auto matchAndRewrite(PtrIndexOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto ptrType = llvm::cast<PtrType>(op.getPtr().getType());
    auto pointeeType = ptrType.getPointeeType();
    if (llvm::isa<TensorType>(pointeeType))
      return rewriter.notifyMatchFailure(
          op, "internal tensors cannot use pointer storage");

    auto storageType = convertBackendValueType(pointeeType);
    if (!storageType)
      return rewriter.notifyMatchFailure(
          op, "pointer index needs a lowerable backend element type");

    auto index = arith::IndexCastOp::create(
        rewriter, op.getLoc(), getI64Type(op.getContext()),
        adaptor.getIndex());
    auto elementPtr = LLVM::GEPOp::create(
        rewriter, op.getLoc(), getPtrType(op.getContext()), *storageType,
        adaptor.getPtr(), ArrayRef<Value>{index.getResult()});
    rewriter.replaceOp(op, elementPtr.getResult());
    return success();
  }
};

class PtrCastOpLowering : public OpConversionPattern<PtrCastOp> {
public:
  using OpConversionPattern<PtrCastOp>::OpConversionPattern;

  auto matchAndRewrite(PtrCastOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    rewriter.replaceOp(op, adaptor.getPtr());
    return success();
  }
};

class LoadOpLowering : public OpConversionPattern<LoadOp> {
public:
  using OpConversionPattern<LoadOp>::OpConversionPattern;

  auto matchAndRewrite(LoadOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto ptrType = llvm::cast<PtrType>(op.getPtr().getType());
    auto pointeeType = ptrType.getPointeeType();
    if (llvm::isa<TensorType>(pointeeType))
      return rewriter.notifyMatchFailure(
          op, "internal tensors cannot use pointer storage");

    auto valueType = convertBackendValueType(pointeeType);
    if (!valueType)
      return rewriter.notifyMatchFailure(
          op, "load needs a lowerable result type");

    auto load = LLVM::LoadOp::create(rewriter, op.getLoc(), *valueType,
                                     adaptor.getPtr());
    if (llvm::isa<FunctionType>(pointeeType)) {
      // FuncToLLVM later turns the SSA function value into the same opaque
      // pointer already used by LLVM storage; reconciliation removes this cast.
      auto sourceType = getTypeConverter()->convertType(pointeeType);
      auto cast = UnrealizedConversionCastOp::create(
          rewriter, op.getLoc(), sourceType, load.getResult());
      rewriter.replaceOp(op, cast.getResult(0));
      return success();
    }

    rewriter.replaceOp(op, load.getResult());
    return success();
  }
};

class StoreOpLowering : public OpConversionPattern<StoreOp> {
public:
  using OpConversionPattern<StoreOp>::OpConversionPattern;

  auto matchAndRewrite(StoreOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto ptrType = llvm::dyn_cast<PtrType>(op.getPtr().getType());
    if (!ptrType)
      return rewriter.notifyMatchFailure(
          op, "store target must be a Mulberry pointer");

    auto pointeeType = ptrType.getPointeeType();
    if (llvm::isa<TensorType>(pointeeType))
      return rewriter.notifyMatchFailure(
          op, "internal tensors cannot use pointer storage");

    auto valueType = convertBackendValueType(pointeeType);
    if (!valueType)
      return rewriter.notifyMatchFailure(
          op, "store target must be a lowered storage slot");

    auto value = adaptor.getValue();
    if (llvm::isa<FunctionType>(pointeeType)) {
      // This is the inverse bridge of the function-value load above.
      value = UnrealizedConversionCastOp::create(
                  rewriter, op.getLoc(), *valueType, value)
                  .getResult(0);
    }
    LLVM::StoreOp::create(rewriter, op.getLoc(), value, adaptor.getPtr());
    rewriter.eraseOp(op);
    return success();
  }
};

class RecordGetFieldOpLowering : public OpConversionPattern<RecordGetFieldOp> {
public:
  using OpConversionPattern<RecordGetFieldOp>::OpConversionPattern;

  auto matchAndRewrite(RecordGetFieldOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto ptrType = llvm::dyn_cast<PtrType>(op.getRecord().getType());
    if (!ptrType)
      return rewriter.notifyMatchFailure(
          op, "field address needs a pointer to a record");

    auto recordType =
        llvm::dyn_cast<RecordType>(ptrType.getPointeeType());
    if (!recordType)
      return rewriter.notifyMatchFailure(
          op, "field address needs a pointer to a record");

    auto recordBackendType = convertRecordLayoutType(recordType);
    auto fieldType = convertRecordFieldStorageType(
        recordType.getFieldType(op.getField()));
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

class TensorViewOpLowering : public OpConversionPattern<TensorViewOp> {
public:
  using OpConversionPattern<TensorViewOp>::OpConversionPattern;

  auto matchAndRewrite(TensorViewOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto tensorRecordType = llvm::dyn_cast<RecordType>(
        op.getTensorRecord().getType());
    if (!tensorRecordType)
      return rewriter.notifyMatchFailure(
          op, "tensor view needs a Tensor record");

    auto storageType = getTensorStorageType(tensorRecordType);
    auto storage = extractRecordField(op.getLoc(), rewriter,
                                      adaptor.getTensorRecord(),
                                      tensorRecordType, "_storage");
    if (!storageType || failed(storage) ||
        failed(assertTensorStorageAlive(op.getLoc(), rewriter, op,
                                        *storage, storageType)))
      return rewriter.notifyMatchFailure(
          op, "tensor view needs live shared storage");

    auto tensorType = llvm::cast<TensorType>(
        op.getResult().getType());
    auto layout = convertToTensorABILayout(tensorType);
    auto desc = createTensorABIDescFromRecord(
        op.getLoc(), rewriter, layout, adaptor.getTensorRecord(),
        tensorRecordType, tensorType);
    if (failed(desc))
      return rewriter.notifyMatchFailure(
          op, "tensor view needs Tensor-compatible fields");

    auto tensor = createTensorFromABIDesc(op.getLoc(), rewriter, *desc,
                                          tensorType);
    rewriter.replaceOp(op, tensor);
    return success();
  }
};

class TensorPackOpLowering : public OpConversionPattern<TensorPackOp> {
public:
  using OpConversionPattern<TensorPackOp>::OpConversionPattern;

  auto matchAndRewrite(TensorPackOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto tensorRecordType = llvm::dyn_cast<RecordType>(
        op.getTensorRecord().getType());
    if (!tensorRecordType)
      return rewriter.notifyMatchFailure(
          op, "tensor pack needs a Tensor record result");

    auto tensorRecordBackendType = convertRecordLayoutType(tensorRecordType);
    if (!tensorRecordBackendType)
      return rewriter.notifyMatchFailure(
          op, "tensor pack needs a lowerable Tensor record");

    auto tensorType = llvm::cast<TensorType>(
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

    auto sizesType =
        getReferencedRecordType(tensorRecordType.getFieldType("sizes"));
    auto stridesType =
        getReferencedRecordType(tensorRecordType.getFieldType("strides"));
    auto storageType = getTensorStorageType(tensorRecordType);
    if (!sizesType || !stridesType || !storageType)
      return rewriter.notifyMatchFailure(
          op, "tensor pack needs metadata and shared storage fields");
    auto storageBackendType = convertRecordLayoutType(storageType);
    if (!storageBackendType)
      return rewriter.notifyMatchFailure(
          op, "tensor pack needs lowerable shared storage");

    auto elementBytes =
        createSizeOf(op.getLoc(), rewriter, tensorType.getElementType());
    auto elementCount = arith::DivUIOp::create(
        rewriter, op.getLoc(), *payloadBytes, elementBytes);

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

    auto transferOwnership =
        op->hasAttr(kTransferTensorResultOwnershipAttr);
    if (transferOwnership) {
      auto producer = op.getTensor().getDefiningOp<func::CallOp>();
      if (!producer || producer.getNumResults() != 1 ||
          !producer->hasAttr(kTransferTensorResultOwnershipAttr))
        return rewriter.notifyMatchFailure(
            op, "ownership-transferring pack needs a marked single-result "
                "call");
      if (!op.getTensor().hasOneUse())
        return rewriter.notifyMatchFailure(
            op, "ownership-transferring call result must only feed its pack");
    }

    Value storage;
    Value payloadPointer;
    if (transferOwnership) {
      // NN result allocations use memref.alloc without custom alignment, so
      // their allocated and aligned pointers coincide.
      payloadPointer = createMemRefDataPointer(op.getLoc(), rewriter, tensor);
      auto owner = createTensorStorageOwner(
          op.getLoc(), rewriter, op, *storageBackendType, payloadPointer,
          payloadPointer);
      if (failed(owner))
        return failure();
      storage = *owner;
    } else {
      auto flatMemRefType = MemRefType::get(
          {ShapedType::kDynamic}, tensorType.getElementType());
      auto allocation = TensorStorageAllocLoweredOp::create(
          rewriter, op.getLoc(),
          TypeRange{getPtrType(context), flatMemRefType}, *storageBackendType,
          tensorType.getElementType(), elementCount.getResult());
      allocation->setAttr(
          bufferization::BufferizationDialect::kManualDeallocation,
          rewriter.getUnitAttr());

      auto ownedTensor = createTensorFromFlatStorage(
          op.getLoc(), rewriter, allocation.getPayload(), sourceDesc,
          tensorType);
      memref::CopyOp::create(rewriter, op.getLoc(), tensor, ownedTensor);

      auto payloadAddress = createRecordFieldAddress(
          op.getLoc(), rewriter, allocation.getStorage(), storageType,
          "data");
      if (failed(payloadAddress))
        return failure();
      payloadPointer =
          LLVM::LoadOp::create(rewriter, op.getLoc(), getPtrType(context),
                               *payloadAddress)
              .getResult();
      storage = allocation.getStorage();
    }

    auto sizesList = createListRecordValue(
        op.getLoc(), rewriter, sizesType, rankValue.getResult(), *sizes);
    auto stridesList = createListRecordValue(
        op.getLoc(), rewriter, stridesType, rankValue.getResult(), *strides);
    if (failed(sizesList) || failed(stridesList))
      return failure();
    // The package bridge returns a Tensor header by value, while its metadata
    // fields keep normal source-object reference semantics.
    auto sizesReference = createHeapRecordReference(
        op.getLoc(), rewriter, op, sizesType, *sizesList);
    auto stridesReference = createHeapRecordReference(
        op.getLoc(), rewriter, op, stridesType, *stridesList);
    if (failed(sizesReference) || failed(stridesReference))
      return failure();

    auto tensorRecord = LLVM::UndefOp::create(rewriter, op.getLoc(),
                                              *tensorRecordBackendType);
    Value current = tensorRecord.getResult();
    auto storageIndex =
        static_cast<int64_t>(tensorRecordType.getFieldIndex("_storage"));
    auto dataIndex =
        static_cast<int64_t>(tensorRecordType.getFieldIndex("data"));
    auto rankIndex =
        static_cast<int64_t>(tensorRecordType.getFieldIndex("rank"));
    auto numelIndex =
        static_cast<int64_t>(tensorRecordType.getFieldIndex("numel"));
    auto sizesIndex =
        static_cast<int64_t>(tensorRecordType.getFieldIndex("sizes"));
    auto stridesIndex =
        static_cast<int64_t>(tensorRecordType.getFieldIndex("strides"));
    current = LLVM::InsertValueOp::create(
        rewriter, op.getLoc(), current, storage,
        ArrayRef<int64_t>{storageIndex});
    current = LLVM::InsertValueOp::create(
        rewriter, op.getLoc(), current, payloadPointer,
        ArrayRef<int64_t>{dataIndex});
    current = LLVM::InsertValueOp::create(
        rewriter, op.getLoc(), current, rankValue.getResult(),
        ArrayRef<int64_t>{rankIndex});
    current = LLVM::InsertValueOp::create(
        rewriter, op.getLoc(), current, elementCount.getResult(),
        ArrayRef<int64_t>{numelIndex});
    current = LLVM::InsertValueOp::create(
        rewriter, op.getLoc(), current, *sizesReference,
        ArrayRef<int64_t>{sizesIndex});
    current = LLVM::InsertValueOp::create(
        rewriter, op.getLoc(), current, *stridesReference,
        ArrayRef<int64_t>{stridesIndex});

    if (!current)
      return rewriter.notifyMatchFailure(
          op, "tensor pack needs Tensor-compatible fields");

    rewriter.replaceOp(op, current);
    return success();
  }
};

class TensorDisposeOpLowering : public OpConversionPattern<TensorDisposeOp> {
public:
  using OpConversionPattern<TensorDisposeOp>::OpConversionPattern;

  auto matchAndRewrite(TensorDisposeOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto tensorPtrType = llvm::dyn_cast<PtrType>(
        op.getTensor().getType());
    auto tensorType = tensorPtrType
                          ? llvm::dyn_cast<RecordType>(
                                tensorPtrType.getPointeeType())
                          : RecordType{};
    auto storageType = tensorType ? getTensorStorageType(tensorType)
                                  : RecordType{};
    if (!tensorType || !storageType)
      return rewriter.notifyMatchFailure(
          op, "tensor dispose needs a Tensor object reference");

    auto storage = loadTensorStorageReference(
        op.getLoc(), rewriter, adaptor.getTensor(), tensorType);
    if (failed(storage) ||
        failed(disposeTensorStorage(op.getLoc(), rewriter, op,
                                    *storage, storageType)))
      return rewriter.notifyMatchFailure(
          op, "tensor dispose needs lowerable shared storage");
    rewriter.eraseOp(op);
    return success();
  }
};

class TensorAssertAliveOpLowering
    : public OpConversionPattern<TensorAssertAliveOp> {
public:
  using OpConversionPattern<
      TensorAssertAliveOp>::OpConversionPattern;

  auto matchAndRewrite(TensorAssertAliveOp op,
                       OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto tensorPtrType = llvm::dyn_cast<PtrType>(
        op.getTensor().getType());
    auto tensorType = tensorPtrType
                          ? llvm::dyn_cast<RecordType>(
                                tensorPtrType.getPointeeType())
                          : RecordType{};
    auto storageType = tensorType ? getTensorStorageType(tensorType)
                                  : RecordType{};
    if (!tensorType || !storageType)
      return rewriter.notifyMatchFailure(
          op, "tensor assertion needs a Tensor object reference");

    auto storage = loadTensorStorageReference(
        op.getLoc(), rewriter, adaptor.getTensor(), tensorType);
    if (failed(storage) ||
        failed(assertTensorStorageAlive(op.getLoc(), rewriter, op,
                                        *storage, storageType)))
      return rewriter.notifyMatchFailure(
          op, "tensor assertion needs lowerable shared storage");
    rewriter.eraseOp(op);
    return success();
  }
};

class FinalizeTensorStorageAllocPattern
    : public OpRewritePattern<TensorStorageAllocLoweredOp> {
public:
  using OpRewritePattern<TensorStorageAllocLoweredOp>::OpRewritePattern;

  auto matchAndRewrite(TensorStorageAllocLoweredOp op,
                       PatternRewriter& rewriter) const
      -> LogicalResult final {
    auto memRefType = llvm::dyn_cast<MemRefType>(op.getPayload().getType());
    auto storageLayout =
        llvm::dyn_cast<LLVM::LLVMStructType>(op.getStorageLayout());
    if (!memRefType || !storageLayout)
      return rewriter.notifyMatchFailure(
          op, "final Tensor storage needs memref and LLVM storage types");

    auto count = arith::IndexCastOp::create(
        rewriter, op.getLoc(), rewriter.getIndexType(), op.getCount());
    auto payload = memref::AllocOp::create(
        rewriter, op.getLoc(), memRefType, ValueRange{count.getResult()});
    auto data = createMemRefDataPointer(op.getLoc(), rewriter,
                                        payload.getResult());

    // This path uses memref.alloc without custom alignment, so its allocated
    // and aligned pointers coincide. Keep both slots explicit in the owner.
    auto storage = createTensorStorageOwner(
        op.getLoc(), rewriter, op, storageLayout, data, data);
    if (failed(storage))
      return failure();

    rewriter.replaceOp(op, ValueRange{*storage, payload.getResult()});
    return success();
  }
};

struct FinalizeMulberryTensorStorage
    : public impl::FinalizeMulberryTensorStorageBase<
          FinalizeMulberryTensorStorage> {
  using impl::FinalizeMulberryTensorStorageBase<
      FinalizeMulberryTensorStorage>::FinalizeMulberryTensorStorageBase;

  auto runOnOperation() -> void final {
    RewritePatternSet patterns(&getContext());
    patterns.add<FinalizeTensorStorageAllocPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
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
    target.addLegalOp<TensorStorageAllocLoweredOp>();
    target.addIllegalDialect<MulberryDialect>();
    target.addDynamicallyLegalOp<func::FuncOp>(
        [&](func::FuncOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType());
        });
    target.addDynamicallyLegalOp<func::CallOp, func::ReturnOp>(
        [&](Operation* op) {
          return typeConverter.isLegal(op);
        });
    target.addDynamicallyLegalOp<func::ConstantOp, func::CallIndirectOp>(
        [&](Operation* op) {
          return typeConverter.isLegal(op);
        });

    RewritePatternSet patterns(&getContext());
    patterns.add<AllocaOpLowering, CallIndirectTypeConversion,
                 FunctionConstantTypeConversion, HeapAllocOpLowering,
                 LoadOpLowering, PtrCastOpLowering, PtrIndexOpLowering,
                 RecordGetFieldOpLowering, StoreOpLowering,
                 TensorStorageAllocOpLowering,
                 TensorAssertAliveOpLowering, TensorDisposeOpLowering,
                 TensorPackOpLowering, TensorViewOpLowering>(
        typeConverter, &getContext());
    patterns.add<TensorOwnershipCallOpLowering>(
        typeConverter, &getContext(), PatternBenefit(2));
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
