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
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <optional>
#include <string>

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

static auto getI32Type(MLIRContext* context) -> IntegerType {
  return IntegerType::get(context, 32);
}

static auto getTensorDataAddressSpace(MLIRContext* context) -> Attribute {
  return LLVM::AddressSpaceAttr::get(context, 0);
}

static auto getTensorDataPtrType(MLIRContext* context) -> ptr::PtrType {
  auto addressSpace = llvm::cast<ptr::MemorySpaceAttrInterface>(
      getTensorDataAddressSpace(context));
  return ptr::PtrType::get(addressSpace);
}

static auto isScalarStorageType(Type type) -> bool {
  return type.isIndex() || llvm::isa<IntegerType, FloatType>(type);
}

using StringABIDescriptorType = LLVM::LLVMStructType;

// String ABI descriptor layout:
//
//   { length: i64, data: ptr }
//
// `length` is the source string byte count and excludes the trailing NUL.
// String literals still materialize immutable global bytes with an extra NUL
// so future runtime calls can pass the same data pointer to C APIs.
struct StringABILayout {
  StringABIDescriptorType descriptorType;
};

static auto convertToStringABILayout(mulberry::StringType type)
    -> StringABILayout {
  auto context = type.getContext();

  std::vector<Type> fields;
  fields.push_back(getI64Type(context)); // byte length, excluding trailing NUL
  fields.push_back(getPtrType(context)); // data pointer
  auto descriptorType = LLVM::LLVMStructType::getLiteral(context, fields);

  return StringABILayout{descriptorType};
}

static auto convertRecordToBackendType(mulberry::RecordType type)
    -> std::optional<Type>;
static auto convertToListABIElementType(Type type) -> std::optional<Type>;

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

static auto convertTensorToDataMemRefType(mulberry::TensorType type)
    -> MemRefType {
  auto layout = MemRefLayoutAttrInterface{};
  return MemRefType::get(convertMemRefShape(type.getShape()),
                         type.getElementType(), layout,
                         getTensorDataAddressSpace(type.getContext()));
}

static auto convertMemRefToDataMemRefType(MemRefType type) -> MemRefType {
  auto layout = type.getLayout();
  return MemRefType::get(type.getShape(), type.getElementType(),
                         layout,
                         getTensorDataAddressSpace(type.getContext()));
}

// This pass is a transitional storage lowering, not the final
// Mulberry-to-LLVM ABI lowering. Tensor values become memrefs so cherry_nn can
// lower to linalg, while scalar/record stack storage still uses LLVM dialect.
static auto convertToBackendType(Type type) -> std::optional<Type> {
  if (auto recordType = llvm::dyn_cast<mulberry::RecordType>(type))
    return convertRecordToBackendType(recordType);

  if (auto stringType = llvm::dyn_cast<mulberry::StringType>(type))
    return convertToStringABILayout(stringType).descriptorType;

  if (llvm::isa<mulberry::FileType>(type))
    return getPtrType(type.getContext());

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

static auto isEscapableListStorage(Type type) -> bool {
  auto storageType = llvm::dyn_cast<mulberry::ListStorageType>(type);
  return storageType &&
         convertToListABIElementType(storageType.getElementType()).has_value();
}

using TensorABIDescriptorType = LLVM::LLVMStructType;

// Tensor ABI descriptor layout:
//
//   { data: ptr<#llvm.address_space<0>>,
//     sizes: array<rank x i64>,
//     strides: array<rank x i64> }
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
  fields.push_back(getTensorDataPtrType(context)); // data pointer
  fields.push_back(indexArrayType);                // sizes[rank]
  fields.push_back(indexArrayType);                // strides[rank]
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
    auto sizeI64 = arith::IndexCastOp::create(rewriter, location, i64Type,
                                              size.getResult());
    elementCount = arith::MulIOp::create(rewriter, location, elementCount,
                                         sizeI64.getResult());
  }

  // File IO uses raw bytes. Compute the element ABI size through LLVM
  // datalayout instead of hardcoding language type sizes here.
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

static auto createEscapedListData(
    Location location, ConversionPatternRewriter& rewriter, Operation* op,
    mulberry::ListStorageType storageType, Value storage, Value length)
    -> FailureOr<Value> {
  auto elementType = convertToListABIElementType(storageType.getElementType());
  if (!elementType)
    return failure();

  auto sourceType = storage.getType();
  if (!llvm::isa<MemRefType, LLVM::LLVMPointerType>(sourceType))
    return failure();

  auto lengthI64 = arith::IndexCastOp::create(
      rewriter, location, getI64Type(rewriter.getContext()), length);
  auto elementBytes = createSizeOf(location, rewriter, *elementType);
  auto totalBytes = arith::MulIOp::create(
      rewriter, location, lengthI64.getResult(), elementBytes);
  auto heapStorage = callBoehmMalloc(location, rewriter, op,
                                     totalBytes.getResult());
  if (failed(heapStorage))
    return failure();

  auto zero = arith::ConstantIndexOp::create(rewriter, location, 0);
  auto one = arith::ConstantIndexOp::create(rewriter, location, 1);
  // Return descriptors use Boehm-managed storage so callers do not need an
  // explicit ownership convention. Scalar list storage starts as a memref, while
  // TensorDesc list storage starts as an LLVM pointer to stack alloca; both must
  // be copied into the same ABI data buffer.
  scf::ForOp::create(
      rewriter, location, zero, length, one, ValueRange{},
      [&](OpBuilder& builder, Location loopLoc, Value index, ValueRange args) {
        auto indexI64 = arith::IndexCastOp::create(
            builder, loopLoc, getI64Type(builder.getContext()), index);
        Value value;
        if (llvm::isa<MemRefType>(sourceType)) {
          value = memref::LoadOp::create(
              builder, loopLoc, storage, index).getResult();
        } else {
          auto sourcePtr = LLVM::GEPOp::create(
              builder, loopLoc, getPtrType(builder.getContext()), *elementType,
              storage, ArrayRef<Value>{indexI64.getResult()});
          value = LLVM::LoadOp::create(builder, loopLoc, *elementType,
                                       sourcePtr.getResult()).getResult();
        }
        auto destPtr = LLVM::GEPOp::create(
            builder, loopLoc, getPtrType(builder.getContext()), *elementType,
            *heapStorage, ArrayRef<Value>{indexI64.getResult()});
        LLVM::StoreOp::create(builder, loopLoc, value, destPtr.getResult());
        scf::YieldOp::create(builder, loopLoc);
      });

  return *heapStorage;
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

static auto createStringABIDesc(
    Location location, ConversionPatternRewriter& rewriter,
    const StringABILayout& layout, Value length, Value dataPointer) -> Value {
  auto desc = LLVM::UndefOp::create(rewriter, location,
                                    layout.descriptorType);
  auto withLength = LLVM::InsertValueOp::create(
      rewriter, location, desc.getResult(), length, ArrayRef<int64_t>{0});
  auto withData = LLVM::InsertValueOp::create(
      rewriter, location, withLength.getResult(), dataPointer,
      ArrayRef<int64_t>{1});
  return withData.getResult();
}

static auto extractStringDataPointer(Location location, OpBuilder& builder,
                                     Value stringDesc) -> Value {
  return LLVM::ExtractValueOp::create(builder, location,
                                      getPtrType(builder.getContext()),
                                      stringDesc, ArrayRef<int64_t>{1})
      .getResult();
}

static auto createStringGlobalName(ModuleOp moduleOp) -> std::string {
  for (size_t index = 0;; ++index) {
    auto name = "__mulberry_string_" + std::to_string(index);
    if (!moduleOp.lookupSymbol<LLVM::GlobalOp>(name))
      return name;
  }
}

static auto createStringDataPointer(
    Location location, OpBuilder& builder, Operation* op, StringRef value)
    -> FailureOr<Value> {
  auto moduleOp = op->getParentOfType<ModuleOp>();
  if (!moduleOp)
    return failure();

  auto bytes = value.str();
  bytes.push_back('\0');
  auto name = createStringGlobalName(moduleOp);
  auto context = builder.getContext();

  LLVM::GlobalOp global;
  {
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(moduleOp.getBody());
    auto type = LLVM::LLVMArrayType::get(IntegerType::get(context, 8),
                                         bytes.size());
    global = LLVM::GlobalOp::create(builder, location, type,
                                    /*isConstant=*/true,
                                    LLVM::Linkage::Internal, name,
                                    builder.getStringAttr(bytes),
                                    /*alignment=*/0);
  }

  auto globalPtr = LLVM::AddressOfOp::create(builder, location, global);
  auto zero = LLVM::ConstantOp::create(builder, location, getI64Type(context),
                                       builder.getI64IntegerAttr(0));
  auto dataPointer = LLVM::GEPOp::create(
      builder, location, getPtrType(context), global.getType(), globalPtr,
      ArrayRef<Value>{zero.getResult(), zero.getResult()});
  return dataPointer.getResult();
}

static auto createI64StackArray(Location location,
                                ConversionPatternRewriter& rewriter,
                                size_t length) -> Value {
  auto context = rewriter.getContext();
  auto count = LLVM::ConstantOp::create(
      rewriter, location, getI64Type(context),
      rewriter.getI64IntegerAttr(length));
  return LLVM::AllocaOp::create(
      rewriter, location, getPtrType(context), getI64Type(context),
      count.getResult(), /*alignment=*/0).getResult();
}

static auto createExpectedShapeArray(Location location,
                                     ConversionPatternRewriter& rewriter,
                                     ArrayRef<int64_t> shape) -> Value {
  auto context = rewriter.getContext();
  auto i64Type = getI64Type(context);
  auto shapeArray = createI64StackArray(location, rewriter, shape.size());

  for (size_t dim = 0; dim < shape.size(); ++dim) {
    auto index = LLVM::ConstantOp::create(
        rewriter, location, i64Type,
        rewriter.getI64IntegerAttr(static_cast<int64_t>(dim)));
    auto slot = LLVM::GEPOp::create(
        rewriter, location, getPtrType(context), i64Type, shapeArray,
        ArrayRef<Value>{index.getResult()});
    auto value = LLVM::ConstantOp::create(
        rewriter, location, i64Type, rewriter.getI64IntegerAttr(shape[dim]));
    LLVM::StoreOp::create(rewriter, location, value.getResult(),
                          slot.getResult());
  }

  return shapeArray;
}

static auto loadRuntimeShapeDim(Location location,
                                ConversionPatternRewriter& rewriter,
                                Value shapeArray, size_t dim) -> Value {
  auto context = rewriter.getContext();
  auto i64Type = getI64Type(context);
  auto index = LLVM::ConstantOp::create(
      rewriter, location, i64Type,
      rewriter.getI64IntegerAttr(static_cast<int64_t>(dim)));
  auto slot = LLVM::GEPOp::create(
      rewriter, location, getPtrType(context), i64Type, shapeArray,
      ArrayRef<Value>{index.getResult()});
  return LLVM::LoadOp::create(rewriter, location, i64Type,
                              slot.getResult()).getResult();
}

static auto createTensorABIDesc(
    Location location, ConversionPatternRewriter& rewriter,
    const TensorABILayout& layout, Value tensor, mulberry::TensorDescType type)
    -> Value {
  auto desc = LLVM::UndefOp::create(rewriter, location,
                                    layout.descriptorType);
  auto tensorType = llvm::cast<MemRefType>(tensor.getType());
  auto dataMemRefType = convertMemRefToDataMemRefType(tensorType);
  // Field 0 stores only the tensor data pointer. The memref metadata is
  // re-materialized below as explicit sizes/strides fields.
  auto dataMemRef = memref::MemorySpaceCastOp::create(
      rewriter, location, dataMemRefType, tensor);
  auto dataPointer = ptr::ToPtrOp::create(
      rewriter, location, getTensorDataPtrType(rewriter.getContext()),
      dataMemRef.getResult());
  Value current = LLVM::InsertValueOp::create(
      rewriter, location, desc.getResult(), dataPointer.getResult(),
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

static auto createTensorFromABIDesc(
    Location location, ConversionPatternRewriter& rewriter, Value desc,
    mulberry::TensorType type) -> Value {
  auto context = rewriter.getContext();
  auto dataMemRefType = convertTensorToDataMemRefType(type);
  auto resultMemRefType =
      llvm::cast<MemRefType>(*convertTensorToMemRefType(type));

  auto dataPointer = LLVM::ExtractValueOp::create(
      rewriter, location, getTensorDataPtrType(context), desc,
      ArrayRef<int64_t>{0});
  auto layout = MemRefLayoutAttrInterface{};
  auto baseType = MemRefType::get({}, type.getElementType(), layout,
                                  getTensorDataAddressSpace(context));
  // The ABI data pointer has no memref metadata attached. Start from a scalar
  // base memref, then reinterpret it with the descriptor's sizes and strides.
  auto base = ptr::FromPtrOp::create(
      rewriter, location, baseType, dataPointer.getResult(), Value{});

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
  std::vector<int64_t> dynamicStrides(dataMemRefType.getRank(),
                                      ShapedType::kDynamic);
  auto stridedType = MemRefType::get(
      dataMemRefType.getShape(), dataMemRefType.getElementType(),
      StridedLayoutAttr::get(context, ShapedType::kDynamic, dynamicStrides),
      dataMemRefType.getMemorySpace());
  auto stridedTensor = memref::ReinterpretCastOp::create(
      rewriter, location, stridedType, base.getResult(), offset, sizes,
      strides, ArrayRef<NamedAttribute>{});
  auto dataTensor = memref::CastOp::create(
      rewriter, location, dataMemRefType, stridedTensor.getResult());
  auto result = memref::MemorySpaceCastOp::create(
      rewriter, location, resultMemRefType, dataTensor.getResult());
  return result.getResult();
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

static auto containsListStorageType(Type type) -> bool {
  if (llvm::isa<mulberry::ListStorageType>(type))
    return true;

  if (auto recordType = llvm::dyn_cast<mulberry::RecordType>(type))
    for (auto field : recordType.getFields())
      if (containsListStorageType(field.type))
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

static auto isLowerableListDesc(Type type) -> bool {
  auto listDescType = llvm::dyn_cast<mulberry::ListDescType>(type);
  return listDescType && convertToListABIElementType(
                             listDescType.getElementType()).has_value();
}

static auto containsLowerableListDesc(TypeRange types) -> bool {
  for (auto type : types)
    if (isLowerableListDesc(type))
      return true;

  return false;
}

static auto containsUnsupportedListBoundary(Type type) -> bool {
  if (containsListType(type) || containsListStorageType(type))
    return true;

  if (containsListDescType(type) && !isLowerableListDesc(type))
    return true;

  return false;
}

static auto containsUnsupportedListBoundary(TypeRange types) -> bool {
  for (auto type : types)
    if (containsUnsupportedListBoundary(type))
      return true;

  return false;
}

static auto isEscapedListDesc(Value value) -> bool {
  auto pack = value.getDefiningOp<mulberry::ListDescPackOp>();
  if (!pack)
    return false;

  return !!pack.getData().getDefiningOp<mulberry::ListEscapeStorageOp>();
}

static auto rejectListReturns(Operation* root) -> LogicalResult {
  auto result = success();
  root->walk([&](Operation* op) {
    if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {
      if (funcOp.isExternal() &&
          containsLowerableListDesc(funcOp.getFunctionType().getResults())) {
        funcOp.emitError("failed to legalize operation 'func.func': "
                         "List descriptor external returns need an explicit "
                         "ownership ABI");
        result = failure();
        return WalkResult::interrupt();
      }
    }

    if (auto returnOp = llvm::dyn_cast<func::ReturnOp>(op)) {
      for (auto operand : returnOp.getOperands()) {
        if (!isLowerableListDesc(operand.getType()))
          continue;

        if (!isEscapedListDesc(operand)) {
          returnOp.emitError("failed to legalize operation 'func.return': "
                             "List descriptor returns must use escaping "
                             "list storage");
          result = failure();
          return WalkResult::interrupt();
        }
      }
    }

    return WalkResult::advance();
  });

  return result;
}

static auto rejectListBoundaries(Operation* root) -> LogicalResult {
  auto result = success();
  root->walk([&](Operation* op) {
    if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {
      auto funcType = funcOp.getFunctionType();
      if (containsUnsupportedListBoundary(funcType.getInputs()) ||
          containsUnsupportedListBoundary(funcType.getResults())) {
        funcOp.emitError("failed to legalize operation 'func.func': "
                         "List function boundaries are not supported yet");
        result = failure();
        return WalkResult::interrupt();
      }
    }

    if (llvm::isa<func::CallOp>(op)) {
      if (containsUnsupportedListBoundary(op->getOperandTypes()) ||
          containsUnsupportedListBoundary(op->getResultTypes())) {
        op->emitError()
            << "failed to legalize operation '" << op->getName()
            << "': List function boundaries are not supported yet";
        result = failure();
        return WalkResult::interrupt();
      }
    }

    if (llvm::isa<func::ReturnOp>(op)) {
      if (containsUnsupportedListBoundary(op->getOperandTypes())) {
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

static auto rejectListEscape(Operation* root) -> LogicalResult {
  auto result = success();
  root->walk([&](mulberry::ListEscapeStorageOp op) {
    if (isEscapableListStorage(op.getStorage().getType()))
      return WalkResult::advance();

    op.emitError("failed to legalize operation 'mulberry.list.escape_storage': "
                 "list escaping storage needs a lowerable ABI element");
    result = failure();
    return WalkResult::interrupt();
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

  if (llvm::isa<mulberry::FileType>(type))
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
    addConversion([](mulberry::StringType type) -> Type {
      return convertToStringABILayout(type).descriptorType;
    });
    addConversion([](mulberry::FileType type) -> Type {
      return getPtrType(type.getContext());
    });
    addConversion(convertRecordToBackendType);
    addConversion(convertPtrType);
    addConversion(convertTensorToMemRefType);
    // Keep unsupported Mulberry types illegal until each one has a real
    // lowering. The identity conversion above is only for non-Mulberry types.
  }
};

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
    auto ptrType = llvm::dyn_cast<mulberry::PtrType>(op.getPtr().getType());
    if (!ptrType)
      return rewriter.notifyMatchFailure(
          op, "store target must be a Mulberry pointer");

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

class StringLiteralOpLowering
    : public OpConversionPattern<mulberry::StringLiteralOp> {
public:
  using OpConversionPattern<mulberry::StringLiteralOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::StringLiteralOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto stringType = llvm::cast<mulberry::StringType>(
        op.getResult().getType());
    auto layout = convertToStringABILayout(stringType);
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (resultType != layout.descriptorType)
      return rewriter.notifyMatchFailure(
          op, "string literal result type does not match its ABI");

    auto dataPointer = createStringDataPointer(
        op.getLoc(), rewriter, op.getOperation(), op.getValue());
    if (failed(dataPointer))
      return rewriter.notifyMatchFailure(
          op, "string literal needs a parent module");

    auto length = LLVM::ConstantOp::create(
        rewriter, op.getLoc(), getI64Type(op.getContext()),
        rewriter.getI64IntegerAttr(op.getValue().size()));
    auto desc = createStringABIDesc(op.getLoc(), rewriter, layout,
                                    length.getResult(), *dataPointer);
    rewriter.replaceOp(op, desc);
    return success();
  }
};

class FileOpenOpLowering : public OpConversionPattern<mulberry::FileOpenOp> {
public:
  using OpConversionPattern<mulberry::FileOpenOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::FileOpenOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto moduleOp = op->getParentOfType<ModuleOp>();
    if (!moduleOp)
      return rewriter.notifyMatchFailure(op, "file open needs a parent module");

    auto ptrType = getPtrType(op.getContext());
    auto fopenFn = LLVM::lookupOrCreateFn(
        rewriter, moduleOp, "fopen", {ptrType, ptrType}, ptrType);
    if (failed(fopenFn))
      return rewriter.notifyMatchFailure(
          op, "file open needs an fopen declaration");

    // String ABI field 1 is the NUL-terminated byte pointer prepared by string
    // literal lowering, so it can be passed directly to C stdio calls.
    auto path = extractStringDataPointer(op.getLoc(), rewriter,
                                         adaptor.getPath());
    auto mode = extractStringDataPointer(op.getLoc(), rewriter,
                                         adaptor.getMode());
    auto opened = LLVM::CallOp::create(rewriter, op.getLoc(), *fopenFn,
                                       ValueRange{path, mode});
    rewriter.replaceOp(op, opened.getResult());
    return success();
  }
};

class FileReadOpLowering : public OpConversionPattern<mulberry::FileReadOp> {
public:
  using OpConversionPattern<mulberry::FileReadOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::FileReadOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto moduleOp = op->getParentOfType<ModuleOp>();
    if (!moduleOp)
      return rewriter.notifyMatchFailure(op, "file read needs a parent module");

    auto context = op.getContext();
    auto ptrType = getPtrType(context);
    auto i64Type = getI64Type(context);
    auto freadFn = LLVM::lookupOrCreateFn(
        rewriter, moduleOp, "fread",
        {ptrType, i64Type, i64Type, ptrType}, i64Type);
    if (failed(freadFn))
      return rewriter.notifyMatchFailure(
          op, "file read needs an fread declaration");

    auto byteSize = createTensorByteSize(op.getLoc(), rewriter,
                                         adaptor.getBuffer());
    if (failed(byteSize))
      return rewriter.notifyMatchFailure(
          op, "file read needs a memref-backed tensor");
    auto one = LLVM::ConstantOp::create(rewriter, op.getLoc(), i64Type,
                                        rewriter.getI64IntegerAttr(1));
    auto data = createMemRefDataPointer(op.getLoc(), rewriter,
                                        adaptor.getBuffer());
    // Use fread(ptr, 1, byteSize, file) so the return value is a byte count for
    // every supported Tensor element type, not a type-dependent element count.
    auto read = LLVM::CallOp::create(
        rewriter, op.getLoc(), *freadFn,
        ValueRange{data, one.getResult(), *byteSize, adaptor.getFile()});
    rewriter.replaceOp(op, read.getResult());
    return success();
  }
};

class FileWriteOpLowering : public OpConversionPattern<mulberry::FileWriteOp> {
public:
  using OpConversionPattern<mulberry::FileWriteOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::FileWriteOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto moduleOp = op->getParentOfType<ModuleOp>();
    if (!moduleOp)
      return rewriter.notifyMatchFailure(
          op, "file write needs a parent module");

    auto context = op.getContext();
    auto ptrType = getPtrType(context);
    auto i64Type = getI64Type(context);
    auto fwriteFn = LLVM::lookupOrCreateFn(
        rewriter, moduleOp, "fwrite",
        {ptrType, i64Type, i64Type, ptrType}, i64Type);
    if (failed(fwriteFn))
      return rewriter.notifyMatchFailure(
          op, "file write needs an fwrite declaration");

    auto byteSize = createTensorByteSize(op.getLoc(), rewriter,
                                         adaptor.getBuffer());
    if (failed(byteSize))
      return rewriter.notifyMatchFailure(
          op, "file write needs a memref-backed tensor");
    auto one = LLVM::ConstantOp::create(rewriter, op.getLoc(), i64Type,
                                        rewriter.getI64IntegerAttr(1));
    auto data = createMemRefDataPointer(op.getLoc(), rewriter,
                                        adaptor.getBuffer());
    // Keep write() symmetric with read(): the result is the number of raw bytes
    // successfully written, independent of Tensor element type.
    auto written = LLVM::CallOp::create(
        rewriter, op.getLoc(), *fwriteFn,
        ValueRange{data, one.getResult(), *byteSize, adaptor.getFile()});
    rewriter.replaceOp(op, written.getResult());
    return success();
  }
};

class FileCloseOpLowering : public OpConversionPattern<mulberry::FileCloseOp> {
public:
  using OpConversionPattern<mulberry::FileCloseOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::FileCloseOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto moduleOp = op->getParentOfType<ModuleOp>();
    if (!moduleOp)
      return rewriter.notifyMatchFailure(
          op, "file close needs a parent module");

    auto i32Type = getI32Type(op.getContext());
    auto i64Type = getI64Type(op.getContext());
    auto fcloseFn = LLVM::lookupOrCreateFn(
        rewriter, moduleOp, "fclose", {getPtrType(op.getContext())},
        i32Type);
    if (failed(fcloseFn))
      return rewriter.notifyMatchFailure(
          op, "file close needs an fclose declaration");

    auto closed = LLVM::CallOp::create(rewriter, op.getLoc(), *fcloseFn,
                                       ValueRange{adaptor.getFile()});
    auto result = arith::ExtSIOp::create(rewriter, op.getLoc(), i64Type,
                                         closed.getResult());
    rewriter.replaceOp(op, result.getResult());
    return success();
  }
};

class SafetensorReadOpLowering
    : public OpConversionPattern<mulberry::SafetensorReadOp> {
public:
  using OpConversionPattern<
      mulberry::SafetensorReadOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::SafetensorReadOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto moduleOp = op->getParentOfType<ModuleOp>();
    if (!moduleOp)
      return rewriter.notifyMatchFailure(
          op, "safetensor read needs a parent module");

    auto tensorType = llvm::cast<mulberry::TensorType>(
        op.getResult().getType());
    if (!tensorType.getElementType().isF32())
      return rewriter.notifyMatchFailure(
          op, "safetensor read currently supports only f32 tensors");

    auto resultType = llvm::dyn_cast_or_null<MemRefType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType)
      return rewriter.notifyMatchFailure(
          op, "safetensor read needs a lowerable memref result type");

    auto context = op.getContext();
    auto ptrType = getPtrType(context);
    auto i64Type = getI64Type(context);
    auto voidType = LLVM::LLVMVoidType::get(context);
    auto shape = tensorType.getShape();
    auto rank = LLVM::ConstantOp::create(
        rewriter, op.getLoc(), i64Type,
        rewriter.getI64IntegerAttr(static_cast<int64_t>(shape.size())));
    auto expectedShape = createExpectedShapeArray(op.getLoc(), rewriter,
                                                  shape);
    auto outShape = createI64StackArray(op.getLoc(), rewriter, shape.size());
    auto name = extractStringDataPointer(op.getLoc(), rewriter,
                                         adaptor.getName());

    auto shapeFn = LLVM::lookupOrCreateFn(
        rewriter, moduleOp, "mulberry_safetensor_shape_f32",
        {ptrType, ptrType, i64Type, ptrType, ptrType}, voidType);
    if (failed(shapeFn))
      return rewriter.notifyMatchFailure(
          op, "safetensor read needs a shape runtime declaration");

    // Runtime owns safetensors parsing. Lowering only passes expected shape
    // metadata, then uses the returned concrete shape to allocate dynamic
    // memref dimensions before reading the payload.
    LLVM::CallOp::create(
        rewriter, op.getLoc(), *shapeFn,
        ValueRange{adaptor.getFile(), name, rank.getResult(), expectedShape,
                   outShape});

    std::vector<Value> dynamicSizes;
    for (size_t dim = 0; dim < shape.size(); ++dim) {
      if (shape[dim] >= 0)
        continue;
      auto sizeI64 = loadRuntimeShapeDim(op.getLoc(), rewriter, outShape, dim);
      dynamicSizes.push_back(arith::IndexCastOp::create(
                                 rewriter, op.getLoc(),
                                 rewriter.getIndexType(), sizeI64)
                                 .getResult());
    }

    auto tensor = memref::AllocOp::create(rewriter, op.getLoc(), resultType,
                                          dynamicSizes);
    auto data = createMemRefDataPointer(op.getLoc(), rewriter,
                                        tensor.getResult());

    auto readFn = LLVM::lookupOrCreateFn(
        rewriter, moduleOp, "mulberry_safetensor_read_f32",
        {ptrType, ptrType, i64Type, ptrType, ptrType}, voidType);
    if (failed(readFn))
      return rewriter.notifyMatchFailure(
          op, "safetensor read needs a payload runtime declaration");

    LLVM::CallOp::create(
        rewriter, op.getLoc(), *readFn,
        ValueRange{adaptor.getFile(), name, rank.getResult(), expectedShape,
                   data});

    rewriter.replaceOp(op, tensor.getResult());
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

class ListEscapeStorageOpLowering
    : public OpConversionPattern<mulberry::ListEscapeStorageOp> {
public:
  using OpConversionPattern<mulberry::ListEscapeStorageOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::ListEscapeStorageOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    // This op is an ownership marker, not a storage transform. desc_pack sees
    // the source marker and materializes the Boehm-managed ABI data pointer.
    rewriter.replaceOp(op, adaptor.getStorage());
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
    auto descElementType = descType.getElementType();

    auto storageType = mulberry::ListStorageType::get(
        op.getContext(), descElementType);
    auto length = arith::ConstantIndexOp::create(
        rewriter, op.getLoc(), createOp.getElements().size());
    auto storage = mulberry::ListAllocOp::create(
        rewriter, op.getLoc(), storageType, length);

    // Bridge high-level List<T> to ABI-ready ListDesc<T>. Tensor elements must
    // be packed first; scalar elements can be stored directly.
    for (auto element : llvm::enumerate(createOp.getElements())) {
      auto index = arith::ConstantIndexOp::create(
          rewriter, op.getLoc(), element.index());
      Value value = element.value();
      if (auto tensorDescType =
              llvm::dyn_cast<mulberry::TensorDescType>(descElementType)) {
        auto desc = mulberry::TensorDescPackOp::create(
            rewriter, op.getLoc(), tensorDescType, element.value());
        value = desc.getResult();
      }
      mulberry::ListStoreOp::create(rewriter, op.getLoc(), value, storage,
                                    index);
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
    auto descPack = op.getDesc().getDefiningOp<mulberry::ListDescPackOp>();
    if (!descPack && isScalarStorageType(descType.getElementType()))
      return rewriter.notifyMatchFailure(
          op, "scalar list descriptor get needs ABI projection");

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

class ListDescGetOpConversion
    : public OpConversionPattern<mulberry::ListDescGetOp> {
public:
  using OpConversionPattern<mulberry::ListDescGetOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::ListDescGetOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto descType = llvm::cast<mulberry::ListDescType>(
        op.getDesc().getType());
    auto elementType = convertToListABIElementType(descType.getElementType());
    if (!elementType || !isScalarStorageType(descType.getElementType()))
      return rewriter.notifyMatchFailure(
          op, "list descriptor get needs a scalar ABI element");

    auto data = LLVM::ExtractValueOp::create(
        rewriter, op.getLoc(), getPtrType(op.getContext()),
        adaptor.getDesc(), ArrayRef<int64_t>{1});
    auto index = arith::IndexCastOp::create(
        rewriter, op.getLoc(), getI64Type(op.getContext()),
        adaptor.getIndex());
    auto elementPtr = LLVM::GEPOp::create(
        rewriter, op.getLoc(), getPtrType(op.getContext()), *elementType,
        data.getResult(), ArrayRef<Value>{index.getResult()});
    auto loaded = LLVM::LoadOp::create(
        rewriter, op.getLoc(), *elementType, elementPtr.getResult());
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

    // desc_pack owns the actual `{length, data}` materialization. Keeping this
    // here prevents desc_to_abi from becoming a hidden function-boundary bridge.
    auto length = arith::IndexCastOp::create(
        rewriter, op.getLoc(), getI64Type(op.getContext()),
        adaptor.getLength());

    Value dataPointer;
    if (auto escape =
            op.getData().getDefiningOp<mulberry::ListEscapeStorageOp>()) {
      auto loweredStorage = rewriter.getRemappedValue(escape.getStorage());
      if (!loweredStorage)
        return rewriter.notifyMatchFailure(
            op, "escaped list descriptor needs lowered storage");

      auto loweredLength = rewriter.getRemappedValue(escape.getLength());
      if (!loweredLength)
        loweredLength = escape.getLength();

      auto storageType = llvm::cast<mulberry::ListStorageType>(
          escape.getStorage().getType());
      auto escapedData = createEscapedListData(
          op.getLoc(), rewriter, op.getOperation(), storageType,
          loweredStorage, loweredLength);
      if (failed(escapedData))
        return rewriter.notifyMatchFailure(
            op, "escaped list descriptor needs Boehm-managed storage");
      dataPointer = *escapedData;
    } else {
      auto dataType = adaptor.getData().getType();
      if (!dataType)
        return rewriter.notifyMatchFailure(
            op, "list descriptor ABI needs lowerable list storage");
      if (!llvm::isa<MemRefType, LLVM::LLVMPointerType>(dataType))
        return rewriter.notifyMatchFailure(
            op, "list descriptor ABI needs lowered list storage");

      dataPointer = adaptor.getData();
      if (llvm::isa<MemRefType>(dataType))
        dataPointer = createMemRefDataPointer(op.getLoc(), rewriter,
                                              adaptor.getData());
    }

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
    auto ptrType = llvm::dyn_cast<mulberry::PtrType>(op.getRecord().getType());
    if (!ptrType)
      return rewriter.notifyMatchFailure(
          op, "field address needs a pointer to a record");

    auto recordType =
        llvm::dyn_cast<mulberry::RecordType>(ptrType.getPointeeType());
    if (!recordType)
      return rewriter.notifyMatchFailure(
          op, "field address needs a pointer to a record");

    auto llvmRecordType = getTypeConverter()->convertType(recordType);
    auto fieldType = convertToBackendType(recordType.getFieldType(
        op.getField()));
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

class TensorDescUnpackOpConversion
    : public OpConversionPattern<mulberry::TensorDescUnpackOp> {
public:
  using OpConversionPattern<
      mulberry::TensorDescUnpackOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::TensorDescUnpackOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto tensorType = llvm::cast<mulberry::TensorType>(
        op.getResult().getType());
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!llvm::isa_and_nonnull<MemRefType>(resultType))
      return rewriter.notifyMatchFailure(
          op, "tensor descriptor unpack needs a lowered memref result type");

    auto tensor = createTensorFromABIDesc(op.getLoc(), rewriter,
                                          adaptor.getDesc(), tensorType);
    rewriter.replaceOp(op, tensor);
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
    if (failed(rejectListEscape(getOperation())))
      return signalPassFailure();

    if (failed(rejectListReturns(getOperation())))
      return signalPassFailure();

    if (failed(rejectListBoundaries(getOperation())))
      return signalPassFailure();

    if (failed(rejectTensorDescBoundaries(getOperation())))
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
                       ListDescLengthOpLowering,
                       ListDescPackOpLowering, ListToDescOpLowering>(
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
                           ptr::PtrDialect, scf::SCFDialect>();
    target.addIllegalDialect<cherry_nn::CherryNNDialect>();
    target.addIllegalDialect<mulberry::MulberryDialect>();
    target.addDynamicallyLegalOp<func::FuncOp>(
        [&](func::FuncOp op) {
          // Tensor function boundaries lower to memref today. Only explicit
          // list_desc<TensorDesc> values may use the List ABI descriptor.
          if (containsTensorDescType(op.getFunctionType()) ||
              containsUnsupportedListBoundary(
                  op.getFunctionType().getInputs()) ||
              containsUnsupportedListBoundary(
                  op.getFunctionType().getResults()))
            return false;
          return typeConverter.isSignatureLegal(op.getFunctionType());
        });
    target.addDynamicallyLegalOp<func::CallOp, func::ReturnOp>(
        [&](Operation* op) {
          if (containsTensorDescType(op->getOperandTypes()) ||
              containsTensorDescType(op->getResultTypes()) ||
              containsUnsupportedListBoundary(op->getOperandTypes()) ||
              containsUnsupportedListBoundary(op->getResultTypes()))
            return false;
          if (llvm::isa<func::ReturnOp>(op) &&
              containsUnsupportedListBoundary(op->getOperandTypes()))
            return false;
          return typeConverter.isLegal(op);
        });

    RewritePatternSet patterns(&getContext());
    patterns.add<AllocaOpLowering, FileCloseOpLowering, FileOpenOpLowering,
                 FileReadOpLowering, FileWriteOpLowering, ListAllocOpLowering,
                 ListCreateOpLowering, ListEscapeStorageOpLowering,
                 ListGetOpLowering, ListLengthOpLowering, ListLoadOpLowering,
                 ListDescDataOpConversion, ListDescGetOpConversion,
                 ListDescLengthOpConversion,
                 ListDescPackOpConversion, ListDescToABIOpLowering,
                 ListSizeOpLowering, ListStoreOpLowering, LoadOpLowering,
                 RecordExtractOpLowering, RecordGetFieldOpLowering,
                 SafetensorReadOpLowering, StoreOpLowering,
                 StringLiteralOpLowering,
                 TensorAllocOpLowering, TensorCastOpLowering,
                 TensorDimOpLowering,
                 TensorDescPackOpConversion, TensorDescToABIOpLowering,
                 TensorDescUnpackOpConversion, TensorLoadOpLowering,
                 TensorStoreOpLowering>(
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
