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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
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

static auto getTensorDataAddressSpace(MLIRContext* context) -> Attribute {
  return LLVM::AddressSpaceAttr::get(context, 0);
}

static auto isScalarStorageType(Type type) -> bool {
  return type.isIndex() || llvm::isa<IntegerType, FloatType>(type);
}

static constexpr std::string_view kFileReadMarkerPrefix =
    "__cherry_file_read_";
static constexpr std::string_view kFileWriteMarkerPrefix =
    "__cherry_file_write_";
static constexpr std::string_view kSafetensorReadMarkerPrefix =
    "__cherry_safetensor_read_";

using StringStorageType = LLVM::LLVMStructType;
using FileStorageType = LLVM::LLVMStructType;

static constexpr int32_t kStringDataField = 1;
static constexpr int32_t kFileHandleField = 0;

// Lowered StringStorage heap layout:
//
//   { length: i64, data: ptr }
//
// `length` is the source string byte count and excludes the trailing NUL.
// `data` points to a NUL-terminated heap byte buffer so C runtime calls can
// use it directly.
static auto getStringStorageType(MLIRContext* context) -> StringStorageType {
  std::vector<Type> fields;
  fields.push_back(getI64Type(context)); // byte length, excluding trailing NUL
  fields.push_back(getPtrType(context)); // data pointer
  return LLVM::LLVMStructType::getLiteral(context, fields);
}

// Lowered FileStorage heap layout:
//
//   { handle: ptr }
//
// `handle` is an opaque C runtime FILE* stored as an LLVM pointer. Source-level
// File is just Ptr<FileStorage>; only the file runtime boundary looks inside.
static auto getFileStorageType(MLIRContext* context) -> FileStorageType {
  std::vector<Type> fields;
  fields.push_back(getPtrType(context)); // opaque FILE* handle
  return LLVM::LLVMStructType::getLiteral(context, fields);
}

static auto convertRecordToBackendType(mulberry::RecordType type)
    -> std::optional<Type>;
static auto convertPtrType(mulberry::PtrType type) -> std::optional<Type>;

static auto convertMemRefShape(ArrayRef<int64_t> shape)
    -> std::vector<int64_t> {
  std::vector<int64_t> memrefShape;
  for (auto dim : shape) {
    memrefShape.push_back(dim < 0 ? ShapedType::kDynamic : dim);
  }
  return memrefShape;
}

static auto convertTensorToMemRefType(mulberry::TensorType type) -> MemRefType {
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

// This pass is a transitional storage lowering, not the final
// Mulberry-to-LLVM ABI lowering. Tensor values become memrefs so cherry_nn can
// lower to linalg, while scalar/record stack storage still uses LLVM dialect.
static auto convertToBackendType(Type type) -> std::optional<Type> {
  if (auto recordType = llvm::dyn_cast<mulberry::RecordType>(type))
    return convertRecordToBackendType(recordType);

  if (auto ptrType = llvm::dyn_cast<mulberry::PtrType>(type))
    return convertPtrType(ptrType);

  if (isScalarStorageType(type))
    return type;

  return std::nullopt;
}

static auto convertAllocaElementType(Type type) -> std::optional<Type> {
  if (auto tensorType = llvm::dyn_cast<mulberry::TensorType>(type))
    return convertTensorToMemRefType(tensorType);

  return convertToBackendType(type);
}

using TensorABIDescriptorType = LLVM::LLVMStructType;

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

static auto getTensorDescType(mulberry::TensorType type)
    -> mulberry::TensorDescType {
  return mulberry::TensorDescType::get(type.getContext(), type.getShape(),
                                       type.getElementType());
}

static auto convertTensorToABIDescriptorType(mulberry::TensorType type)
    -> TensorABIDescriptorType {
  return convertToTensorABILayout(getTensorDescType(type)).descriptorType;
}

static auto convertPointerElementType(Type type) -> std::optional<Type> {
  if (auto tensorType = llvm::dyn_cast<mulberry::TensorType>(type))
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

// Shared raw-byte view for tensor-backed runtime edges:
//   data: raw aligned pointer
//   byteSize: total payload byte count
// This is lowering-only ABI glue, not a source-level value type.
struct TensorByteView {
  Value data;
  Value byteSize;
};

static auto createTensorByteView(Location location,
                                 ConversionPatternRewriter& rewriter,
                                 Value tensor)
    -> FailureOr<TensorByteView> {
  auto byteSize = createTensorByteSize(location, rewriter, tensor);
  if (failed(byteSize))
    return failure();

  auto data = createMemRefDataPointer(location, rewriter, tensor);
  return TensorByteView{data, *byteSize};
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

static auto loadStringDataPointer(Location location, OpBuilder& builder,
                                  Value stringStorage) -> Value {
  auto context = builder.getContext();
  auto storageType = getStringStorageType(context);
  auto dataPtr = LLVM::GEPOp::create(
      builder, location, getPtrType(context), storageType, stringStorage,
      ArrayRef<LLVM::GEPArg>{0, kStringDataField});
  return LLVM::LoadOp::create(builder, location, getPtrType(context),
                              dataPtr.getResult()).getResult();
}

static auto loadFileHandlePointer(Location location, OpBuilder& builder,
                                  Value fileStorage) -> Value {
  auto context = builder.getContext();
  auto storageType = getFileStorageType(context);
  auto handlePtr = LLVM::GEPOp::create(
      builder, location, getPtrType(context), storageType, fileStorage,
      ArrayRef<LLVM::GEPArg>{0, kFileHandleField});
  return LLVM::LoadOp::create(builder, location, getPtrType(context),
                              handlePtr.getResult()).getResult();
}

static auto isFileMarkerName(StringRef name, std::string_view prefix) -> bool {
  return name.starts_with(prefix);
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
    mulberry::TensorType type) -> Value {
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

static auto eraseDeadRuntimeMarkers(ModuleOp moduleOp) -> void {
  std::vector<func::FuncOp> deadMarkers;
  moduleOp.walk([&](func::FuncOp funcOp) {
    auto name = funcOp.getSymName();
    if (!isFileMarkerName(name, kFileReadMarkerPrefix) &&
        !isFileMarkerName(name, kFileWriteMarkerPrefix) &&
        !isFileMarkerName(name, kSafetensorReadMarkerPrefix))
      return;

    if (SymbolTable::symbolKnownUseEmpty(funcOp, moduleOp))
      deadMarkers.push_back(funcOp);
  });

  for (auto funcOp : deadMarkers)
    funcOp.erase();
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
  if (auto recordType = llvm::dyn_cast<mulberry::RecordType>(type))
    if (!convertRecordToBackendType(recordType))
      return failure();

  if (auto ptrType = llvm::dyn_cast<mulberry::PtrType>(type))
    if (!convertPtrType(ptrType))
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

class AllocaOpLowering : public OpConversionPattern<mulberry::AllocaOp> {
public:
  using OpConversionPattern<mulberry::AllocaOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::AllocaOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    if (auto tensorType = llvm::dyn_cast<mulberry::TensorType>(
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
    : public OpConversionPattern<mulberry::HeapAllocOp> {
public:
  using OpConversionPattern<mulberry::HeapAllocOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::HeapAllocOp op, OpAdaptor adaptor,
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
    : public OpConversionPattern<mulberry::PtrIndexOp> {
public:
  using OpConversionPattern<mulberry::PtrIndexOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::PtrIndexOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto ptrType = llvm::cast<mulberry::PtrType>(op.getPtr().getType());
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

class LoadOpLowering : public OpConversionPattern<mulberry::LoadOp> {
public:
  using OpConversionPattern<mulberry::LoadOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::LoadOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto ptrType = llvm::cast<mulberry::PtrType>(op.getPtr().getType());
    if (auto tensorType = llvm::dyn_cast<mulberry::TensorType>(
            ptrType.getPointeeType())) {
      auto descType = getTensorDescType(tensorType);
      auto layout = convertToTensorABILayout(descType);
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

    if (auto tensorType = llvm::dyn_cast<mulberry::TensorType>(
            ptrType.getPointeeType())) {
      if (!llvm::isa<MemRefType>(adaptor.getValue().getType()))
        return rewriter.notifyMatchFailure(
            op, "tensor store needs lowered tensor storage");

      auto descType = getTensorDescType(tensorType);
      auto layout = convertToTensorABILayout(descType);
      auto desc = createTensorABIDesc(op.getLoc(), rewriter, layout,
                                      adaptor.getValue(), descType);
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

class FileReadCallLowering : public OpConversionPattern<func::CallOp> {
public:
  using OpConversionPattern<func::CallOp>::OpConversionPattern;

  auto matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    if (!isFileMarkerName(op.getCallee(), kFileReadMarkerPrefix))
      return failure();
    if (adaptor.getOperands().size() != 2)
      return rewriter.notifyMatchFailure(op, "file read marker needs 2 args");

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

    auto file = adaptor.getOperands()[0];
    auto buffer = adaptor.getOperands()[1];
    auto rawByteView = createTensorByteView(op.getLoc(), rewriter, buffer);
    if (failed(rawByteView))
      return rewriter.notifyMatchFailure(
          op, "file read needs a tensor raw-byte view");
    auto view = *rawByteView;
    auto one = LLVM::ConstantOp::create(rewriter, op.getLoc(), i64Type,
                                        rewriter.getI64IntegerAttr(1));
    // Use fread(ptr, 1, byteSize, file) so the return value is a byte count for
    // every supported Tensor element type, not a type-dependent element count.
    auto fileHandle = loadFileHandlePointer(op.getLoc(), rewriter, file);
    auto read = LLVM::CallOp::create(
        rewriter, op.getLoc(), *freadFn,
        ValueRange{view.data, one.getResult(), view.byteSize, fileHandle});
    rewriter.replaceOp(op, read.getResult());
    return success();
  }
};

class FileWriteCallLowering : public OpConversionPattern<func::CallOp> {
public:
  using OpConversionPattern<func::CallOp>::OpConversionPattern;

  auto matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    if (!isFileMarkerName(op.getCallee(), kFileWriteMarkerPrefix))
      return failure();
    if (adaptor.getOperands().size() != 2)
      return rewriter.notifyMatchFailure(op, "file write marker needs 2 args");

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

    auto file = adaptor.getOperands()[0];
    auto buffer = adaptor.getOperands()[1];
    auto rawByteView = createTensorByteView(op.getLoc(), rewriter, buffer);
    if (failed(rawByteView))
      return rewriter.notifyMatchFailure(
          op, "file write needs a tensor raw-byte view");
    auto view = *rawByteView;
    auto one = LLVM::ConstantOp::create(rewriter, op.getLoc(), i64Type,
                                        rewriter.getI64IntegerAttr(1));
    // Keep write() symmetric with read(): the result is the number of raw bytes
    // successfully written, independent of Tensor element type.
    auto fileHandle = loadFileHandlePointer(op.getLoc(), rewriter, file);
    auto written = LLVM::CallOp::create(
        rewriter, op.getLoc(), *fwriteFn,
        ValueRange{view.data, one.getResult(), view.byteSize, fileHandle});
    rewriter.replaceOp(op, written.getResult());
    return success();
  }
};

class SafetensorReadCallLowering : public OpConversionPattern<func::CallOp> {
public:
  using OpConversionPattern<func::CallOp>::OpConversionPattern;

  auto matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    if (!isFileMarkerName(op.getCallee(), kSafetensorReadMarkerPrefix))
      return failure();
    if (adaptor.getOperands().size() != 2)
      return rewriter.notifyMatchFailure(
          op, "safetensor read marker needs 2 args");

    auto moduleOp = op->getParentOfType<ModuleOp>();
    if (!moduleOp)
      return rewriter.notifyMatchFailure(
          op, "safetensor read needs a parent module");

    auto tensorType = llvm::dyn_cast<mulberry::TensorType>(
        op.getResult(0).getType());
    if (!tensorType)
      return rewriter.notifyMatchFailure(
          op, "safetensor read result must be a Mulberry tensor");
    if (!tensorType.getElementType().isF32())
      return rewriter.notifyMatchFailure(
          op, "safetensor read currently supports only f32 tensors");

    auto resultType = convertTensorToMemRefType(tensorType);

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
    auto file = adaptor.getOperands()[0];
    auto nameArg = adaptor.getOperands()[1];
    auto name = loadStringDataPointer(op.getLoc(), rewriter, nameArg);
    auto fileHandle = loadFileHandlePointer(op.getLoc(), rewriter, file);

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
        ValueRange{fileHandle, name, rank.getResult(), expectedShape,
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
        ValueRange{fileHandle, name, rank.getResult(), expectedShape,
                   data});

    rewriter.replaceOp(op, tensor.getResult());
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
    auto tensorType = llvm::cast<mulberry::TensorType>(
        op.getResult().getType());
    auto resultType = convertTensorToMemRefType(tensorType);

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
    auto destType = convertTensorToMemRefType(
        llvm::cast<mulberry::TensorType>(op.getDest().getType()));

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
    rewriter.replaceOp(op, adaptor.getDesc());
    return success();
  }
};

struct LowerMulberry : public impl::LowerMulberryBase<LowerMulberry> {
  using impl::LowerMulberryBase<LowerMulberry>::LowerMulberryBase;

  auto runOnOperation() -> void final {
    if (failed(rejectTensorDescBoundaries(getOperation())))
      return signalPassFailure();

    MulberryTypeConverter typeConverter;

    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect,
                           linalg::LinalgDialect, LLVM::LLVMDialect,
                           math::MathDialect, memref::MemRefDialect,
                           scf::SCFDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addIllegalDialect<cherry_nn::CherryNNDialect>();
    target.addIllegalDialect<mulberry::MulberryDialect>();
    target.addDynamicallyLegalOp<func::FuncOp>(
        [&](func::FuncOp op) {
          // Tensor function boundaries lower to memref today. Tensor descriptor
          // values are local ABI helpers and must not cross function boundaries.
          if (containsTensorDescType(op.getFunctionType()))
            return false;
          return typeConverter.isSignatureLegal(op.getFunctionType());
        });
    target.addDynamicallyLegalOp<func::CallOp, func::ReturnOp>(
        [&](Operation* op) {
          if (containsTensorDescType(op->getOperandTypes()) ||
              containsTensorDescType(op->getResultTypes()))
            return false;
          return typeConverter.isLegal(op);
        });

    RewritePatternSet patterns(&getContext());
    patterns.add<FileReadCallLowering, FileWriteCallLowering,
                 SafetensorReadCallLowering>(
        typeConverter, &getContext(), /*benefit=*/2);
    patterns.add<AllocaOpLowering, HeapAllocOpLowering,
                 LoadOpLowering, PtrIndexOpLowering,
                 RecordExtractOpLowering, RecordGetFieldOpLowering,
                 StoreOpLowering, TensorAllocOpLowering, TensorCastOpLowering,
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

    eraseDeadRuntimeMarkers(llvm::cast<ModuleOp>(getOperation()));
  }
};

} // namespace
} // namespace mlir::cherry
