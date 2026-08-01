//===--- LowerMulberrySupport.cpp -----------------------------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "LowerMulberrySupport.h"

#include "mulberry/BigInt/BigIntOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

#include <vector>

namespace mlir::mulberry_core::detail {

namespace bigint = ::mlir::bigint;

namespace {

auto isScalarStorageType(Type type) -> bool {
  return type.isIndex() || llvm::isa<IntegerType, FloatType>(type);
}

auto convertBigIntType(bigint::IntType type) -> Type {
  return getPtrType(type.getContext());
}

auto convertMemRefShape(ArrayRef<int64_t> shape) -> std::vector<int64_t> {
  std::vector<int64_t> memrefShape;
  for (auto dim : shape)
    memrefShape.push_back(dim < 0 ? ShapedType::kDynamic : dim);
  return memrefShape;
}

// Safety net for Mulberry types that should not fall through to the identity
// conversion after their specific lowering conversion fails.
auto rejectUnloweredMulberryType(Type type, SmallVectorImpl<Type>&)
    -> std::optional<LogicalResult> {
  if (llvm::isa<DataType>(type))
    return failure();

  if (auto recordType = llvm::dyn_cast<RecordType>(type))
    if (!convertRecordLayoutType(recordType))
      return failure();

  if (auto ptrType = llvm::dyn_cast<PtrType>(type))
    if (!convertPointerValueType(ptrType))
      return failure();

  return std::nullopt;
}

} // namespace

auto getPtrType(MLIRContext* context) -> LLVM::LLVMPointerType {
  return LLVM::LLVMPointerType::get(context);
}

auto getI64Type(MLIRContext* context) -> IntegerType {
  return IntegerType::get(context, 64);
}

auto getI32Type(MLIRContext* context) -> IntegerType {
  return IntegerType::get(context, 32);
}

// mulberry_core.tensor is the compiler-owned internal tensor IR. It lowers to
// memref; source-level Tensor<T> is a stdlib record header.
auto convertTensorToMemRefType(TensorType type) -> MemRefType {
  return MemRefType::get(convertMemRefShape(type.getShape()),
                         type.getElementType());
}

// This pass is a transitional storage lowering, not the final Mulberry-to-LLVM
// ABI lowering. Domain packages such as mulberry.nn are intentionally not
// lowered here; they live outside core.
auto convertBackendValueType(Type type) -> std::optional<Type> {
  if (auto bigintType = llvm::dyn_cast<bigint::IntType>(type))
    return convertBigIntType(bigintType);

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

auto convertRecordFieldStorageType(Type type) -> std::optional<Type> {
  // Internal tensors have a context-dependent representation and cannot be
  // embedded directly in an ordinary object record layout.
  if (llvm::isa<TensorType>(type))
    return std::nullopt;

  return convertBackendValueType(type);
}

auto convertRecordLayoutType(RecordType type) -> std::optional<Type> {
  std::vector<Type> fieldTypes;
  for (auto field : type.getFields()) {
    auto fieldType = convertRecordFieldStorageType(field.type);
    if (!fieldType)
      return std::nullopt;
    fieldTypes.push_back(*fieldType);
  }

  return LLVM::LLVMStructType::getLiteral(type.getContext(), fieldTypes);
}

auto convertPointerValueType(PtrType type) -> std::optional<Type> {
  auto pointeeType = type.getPointeeType();
  if (llvm::isa<DataType>(pointeeType))
    return getPtrType(type.getContext());
  if (llvm::isa<TensorType>(pointeeType) ||
      !convertBackendValueType(pointeeType))
    return std::nullopt;

  return getPtrType(type.getContext());
}

MulberryTypeConverter::MulberryTypeConverter() {
  addConversion([](Type type) { return type; });
  addConversion(rejectUnloweredMulberryType);
  addConversion(convertBigIntType);
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

auto createSizeOf(Location location, OpBuilder& builder, Type type) -> Value {
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

auto callBoehmMalloc(Location location, OpBuilder& builder, Operation* op,
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

auto callRuntimeVoid(Location location, OpBuilder& builder, Operation* op,
                     llvm::StringRef name, ValueRange arguments)
    -> LogicalResult {
  auto moduleOp = op->getParentOfType<ModuleOp>();
  if (!moduleOp)
    return failure();

  std::vector<Type> argumentTypes;
  for (auto argument : arguments)
    argumentTypes.push_back(argument.getType());
  auto function = LLVM::lookupOrCreateFn(
      builder, moduleOp, name, argumentTypes,
      LLVM::LLVMVoidType::get(builder.getContext()));
  if (failed(function))
    return failure();

  LLVM::CallOp::create(builder, location, *function, arguments);
  return success();
}

auto callRuntimeValue(Location location, OpBuilder& builder, Operation* op,
                      llvm::StringRef name, Type resultType,
                      ValueRange arguments) -> FailureOr<Value> {
  auto moduleOp = op->getParentOfType<ModuleOp>();
  if (!moduleOp)
    return failure();

  std::vector<Type> argumentTypes;
  for (auto argument : arguments)
    argumentTypes.push_back(argument.getType());
  auto function = LLVM::lookupOrCreateFn(builder, moduleOp, name,
                                          argumentTypes, resultType);
  if (failed(function))
    return failure();

  return LLVM::CallOp::create(builder, location, *function, arguments)
      .getResult();
}

} // namespace mlir::mulberry_core::detail
