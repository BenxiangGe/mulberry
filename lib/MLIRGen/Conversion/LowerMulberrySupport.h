//===--- LowerMulberrySupport.h ---------------------------------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_MLIRGEN_CONVERSION_LOWERMULBERRYSUPPORT_H
#define MULBERRY_MLIRGEN_CONVERSION_LOWERMULBERRYSUPPORT_H

#include "mulberry/MLIRGen/IR/MulberryTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace mlir::mulberry_core::detail {

class MulberryTypeConverter : public TypeConverter {
public:
  MulberryTypeConverter();
};

auto getPtrType(MLIRContext* context) -> LLVM::LLVMPointerType;
auto getI64Type(MLIRContext* context) -> IntegerType;
auto getI32Type(MLIRContext* context) -> IntegerType;

auto convertTensorToMemRefType(TensorType type) -> MemRefType;
auto convertBackendValueType(Type type) -> std::optional<Type>;
auto convertRecordFieldStorageType(Type type) -> std::optional<Type>;
auto convertRecordLayoutType(RecordType type) -> std::optional<Type>;
auto convertPointerValueType(PtrType type) -> std::optional<Type>;

auto createSizeOf(Location location, OpBuilder& builder, Type type) -> Value;
auto callBoehmMalloc(Location location, OpBuilder& builder, Operation* op,
                     Value sizeInBytes) -> FailureOr<Value>;
auto callRuntimeVoid(Location location, OpBuilder& builder, Operation* op,
                     llvm::StringRef name, ValueRange arguments = {})
    -> LogicalResult;
auto callRuntimeValue(Location location, OpBuilder& builder, Operation* op,
                      llvm::StringRef name, Type resultType,
                      ValueRange arguments) -> FailureOr<Value>;

void populateMulberryBigIntLoweringPatterns(RewritePatternSet& patterns,
                                            TypeConverter& typeConverter,
                                            MLIRContext* context);
void populateMulberryCoreLoweringPatterns(RewritePatternSet& patterns,
                                          TypeConverter& typeConverter,
                                          MLIRContext* context);
void populateMulberryTensorLoweringPatterns(RewritePatternSet& patterns,
                                            TypeConverter& typeConverter,
                                            MLIRContext* context);
void populateFinalizeMulberryTensorStoragePatterns(RewritePatternSet& patterns,
                                                   MLIRContext* context);

} // namespace mlir::mulberry_core::detail

#endif // MULBERRY_MLIRGEN_CONVERSION_LOWERMULBERRYSUPPORT_H
