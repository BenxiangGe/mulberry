#include "cherry/MLIRGen/TypeConverter.h"
#include "cherry/MLIRGen/IR/MulberryDialect.h"
#include "cherry/MLIRGen/IR/MulberryTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/Casting.h"
#include "gtest/gtest.h"

using namespace cherry;

namespace {

TEST(TypeConverterTest, ConvertsTensorDescriptorType) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::mulberry::MulberryDialect>();
  mlir::OpBuilder builder(&context);

  TypeContext typeContext;
  auto *elementType = typeContext.getBuiltinType(BuiltinTypeKind::Float32);
  auto *tensorType = typeContext.createTensorType(elementType, {30, -1});

  MLIRTypeConverter converter(builder);
  auto descriptorType = converter.convertTensorDescriptor(*tensorType);

  ASSERT_TRUE(descriptorType);
  EXPECT_EQ(descriptorType.getName(), "TensorDescriptorFloat32Rank2");
  ASSERT_EQ(descriptorType.getNumFields(), 5u);

  auto allocatedField = descriptorType.getFields()[0];
  EXPECT_EQ(allocatedField.name, "allocated");
  auto allocatedPtrType =
      llvm::dyn_cast<mlir::mulberry::PtrType>(allocatedField.type);
  ASSERT_TRUE(allocatedPtrType);
  EXPECT_TRUE(allocatedPtrType.getElementType().isF32());

  auto alignedField = descriptorType.getFields()[1];
  EXPECT_EQ(alignedField.name, "aligned");
  auto alignedPtrType =
      llvm::dyn_cast<mlir::mulberry::PtrType>(alignedField.type);
  ASSERT_TRUE(alignedPtrType);
  EXPECT_TRUE(alignedPtrType.getElementType().isF32());

  auto offsetField = descriptorType.getFields()[2];
  EXPECT_EQ(offsetField.name, "offset");
  EXPECT_TRUE(offsetField.type.isInteger(64));

  auto sizesField = descriptorType.getFields()[3];
  EXPECT_EQ(sizesField.name, "sizes");
  auto sizesType =
      llvm::dyn_cast<mlir::mulberry::RecordType>(sizesField.type);
  ASSERT_TRUE(sizesType);
  EXPECT_EQ(sizesType.getName(), "TensorSizesRank2");
  ASSERT_EQ(sizesType.getNumFields(), 2u);
  EXPECT_EQ(sizesType.getFields()[0].name, "dim0");
  EXPECT_TRUE(sizesType.getFields()[0].type.isInteger(64));
  EXPECT_EQ(sizesType.getFields()[1].name, "dim1");
  EXPECT_TRUE(sizesType.getFields()[1].type.isInteger(64));

  auto stridesField = descriptorType.getFields()[4];
  EXPECT_EQ(stridesField.name, "strides");
  auto stridesType =
      llvm::dyn_cast<mlir::mulberry::RecordType>(stridesField.type);
  ASSERT_TRUE(stridesType);
  EXPECT_EQ(stridesType.getName(), "TensorStridesRank2");
  ASSERT_EQ(stridesType.getNumFields(), 2u);
  EXPECT_EQ(stridesType.getFields()[0].name, "dim0");
  EXPECT_TRUE(stridesType.getFields()[0].type.isInteger(64));
  EXPECT_EQ(stridesType.getFields()[1].name, "dim1");
  EXPECT_TRUE(stridesType.getFields()[1].type.isInteger(64));
}

} // namespace
