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
  ASSERT_EQ(descriptorType.getNumFields(), 2u);

  auto dataField = descriptorType.getFields()[0];
  EXPECT_EQ(dataField.name, "data");
  auto dataPtrType =
      llvm::dyn_cast<mlir::mulberry::PtrType>(dataField.type);
  ASSERT_TRUE(dataPtrType);
  EXPECT_TRUE(dataPtrType.getElementType().isF32());

  auto shapeField = descriptorType.getFields()[1];
  EXPECT_EQ(shapeField.name, "shape");
  auto shapeType =
      llvm::dyn_cast<mlir::mulberry::RecordType>(shapeField.type);
  ASSERT_TRUE(shapeType);
  EXPECT_EQ(shapeType.getName(), "TensorShapeRank2");
  ASSERT_EQ(shapeType.getNumFields(), 2u);
  EXPECT_EQ(shapeType.getFields()[0].name, "dim0");
  EXPECT_TRUE(shapeType.getFields()[0].type.isInteger(64));
  EXPECT_EQ(shapeType.getFields()[1].name, "dim1");
  EXPECT_TRUE(shapeType.getFields()[1].type.isInteger(64));
}

} // namespace
