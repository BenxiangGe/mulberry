//===--- MulberryTypes.cpp - Mulberry core dialect types -----------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "mulberry/MLIRGen/IR/MulberryTypes.h"

#include "mulberry/MLIRGen/IR/MulberryDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include <set>

using namespace mlir;
using namespace mlir::mulberry_core;

#define GET_TYPEDEF_CLASSES
#include "mulberry/MLIRGen/IR/MulberryOpsTypes.cpp.inc"

static ParseResult parseField(AsmParser& parser,
                              RecordType::Field& field) {
  StringRef name;
  Type type;
  if (parser.parseKeyword(&name) || parser.parseColon() ||
      parser.parseType(type))
    return failure();

  field = RecordType::Field{name.str(), type};
  return success();
}

Type RecordType::parse(AsmParser& parser) {
  StringRef name;
  if (parser.parseLess() || parser.parseKeyword(&name))
    return {};

  FieldVector fields;
  auto parseFieldFn = [&]() {
    fields.emplace_back();
    return parseField(parser, fields.back());
  };
  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Braces,
                                     parseFieldFn) ||
      parser.parseGreater())
    return {};

  return RecordType::get(parser.getContext(), name, fields);
}

void RecordType::print(AsmPrinter& printer) const {
  printer << "<" << getName() << " {";
  llvm::interleaveComma(getFields(), printer,
                        [&](const Field& field) {
                          printer << field.name << ": ";
                          printer.printType(field.type);
                        });
  printer << "}>";
}

static auto parseShapedElementType(AsmParser& parser,
                                   SmallVectorImpl<int64_t>& shape,
                                   Type& elementType) -> ParseResult {
  if (parser.parseLess())
    return failure();

  if (parser.parseDimensionList(shape, /*allowDynamic=*/true,
                                /*withTrailingX=*/true))
    return failure();

  if (parser.parseType(elementType) || parser.parseGreater())
    return failure();

  return success();
}

static void printShapedElementType(AsmPrinter& printer,
                                   ArrayRef<int64_t> shape,
                                   Type elementType) {
  printer << "<";
  for (const auto& dim : llvm::enumerate(shape)) {
    if (dim.index() != 0)
      printer << "x";
    if (dim.value() < 0)
      printer << "?";
    else
      printer << dim.value();
  }
  printer << "x";
  printer.printType(elementType);
  printer << ">";
}

Type mlir::mulberry_core::TensorType::parse(AsmParser& parser) {
  SmallVector<int64_t> shape;
  Type elementType;
  if (parseShapedElementType(parser, shape, elementType))
    return {};

  return mlir::mulberry_core::TensorType::get(parser.getContext(), shape,
                                         elementType);
}

void mlir::mulberry_core::TensorType::print(AsmPrinter& printer) const {
  printShapedElementType(printer, getShape(), getElementType());
}

LogicalResult RecordType::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, StringRef name,
    ArrayRef<Field> fields) {
  if (name.empty())
    return emitError() << "record type name cannot be empty";

  std::set<std::string> fieldNames;
  for (const Field& field : fields) {
    if (field.name.empty())
      return emitError() << "record field name cannot be empty";
    if (!field.type)
      return emitError() << "record field type cannot be empty";
    if (!fieldNames.insert(field.name).second)
      return emitError() << "duplicate record field `" << field.name << "`";
  }

  return success();
}

Type RecordType::getFieldType(StringRef name) const {
  auto fields = getFields();
  for (const Field& field : fields)
    if (field.name == name)
      return field.type;
  return {};
}

Type RecordType::getFieldType(unsigned index) const {
  assert(index < getNumFields());
  return getFields()[index].type;
}

unsigned RecordType::getFieldIndex(StringRef name) const {
  for (const auto& field : llvm::enumerate(getFields()))
    if (field.value().name == name)
      return field.index();
  return std::numeric_limits<unsigned>::max();
}

void MulberryDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mulberry/MLIRGen/IR/MulberryOpsTypes.cpp.inc"
      >();
}
