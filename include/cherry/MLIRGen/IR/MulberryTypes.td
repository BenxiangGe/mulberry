//===--- MulberryTypes.td - Mulberry dialect types ---------*- tablegen -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_TYPES
#define MULBERRY_TYPES

include "cherry/MLIRGen/IR/MulberryDialect.td"
include "mlir/IR/AttrTypeBase.td"

class Mulberry_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Mulberry_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Mulberry_RecordType : Mulberry_Type<"Record", "record"> {
  let summary = "Mulberry record type";
  let description = [{
    A minimal named-field aggregate type for Mulberry values.
  }];

  let parameters = (ins
    StringRefParameter<"record name">:$name,
    ArrayRefParameter<"RecordType::Field">:$fields
  );

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    struct Field {
      std::string name;
      mlir::Type type;
    };

    using FieldList = llvm::ArrayRef<Field>;
    using FieldVector = llvm::SmallVector<Field>;

    mlir::Type getFieldType(llvm::StringRef name) const;
    mlir::Type getFieldType(unsigned index) const;
    unsigned getFieldIndex(llvm::StringRef name) const;
    unsigned getNumFields() const { return getFields().size(); }
  }];
}

#endif // MULBERRY_TYPES
