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

def Mulberry_TensorType : Mulberry_Type<"Tensor", "tensor"> {
  let summary = "Mulberry tensor type";
  let description = [{
    A writable shaped numeric data value in high-level Mulberry IR.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "mlir::Type":$elementType
  );

  let hasCustomAssemblyFormat = 1;
}

def Mulberry_TensorDescType : Mulberry_Type<"TensorDesc", "tensor_desc"> {
  let summary = "Mulberry tensor descriptor type";
  let description = [{
    Lowering-internal descriptor for a Mulberry tensor ABI/view value. This is
    not a source-level Tensor heap object handle and must not cross function
    boundaries. The rank is static; dynamic dimensions are carried as
    descriptor fields.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "mlir::Type":$elementType
  );

  let hasCustomAssemblyFormat = 1;
}

def Mulberry_PtrType : Mulberry_Type<"Ptr", "ptr"> {
  let summary = "Mulberry pointer type";
  let description = [{
    Typed address of a Mulberry value. This is an IR type, not a user-visible
    language type. It keeps the pointee type until lowering to opaque pointer
    dialect or LLVM pointer types.
  }];

  let parameters = (ins "mlir::Type":$pointeeType);
  let assemblyFormat = "`<` $pointeeType `>`";
}

#endif // MULBERRY_TYPES
