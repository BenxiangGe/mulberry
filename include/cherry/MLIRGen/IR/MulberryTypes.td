//===--- MulberryTypes.td - Mulberry dialect types --------*- tablegen -*-===//
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

def ListType : Mulberry_Type<"List", "list"> {
  let summary = "Mulberry list value type";
  let parameters = (ins "mlir::Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}

#endif // MULBERRY_TYPES
