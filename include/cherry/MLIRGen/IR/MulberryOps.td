//===--- MulberryOps.td - Mulberry dialect ops -------------*- tablegen -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_OPS
#define MULBERRY_OPS

include "cherry/MLIRGen/IR/MulberryDialect.td"
include "cherry/MLIRGen/IR/MulberryTypes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Mulberry_AnyPtrType : Type<
    CPred<"::llvm::isa<::mlir::mulberry::PtrType>($_self)">,
    "Mulberry pointer type">;

def Mulberry_RecordTypeConstraint : Type<
    CPred<"::llvm::isa<::mlir::mulberry::RecordType>($_self)">,
    "Mulberry record type">;

def Mulberry_TensorPackOp : Mulberry_Op<"tensor.pack", [Pure]> {
  let summary = "Pack a memref into a Mulberry tensor descriptor";

  let arguments = (ins AnyMemRef:$tensor);
  let results = (outs Mulberry_RecordTypeConstraint:$result);

  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `->` "
                       "type($result)";

  let hasVerifier = 1;
}

def Mulberry_TensorUnpackOp : Mulberry_Op<"tensor.unpack", [Pure]> {
  let summary = "Unpack a Mulberry tensor descriptor into a memref";

  let arguments = (ins Mulberry_RecordTypeConstraint:$tensor);
  let results = (outs AnyMemRef:$result);

  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `->` "
                       "type($result)";

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def Mulberry_RecordCreateOp : Mulberry_Op<"record.create", [Pure]> {
  let summary = "Create a Mulberry record value";

  let arguments = (ins Variadic<AnyType>:$fields);
  let results = (outs Mulberry_RecordTypeConstraint:$result);

  let assemblyFormat = "$fields attr-dict `:` functional-type($fields, "
                       "$result)";

  let hasVerifier = 1;
}

def Mulberry_AllocaOp : Mulberry_Op<"alloca"> {
  let summary = "Allocate a Mulberry value";

  let arguments = (ins TypeAttr:$elementType);
  let results = (outs Mulberry_AnyPtrType:$result);

  let assemblyFormat = "$elementType attr-dict `:` type($result)";

  let hasVerifier = 1;
}

def Mulberry_LoadOp : Mulberry_Op<"load"> {
  let summary = "Load a Mulberry value";

  let arguments = (ins Mulberry_AnyPtrType:$ptr);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$ptr attr-dict `:` type($ptr) `->` type($result)";

  let hasVerifier = 1;
}

def Mulberry_StoreOp : Mulberry_Op<"store"> {
  let summary = "Store a Mulberry value";

  let arguments = (ins AnyType:$value, Mulberry_AnyPtrType:$ptr);

  let assemblyFormat = "$value `,` $ptr attr-dict `:` type($value) `,` "
                       "type($ptr)";

  let hasVerifier = 1;
}

def Mulberry_RecordGetFieldOp : Mulberry_Op<"record.get_field"> {
  let summary = "Get the address of a Mulberry record field";

  let arguments = (ins
    Mulberry_AnyPtrType:$record,
    StrAttr:$field
  );
  let results = (outs Mulberry_AnyPtrType:$result);

  let assemblyFormat = "$record `[` $field `]` attr-dict `:` type($record) "
                       "`->` type($result)";

  let hasVerifier = 1;
}

#endif // MULBERRY_OPS
