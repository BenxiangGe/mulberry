//===--- MulberryOps.td - Mulberry core dialect ops -------------*- tablegen -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_OPS
#define MULBERRY_OPS

include "mulberry/MLIRGen/IR/MulberryDialect.td"
include "mulberry/MLIRGen/IR/MulberryTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Mulberry_AnyPtrType : Type<
    CPred<"::llvm::isa<::mlir::mulberry_core::PtrType>($_self)">,
    "Mulberry pointer type">;

def Mulberry_AnyTensorType : Type<
    CPred<"::llvm::isa<::mlir::mulberry_core::TensorType>($_self)">,
    "Mulberry internal tensor type">;

def Mulberry_AnyRecordType : Type<
    CPred<"::llvm::isa<::mlir::mulberry_core::RecordType>($_self)">,
    "Mulberry record type">;

def Mulberry_AllocaOp : Mulberry_Op<"alloca"> {
  let summary = "Allocate a Mulberry value";

  let arguments = (ins TypeAttr:$elementType);
  let results = (outs Mulberry_AnyPtrType:$result);

  let assemblyFormat = "$elementType attr-dict `:` type($result)";

  let hasVerifier = 1;
}

def Mulberry_HeapAllocOp : Mulberry_Op<"heap.alloc"> {
  let summary = "Allocate a Mulberry heap object";

  let arguments = (ins TypeAttr:$elementType, Index:$count);
  let results = (outs Mulberry_AnyPtrType:$result);

  let assemblyFormat = "$elementType `,` $count attr-dict `:` type($result)";

  let hasVerifier = 1;
}

def Mulberry_PtrIndexOp : Mulberry_Op<"ptr.index"> {
  let summary = "Get the address of a pointer element";

  let arguments = (ins Mulberry_AnyPtrType:$ptr, Index:$index);
  let results = (outs Mulberry_AnyPtrType:$result);

  let assemblyFormat = "$ptr `[` $index `]` attr-dict `:` type($ptr) "
                       "`->` type($result)";

  let hasVerifier = 1;
}

def Mulberry_PtrCastOp : Mulberry_Op<"ptr.cast"> {
  let summary = "Reinterpret a Mulberry pointer type";

  let arguments = (ins Mulberry_AnyPtrType:$ptr);
  let results = (outs Mulberry_AnyPtrType:$result);

  let assemblyFormat = "$ptr attr-dict `:` type($ptr) `to` type($result)";
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

def Mulberry_RecordExtractOp : Mulberry_Op<"record.extract"> {
  let summary = "Extract a Mulberry record field value";

  let arguments = (ins
    Mulberry_AnyRecordType:$record,
    StrAttr:$field
  );
  let results = (outs AnyType:$result);

  let assemblyFormat = "$record `[` $field `]` attr-dict `:` type($record) "
                       "`->` type($result)";

  let hasVerifier = 1;
}

def Mulberry_TensorViewOp : Mulberry_Op<"tensor.view"> {
  let summary = "View a stdlib Tensor value as an internal tensor";

  let arguments = (ins Mulberry_AnyRecordType:$tensor);
  let results = (outs Mulberry_AnyTensorType:$result);

  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `->` "
                       "type($result)";

  let hasVerifier = 1;
}

def Mulberry_TensorPackOp : Mulberry_Op<"tensor.pack"> {
  let summary = "Pack an internal tensor into a stdlib Tensor value";

  let arguments = (ins Mulberry_AnyTensorType:$tensor);
  let results = (outs Mulberry_AnyRecordType:$result);

  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `->` "
                       "type($result)";

  let hasVerifier = 1;
}

#endif // MULBERRY_OPS
