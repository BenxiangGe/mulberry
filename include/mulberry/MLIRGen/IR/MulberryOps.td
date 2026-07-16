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
include "mlir/IR/CommonTypeConstraints.td"
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

  let arguments = (ins TypeAttr:$elementType, I64:$count);
  let results = (outs Mulberry_AnyPtrType:$result);

  let assemblyFormat = "$elementType `,` $count attr-dict `:` type($result)";

  let hasVerifier = 1;
}

def Mulberry_TensorStorageAllocOp : Mulberry_Op<"tensor.storage_alloc"> {
  let summary = "Allocate source-owned Tensor storage";

  let arguments = (ins TypeAttr:$elementType, I64:$count);
  let results = (outs Mulberry_AnyPtrType:$storage,
                      Mulberry_AnyTensorType:$payload);

  let assemblyFormat = "$elementType `,` $count attr-dict `:` type($storage) "
                       "`,` type($payload)";

  let hasVerifier = 1;
}

def Mulberry_TensorStorageAllocLoweredOp
    : Mulberry_Op<"tensor.storage_alloc.lowered"> {
  let summary = "Storage-level source-owned Tensor allocation";

  let arguments = (ins TypeAttr:$storageLayout, TypeAttr:$elementType,
                       I64:$count);
  let results = (outs AnyType:$storage,
                      Res<AnyMemRef, "",
                          [MemAlloc<DefaultResource>]>:$payload);

  let assemblyFormat = "$storageLayout `,` $elementType `,` $count attr-dict "
                       "`:` type($storage) `,` type($payload)";

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

def Mulberry_TensorViewOp : Mulberry_Op<"tensor.view"> {
  let summary = "View a stdlib Tensor record ABI value as an internal tensor";

  let arguments = (ins Mulberry_AnyRecordType:$tensorRecord);
  let results = (outs Mulberry_AnyTensorType:$result);

  let assemblyFormat = "$tensorRecord attr-dict `:` type($tensorRecord) `->` "
                       "type($result)";

  let hasVerifier = 1;
}

def Mulberry_TensorPackOp : Mulberry_Op<"tensor.pack"> {
  let summary = "Pack an internal tensor into a stdlib Tensor record ABI value";

  let arguments = (ins Mulberry_AnyTensorType:$tensor);
  let results = (outs Mulberry_AnyRecordType:$tensorRecord);

  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `->` "
                       "type($tensorRecord)";

  let hasVerifier = 1;
}

def Mulberry_TensorDisposeOp : Mulberry_Op<"tensor.dispose"> {
  let summary = "Dispose a source Tensor object's shared payload";

  let arguments = (ins Mulberry_AnyPtrType:$tensor);

  let assemblyFormat = "$tensor attr-dict `:` type($tensor)";

  let hasVerifier = 1;
}

def Mulberry_TensorAssertAliveOp : Mulberry_Op<"tensor.assert_alive"> {
  let summary = "Reject access through a disposed source Tensor object";

  let arguments = (ins Mulberry_AnyPtrType:$tensor);

  let assemblyFormat = "$tensor attr-dict `:` type($tensor)";

  let hasVerifier = 1;
}

#endif // MULBERRY_OPS
