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
include "mlir/Interfaces/SideEffectInterfaces.td"

def Mulberry_AnyPtrType : Type<
    CPred<"::llvm::isa<::mlir::mulberry::PtrType>($_self)">,
    "Mulberry pointer type">;

def Mulberry_AnyTensorType : Type<
    CPred<"::llvm::isa<::mlir::mulberry::TensorType>($_self)">,
    "Mulberry tensor type">;

def Mulberry_AnyRecordType : Type<
    CPred<"::llvm::isa<::mlir::mulberry::RecordType>($_self)">,
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

def Mulberry_TensorAllocOp : Mulberry_Op<"tensor.alloc"> {
  let summary = "Allocate a Mulberry tensor";

  let arguments = (ins Variadic<Index>:$dynamicSizes);
  let results = (outs Mulberry_AnyTensorType:$result);

  let assemblyFormat = "`(` $dynamicSizes `)` attr-dict `:` type($result)";

  let hasVerifier = 1;
}

def Mulberry_TensorDimOp : Mulberry_Op<"tensor.dim"> {
  let summary = "Get a Mulberry tensor dimension";

  let arguments = (ins Mulberry_AnyTensorType:$tensor, Index:$index);
  let results = (outs Index:$result);

  let assemblyFormat = "$tensor `,` $index attr-dict `:` type($tensor)";

  let hasVerifier = 1;
}

def Mulberry_TensorCastOp : Mulberry_Op<"tensor.cast"> {
  let summary = "Cast a Mulberry tensor shape";

  let arguments = (ins Mulberry_AnyTensorType:$source);
  let results = (outs Mulberry_AnyTensorType:$dest);

  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($dest)";

  let hasVerifier = 1;
}

def Mulberry_TensorViewOp : Mulberry_Op<"tensor.view"> {
  let summary = "View a stdlib Tensor value as a Mulberry tensor";

  let arguments = (ins Mulberry_AnyRecordType:$tensor);
  let results = (outs Mulberry_AnyTensorType:$result);

  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `->` "
                       "type($result)";

  let hasVerifier = 1;
}

def Mulberry_TensorPackOp : Mulberry_Op<"tensor.pack"> {
  let summary = "Pack a Mulberry tensor into a stdlib Tensor value";

  let arguments = (ins Mulberry_AnyTensorType:$tensor);
  let results = (outs Mulberry_AnyRecordType:$result);

  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `->` "
                       "type($result)";

  let hasVerifier = 1;
}

def Mulberry_TensorLoadOp : Mulberry_Op<"tensor.load"> {
  let summary = "Load a Mulberry tensor element";

  let arguments = (ins Mulberry_AnyTensorType:$tensor, Variadic<Index>:$indices);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$tensor `[` $indices `]` attr-dict `:` type($tensor) "
                       "`->` type($result)";

  let hasVerifier = 1;
}

def Mulberry_TensorStoreOp : Mulberry_Op<"tensor.store"> {
  let summary = "Store a Mulberry tensor element";

  let arguments = (ins AnyType:$value, Mulberry_AnyTensorType:$tensor,
                       Variadic<Index>:$indices);

  let assemblyFormat = "$value `,` $tensor `[` $indices `]` attr-dict `:` "
                       "type($value) `,` type($tensor)";

  let hasVerifier = 1;
}

#endif // MULBERRY_OPS
