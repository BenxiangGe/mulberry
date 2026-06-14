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

def Mulberry_AnyStringType : Type<
    CPred<"::llvm::isa<::mlir::mulberry::StringType>($_self)">,
    "Mulberry string type">;

def Mulberry_AnyFileType : Type<
    CPred<"::llvm::isa<::mlir::mulberry::FileType>($_self)">,
    "Mulberry file type">;

def Mulberry_AnyTensorDescType : Type<
    CPred<"::llvm::isa<::mlir::mulberry::TensorDescType>($_self)">,
    "Mulberry tensor descriptor type">;

def Mulberry_AnyTensorHandleType : Type<
    CPred<"::llvm::isa<::mlir::mulberry::TensorHandleType>($_self)">,
    "Mulberry tensor handle type">;

def Mulberry_AnyListType : Type<
    CPred<"::llvm::isa<::mlir::mulberry::ListType>($_self)">,
    "Mulberry list type">;

def Mulberry_AnyListStorageType : Type<
    CPred<"::llvm::isa<::mlir::mulberry::ListStorageType>($_self)">,
    "Mulberry list storage type">;

def Mulberry_AnyListDescType : Type<
    CPred<"::llvm::isa<::mlir::mulberry::ListDescType>($_self)">,
    "Mulberry list descriptor type">;

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

def Mulberry_StringLiteralOp : Mulberry_Op<"string.literal"> {
  let summary = "Create a Mulberry string literal";

  let arguments = (ins StrAttr:$value);
  let results = (outs Mulberry_AnyStringType:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";
}

def Mulberry_FileOpenOp : Mulberry_Op<"file.open"> {
  let summary = "Open a Mulberry file";

  let arguments = (ins Mulberry_AnyStringType:$path,
                       Mulberry_AnyStringType:$mode);
  let results = (outs Mulberry_AnyFileType:$result);

  let assemblyFormat = "$path `,` $mode attr-dict `:` type($path) `,` "
                       "type($mode) `->` type($result)";
}

def Mulberry_FileReadOp : Mulberry_Op<"file.read"> {
  let summary = "Read bytes from a Mulberry file";

  let arguments = (ins Mulberry_AnyFileType:$file,
                       Mulberry_AnyTensorType:$buffer);
  let results = (outs I64:$result);

  let assemblyFormat = "$file `,` $buffer attr-dict `:` type($file) `,` "
                       "type($buffer)";

  let hasVerifier = 1;
}

def Mulberry_FileWriteOp : Mulberry_Op<"file.write"> {
  let summary = "Write bytes to a Mulberry file";

  let arguments = (ins Mulberry_AnyFileType:$file,
                       Mulberry_AnyTensorType:$buffer);
  let results = (outs I64:$result);

  let assemblyFormat = "$file `,` $buffer attr-dict `:` type($file) `,` "
                       "type($buffer)";

  let hasVerifier = 1;
}

def Mulberry_FileCloseOp : Mulberry_Op<"file.close"> {
  let summary = "Close a Mulberry file";

  let arguments = (ins Mulberry_AnyFileType:$file);
  let results = (outs I64:$result);

  let assemblyFormat = "$file attr-dict `:` type($file)";
}

def Mulberry_SafetensorReadOp : Mulberry_Op<"safetensor.read"> {
  let summary = "Read a named tensor from a safetensors file";

  let arguments = (ins Mulberry_AnyFileType:$file,
                       Mulberry_AnyStringType:$name);
  let results = (outs Mulberry_AnyTensorType:$result);

  let assemblyFormat = "$file `,` $name attr-dict `:` type($file) `,` "
                       "type($name) `->` type($result)";

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

def Mulberry_TensorDescPackOp : Mulberry_Op<"tensor.desc_pack"> {
  let summary = "Pack Mulberry tensor descriptor fields";

  let arguments = (ins Mulberry_AnyTensorType:$tensor);
  let results = (outs Mulberry_AnyTensorDescType:$result);

  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `->` "
                       "type($result)";

  let hasVerifier = 1;
}

def Mulberry_TensorDescUnpackOp : Mulberry_Op<"tensor.desc_unpack"> {
  let summary = "Reconstruct a Mulberry tensor from descriptor fields";

  let arguments = (ins Mulberry_AnyTensorDescType:$desc);
  let results = (outs Mulberry_AnyTensorType:$result);

  let assemblyFormat = "$desc attr-dict `:` type($desc) `->` "
                       "type($result)";

  let hasVerifier = 1;
}

def Mulberry_TensorHandleFromDescOp
    : Mulberry_Op<"tensor.handle_from_desc"> {
  let summary = "Reconstruct a Mulberry tensor handle from descriptor fields";

  let arguments = (ins Mulberry_AnyTensorDescType:$desc);
  let results = (outs Mulberry_AnyTensorHandleType:$result);

  let assemblyFormat = "$desc attr-dict `:` type($desc) `->` "
                       "type($result)";

  let hasVerifier = 1;
}

def Mulberry_TensorDescToABIOp : Mulberry_Op<"tensor.desc_to_abi"> {
  let summary = "Materialize a Mulberry tensor descriptor backend ABI value";

  let arguments = (ins Mulberry_AnyTensorDescType:$desc);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$desc attr-dict `:` type($desc) `->` "
                       "type($result)";

  let hasVerifier = 1;
}

def Mulberry_ListCreateOp : Mulberry_Op<"list.create"> {
  let summary = "Create a Mulberry list value";

  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs Mulberry_AnyListType:$result);

  let assemblyFormat = "`(` $elements `)` attr-dict `:` "
                       "functional-type($elements, $result)";

  let hasVerifier = 1;
}

def Mulberry_ListGetOp : Mulberry_Op<"list.get"> {
  let summary = "Get a Mulberry list element";

  let arguments = (ins Mulberry_AnyListType:$list, Index:$index);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$list `[` $index `]` attr-dict `:` type($list) "
                       "`->` type($result)";

  let hasVerifier = 1;
}

def Mulberry_ListSizeOp : Mulberry_Op<"list.size"> {
  let summary = "Get a Mulberry list size";

  let arguments = (ins Mulberry_AnyListType:$list);
  let results = (outs I64:$result);

  let assemblyFormat = "$list attr-dict `:` type($list)";
}

def Mulberry_ListToDescOp : Mulberry_Op<"list.to_desc"> {
  let summary = "Materialize a Mulberry list descriptor";

  let arguments = (ins Mulberry_AnyListType:$list);
  let results = (outs Mulberry_AnyListDescType:$result);

  let assemblyFormat = "$list attr-dict `:` type($list) `->` "
                       "type($result)";

  let hasVerifier = 1;
}

def Mulberry_ListAllocOp : Mulberry_Op<"list.alloc"> {
  let summary = "Allocate Mulberry list storage";

  let arguments = (ins Index:$length);
  let results = (outs Mulberry_AnyListStorageType:$result);

  let assemblyFormat = "$length attr-dict `:` type($result)";
}

def Mulberry_ListEscapeStorageOp : Mulberry_Op<"list.escape_storage"> {
  let summary = "Mark Mulberry list storage as escaping";

  let arguments = (ins Mulberry_AnyListStorageType:$storage, Index:$length);
  let results = (outs Mulberry_AnyListStorageType:$result);

  let assemblyFormat = "$storage `,` $length attr-dict `:` type($storage) `->` "
                       "type($result)";

  let hasVerifier = 1;
}

def Mulberry_ListLoadOp : Mulberry_Op<"list.load"> {
  let summary = "Load a Mulberry list storage element";

  let arguments = (ins Mulberry_AnyListStorageType:$storage, Index:$index);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$storage `[` $index `]` attr-dict `:` type($storage) "
                       "`->` type($result)";

  let hasVerifier = 1;
}

def Mulberry_ListStoreOp : Mulberry_Op<"list.store"> {
  let summary = "Store a Mulberry list storage element";

  let arguments = (ins AnyType:$value, Mulberry_AnyListStorageType:$storage,
                       Index:$index);

  let assemblyFormat = "$value `,` $storage `[` $index `]` attr-dict `:` "
                       "type($value) `,` type($storage)";

  let hasVerifier = 1;
}

def Mulberry_ListLengthOp : Mulberry_Op<"list.length"> {
  let summary = "Get Mulberry list storage length";

  let arguments = (ins Mulberry_AnyListStorageType:$storage);
  let results = (outs Index:$result);

  let assemblyFormat = "$storage attr-dict `:` type($storage)";
}

def Mulberry_ListDescPackOp : Mulberry_Op<"list.desc_pack"> {
  let summary = "Pack Mulberry list descriptor fields";

  let arguments = (ins Index:$length, Mulberry_AnyListStorageType:$data);
  let results = (outs Mulberry_AnyListDescType:$result);

  let assemblyFormat = "$length `,` $data attr-dict `:` type($data) "
                       "`->` type($result)";

  let hasVerifier = 1;
}

def Mulberry_ListDescLengthOp : Mulberry_Op<"list.desc_length"> {
  let summary = "Get Mulberry list descriptor length";

  let arguments = (ins Mulberry_AnyListDescType:$desc);
  let results = (outs Index:$result);

  let assemblyFormat = "$desc attr-dict `:` type($desc)";
}

def Mulberry_ListDescDataOp : Mulberry_Op<"list.desc_data"> {
  let summary = "Get Mulberry list descriptor data storage";

  let arguments = (ins Mulberry_AnyListDescType:$desc);
  let results = (outs Mulberry_AnyListStorageType:$result);

  let assemblyFormat = "$desc attr-dict `:` type($desc) `->` type($result)";

  let hasVerifier = 1;
}

def Mulberry_ListDescGetOp : Mulberry_Op<"list.desc_get"> {
  let summary = "Get a Mulberry list descriptor element";

  let arguments = (ins Mulberry_AnyListDescType:$desc, Index:$index);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$desc `[` $index `]` attr-dict `:` type($desc) "
                       "`->` type($result)";

  let hasVerifier = 1;
}

def Mulberry_ListDescToABIOp : Mulberry_Op<"list.desc_to_abi"> {
  let summary = "Materialize a Mulberry list descriptor backend ABI value";

  let arguments = (ins Mulberry_AnyListDescType:$desc);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$desc attr-dict `:` type($desc) `->` type($result)";

  let hasVerifier = 1;
}

#endif // MULBERRY_OPS
