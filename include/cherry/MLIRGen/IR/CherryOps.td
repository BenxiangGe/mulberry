//===--- CherryOps.h - Cherry dialect ops -----------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_OPS
#define CHERRY_OPS

include "cherry/MLIRGen/IR/CherryTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

def Cherry_Type : AnyTypeOf<[I64, I1, Cherry_Struct]>;

def PrintOp : Cherry_Op<"print"> {
  let summary = "Print operation";
  let description = [{
      The "print" builtin operation prints a given input
  }];

  let arguments = (ins AnyTypeOf<[I64, I64MemRef]>:$input);
  let results = (outs I64:$result);

  let assemblyFormat = "$input attr-dict `:` functional-type($input, $result)";
}

def CastOp : Cherry_Op<"cast"> {
  let summary = "Cast operation";
  let description = [{
    Casts from I64 to I1
  }];

  let arguments = (ins AnyTypeOf<[I1, I1MemRef]>:$input);
  let results = (outs I64:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($result)";
}

def IfOp : Cherry_Op<"if"> {
  let summary = "If-then-else operation";

  let arguments = (ins I1:$condition);
  let results = (outs Cherry_Type:$result);
  let regions = (region SizedRegion<1>:$thenRegion, SizedRegion<1>:$elseRegion);

  let builders = [
    OpBuilder<(ins "Type":$resultType, "Value":$cond,
                    "function_ref<void(OpBuilder &, Location)>":$thenBuilder,
                    "function_ref<void(OpBuilder &, Location)>":$elseBuilder)>
  ];

  let assemblyFormat = [{
    $condition `:` type($result) $thenRegion `else` $elseRegion attr-dict 
  }];
}

def YieldIfOp : Cherry_Op<"yield_if", [Pure, Terminator]> {
  let summary = "Yield if operation";

  let arguments = (ins Variadic<Cherry_Type>:$result);

  let builders = [
    OpBuilder<(ins ), [{ /* nothing to do */ }]>
  ];

  let assemblyFormat = "($result^ `:` type($result))? attr-dict";
}

def WhileOp : Cherry_Op<"while"> {
  let summary = "While operation";

  let regions = (region SizedRegion<1>:$conditionRegion, SizedRegion<1>:$bodyRegion);

  let builders = [
    OpBuilder<(ins "function_ref<void(OpBuilder &, Location)>":$conditionBuilder,
                    "function_ref<void(OpBuilder &, Location)>":$bodyBuilder)>
  ];

  let assemblyFormat = "$conditionRegion `do` $bodyRegion attr-dict";
}

def YieldWhileOp : Cherry_Op<"yield_while", [Pure, Terminator]> {
  let summary = "Yield while operation";

  let arguments = (ins Variadic<Cherry_Type>:$result);

  let builders = [
    OpBuilder<(ins ), [{ /* nothing to do */ }]>
  ];

  let assemblyFormat = "($result^ `:` type($result))? attr-dict";
}

def StructInitOp : Cherry_Op<"struct_init"> {
  let summary = "Struct initializer";

  let arguments = (ins Variadic<Cherry_Type>:$values);
  let results = (outs Cherry_Struct:$result);

  let assemblyFormat = [{
    `(` $values `)` attr-dict `:` functional-type($values, $result)
  }];
}

def StructReadOp : Cherry_Op<"struct_read", [Pure]> {
  let summary = "Struct read";
  
  let arguments = (ins Cherry_Struct:$structValue, I64Attr:$index);
  let results = (outs Cherry_Type:$result);

  let builders = [
    OpBuilder<(ins "Value":$structValue, "int64_t":$index)>
  ];

  let assemblyFormat = [{
    $structValue `[` $index `]` attr-dict `:` functional-type($structValue, $result)
  }];
}

def StructWriteOp : Cherry_Op<"struct_write", [Pure]> {
  let summary = "Struct write";
  
  let arguments = (ins Cherry_Struct:$structValue, DenseI64ArrayAttr:$indexes, Cherry_Type:$valueToStore);
  let results = (outs Cherry_Struct:$result);

  let builders = [
     OpBuilder<(ins "Value":$structValue, "ArrayRef<int64_t>":$indexes, "Value":$valueToStore)>
  ];

  let assemblyFormat = [{
     $valueToStore `,` $structValue $indexes attr-dict `:` functional-type(operands, $result)
  }];
}

#endif // CHERRY_OPS
