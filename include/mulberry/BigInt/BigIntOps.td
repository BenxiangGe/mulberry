//===--- BigIntOps.td - BigInt semantic operations --------*- tablegen -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_BIGINT_OPS
#define MULBERRY_BIGINT_OPS

include "mulberry/BigInt/BigIntDialect.td"
include "mulberry/BigInt/BigIntTypes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def BigInt_AnyIntType : Type<
    CPred<"::llvm::isa<::mlir::bigint::IntType>($_self)">,
    "BigInt semantic integer type">;

def BigInt_ConstantOp : BigInt_Op<"constant", [Pure]> {
  let summary = "Create a semantic arbitrary-precision integer constant";
  let description = [{
    Preserves the source integer spelling until BigInt lowering chooses a
    runtime representation. The value is not parsed into a fixed-width MLIR
    integer at this stage.
  }];

  let arguments = (ins StrAttr:$value);
  let results = (outs BigInt_IntType:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def BigInt_FromUInt64Op : BigInt_Op<"from_uint64", [Pure]> {
  let summary = "Widen a UInt64 into an arbitrary-precision integer";

  let arguments = (ins I64:$value);
  let results = (outs BigInt_IntType:$result);

  let assemblyFormat = "$value attr-dict `:` type($value) `->` "
                       "type($result)";
}

class BigInt_BinaryOp<string mnemonic> : BigInt_Op<mnemonic,
                                                    [Pure,
                                                     SameOperandsAndResultType]> {
  let arguments = (ins BigInt_AnyIntType:$lhs, BigInt_AnyIntType:$rhs);
  let results = (outs BigInt_IntType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `->` "
                       "type($result)";
}

def BigInt_AddOp : BigInt_BinaryOp<"add"> {
  let summary = "Add two arbitrary-precision integers";
}

def BigInt_SubOp : BigInt_BinaryOp<"sub"> {
  let summary = "Subtract two arbitrary-precision integers";
}

def BigInt_MulOp : BigInt_BinaryOp<"mul"> {
  let summary = "Multiply two arbitrary-precision integers";
}

def BigInt_AndOp : BigInt_BinaryOp<"and"> {
  let summary = "Apply two's-complement bitwise AND";
}

def BigInt_OrOp : BigInt_BinaryOp<"or"> {
  let summary = "Apply two's-complement bitwise OR";
}

def BigInt_XorOp : BigInt_BinaryOp<"xor"> {
  let summary = "Apply two's-complement bitwise XOR";
}

class BigInt_ShiftOp<string mnemonic> : BigInt_Op<mnemonic, [Pure]> {
  let arguments = (ins BigInt_AnyIntType:$value, I64:$count);
  let results = (outs BigInt_IntType:$result);

  let assemblyFormat = "$value `,` $count attr-dict `:` type($value) `,` "
                       "type($count) `->` type($result)";
}

def BigInt_ShiftLeftOp : BigInt_ShiftOp<"shift_left"> {
  let summary = "Shift an arbitrary-precision integer left";
}

def BigInt_ShiftRightOp : BigInt_ShiftOp<"shift_right"> {
  let summary = "Arithmetic-shift an arbitrary-precision integer right";
}

def BigInt_NegOp : BigInt_Op<"neg", [Pure, SameOperandsAndResultType]> {
  let summary = "Negate an arbitrary-precision integer";

  let arguments = (ins BigInt_AnyIntType:$input);
  let results = (outs BigInt_IntType:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def BigInt_CmpOp : BigInt_Op<"cmp", [Pure]> {
  let summary = "Compare two arbitrary-precision integers";
  let description = [{
    The predicate is one of `eq`, `ne`, `lt`, `le`, `gt`, or `ge`.
  }];

  let arguments = (ins StrAttr:$predicate,
                       BigInt_AnyIntType:$lhs,
                       BigInt_AnyIntType:$rhs);
  let results = (outs I1:$result);

  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` "
                       "type($lhs) `,` type($rhs) `->` type($result)";
  let hasVerifier = 1;
}

#endif // MULBERRY_BIGINT_OPS
