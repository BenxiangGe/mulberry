set(LLVM_TARGET_DEFINITIONS MulberryPasses.td)
mlir_tablegen(MulberryPasses.h.inc -gen-pass-decls -name MulberryConversion)
add_public_tablegen_target(MLIRMulberryConversionPassIncGen)

add_mlir_doc(MulberryPasses MulberryConversionPasses ./mulberry/ -gen-pass-doc)
