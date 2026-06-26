add_mlir_dialect(MulberryNNOps mulberry_nn)
add_mlir_doc(MulberryNNDialect MulberryNNDialect mulberry_nn/ -gen-dialect-doc)
add_mlir_doc(MulberryNNOps MulberryNNOps mulberry_nn/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS MulberryNNPasses.td)
mlir_tablegen(MulberryNNPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRMulberryNNPassesIncGen)
add_mlir_doc(MulberryNNPasses MulberryNNPasses mulberry_nn/ -gen-pass-doc)
