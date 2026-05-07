add_mlir_dialect(CherryOps cherry)
add_mlir_doc(CherryDialect CherryDialect cherry/ -gen-dialect-doc)
add_mlir_doc(CherryOps CherryOps cherry/ -gen-op-doc)

add_mlir_dialect(CherryNNOps cherry_nn)
add_mlir_doc(CherryNNDialect CherryNNDialect cherry_nn/ -gen-dialect-doc)
add_mlir_doc(CherryNNOps CherryNNOps cherry_nn/ -gen-op-doc)
