add_mlir_dialect_library(MulberryNNIR
        MulberryNNDialect.cpp
        MulberryNNOps.cpp

        DEPENDS
        MLIRMulberryNNOpsIncGen
        MLIRMulberryOpsIncGen

        LINK_LIBS PUBLIC
        cherryMLIRGen
        MLIRIR
        MLIRInferTypeOpInterface
)

