add_mlir_dialect_library(MulberryNNConversion
        MulberryNNPasses.cpp
        MulberryNNToLinalgPatterns.cpp

        DEPENDS
        MLIRMulberryNNOpsIncGen
        MLIRMulberryNNPassesIncGen
        MLIRMulberryOpsIncGen

        LINK_LIBS PUBLIC
        MulberryNNIR
        cherryMLIRGen
        MLIRArithDialect
        MLIRLinalgDialect
        MLIRMathDialect
        MLIRMemRefDialect
        MLIRTransforms
)
