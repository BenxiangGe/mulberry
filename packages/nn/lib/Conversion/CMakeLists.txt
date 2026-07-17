set(BUILD_SHARED_LIBS OFF)

add_mlir_dialect_library(MulberryNNConversion
        MulberryNNPasses.cpp
        MulberryNNToLinalgPatterns.cpp

        DEPENDS
        MLIRMulberryNNOpsIncGen
        MLIRMulberryNNPassesIncGen
        MLIRMulberryOpsIncGen

        LINK_LIBS PUBLIC
        MulberryNNIR
        mulberryMLIRGen
        MLIRArithDialect
        MLIRLinalgDialect
        MLIRMathDialect
        MLIRMemRefDialect
        MLIRSCFDialect
        MLIRTransforms
)
