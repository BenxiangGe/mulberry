set(LIB_HEADERS ${CHERRY_SOURCE_DIR}/include/cherry/MLIRGen)

add_mlir_dialect_library(cherryMLIRGen
        ${LIB_HEADERS}/MLIRGen.h
        MLIRGen.cpp

        # Conversion
        ${LIB_HEADERS}/Conversion/CherryPasses.h
        Conversion/ConvertCherryToArithCfFunc.cpp
        Conversion/ConvertCherryNNToLinalg.cpp
        Conversion/ConvertCherryToLLVM.cpp
        Conversion/ConvertCherryToSCF.cpp

        # IR
        ${LIB_HEADERS}/IR/CherryDialect.h
        IR/CherryDialect.cpp
        ${LIB_HEADERS}/IR/CherryOps.h
        IR/CherryOps.cpp
        ${LIB_HEADERS}/IR/CherryNNDialect.h
        IR/CherryNNDialect.cpp
        ${LIB_HEADERS}/IR/CherryNNOps.h
        IR/CherryNNOps.cpp
        ${LIB_HEADERS}/IR/CherryTypes.h
        IR/CherryTypes.cpp

        DEPENDS
        MLIRCherryOpsIncGen
        MLIRCherryNNOpsIncGen
        MLIRCherryConversionPassIncGen

        LINK_LIBS PUBLIC
        MLIRArithDialect
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        MLIRLinalgDialect
        MLIRLinalgTransforms
        MLIRMathDialect
        MLIRMathToLLVM
        MLIRMemRefDialect
        MLIRSCFDialect
        clangCIR
)

add_subdirectory(CAPI)
