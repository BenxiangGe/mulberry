set(LIB_HEADERS ${CHERRY_SOURCE_DIR}/include/cherry/MLIRGen)

add_mlir_dialect_library(cherryMLIRGen
        ${LIB_HEADERS}/MLIRGen.h
        MLIRGen.cpp
        ${LIB_HEADERS}/TypeConverter.h
        TypeConverter.cpp

        # Conversion
        ${LIB_HEADERS}/Conversion/CherryPasses.h
        Conversion/ConvertCherryToArithCfFunc.cpp
        Conversion/ConvertCherryNNToLinalg.cpp
        Conversion/ConvertCherryToSCF.cpp
        Conversion/LowerMulberry.cpp

        # IR
        ${LIB_HEADERS}/IR/CherryDialect.h
        IR/CherryDialect.cpp
        ${LIB_HEADERS}/IR/CherryOps.h
        IR/CherryOps.cpp
        ${LIB_HEADERS}/IR/CherryNNDialect.h
        IR/CherryNNDialect.cpp
        ${LIB_HEADERS}/IR/CherryNNOps.h
        IR/CherryNNOps.cpp
        ${LIB_HEADERS}/IR/MulberryDialect.h
        IR/MulberryDialect.cpp
        ${LIB_HEADERS}/IR/MulberryOps.h
        IR/MulberryOps.cpp
        ${LIB_HEADERS}/IR/MulberryTypes.h
        IR/MulberryTypes.cpp

        DEPENDS
        MLIRCherryOpsIncGen
        MLIRCherryNNOpsIncGen
        MLIRMulberryOpsIncGen
        MLIRCherryConversionPassIncGen

        LINK_LIBS PUBLIC
        cherryBasic
        MLIRArithDialect
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        MLIRLinalgDialect
        MLIRLinalgTransforms
        MLIRMathDialect
        MLIRMemRefDialect
        MLIRSCFDialect
)

add_subdirectory(CAPI)
