set(LIB_HEADERS ${CHERRY_SOURCE_DIR}/include/cherry/MLIRGen)

add_mlir_dialect_library(cherryMLIRGen
        ${LIB_HEADERS}/MLIRGen.h
        MLIRGen.cpp
        ${LIB_HEADERS}/TypeConverter.h
        TypeConverter.cpp

        # Conversion
        ${LIB_HEADERS}/Conversion/CherryPasses.h
        Conversion/CherryNNToLinalgPatterns.cpp
        Conversion/ConvertCherryToLLVM.cpp
        Conversion/LowerMulberry.cpp

        # IR
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
        MLIRCherryNNOpsIncGen
        MLIRMulberryOpsIncGen
        MLIRCherryConversionPassIncGen

        LINK_LIBS PUBLIC
        cherryBasic
        MLIRArithDialect
        MLIRConvertToLLVMInterface
        MLIRLLVMCommonConversion
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        MLIRLinalgDialect
        MLIRLinalgTransforms
        MLIRLLVMDialect
        MLIRMathDialect
        MLIRMemRefDialect
        MLIRPtrDialect
        MLIRSCFDialect
)
