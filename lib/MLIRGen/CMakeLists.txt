set(LIB_HEADERS ${MULBERRY_SOURCE_DIR}/include/mulberry/MLIRGen)

add_mlir_dialect_library(mulberryMLIRGen
        ${LIB_HEADERS}/MLIRGen.h
        MLIRGen.cpp
        ${LIB_HEADERS}/TypeConverter.h
        TypeConverter.cpp

        # Conversion
        ${LIB_HEADERS}/Conversion/MulberryPasses.h
        Conversion/ConvertMulberryToLLVM.cpp
        Conversion/LowerMulberry.cpp

        # IR
        ${LIB_HEADERS}/IR/MulberryDialect.h
        IR/MulberryDialect.cpp
        ${LIB_HEADERS}/IR/MulberryOps.h
        IR/MulberryOps.cpp
        ${LIB_HEADERS}/IR/MulberryTypes.h
        IR/MulberryTypes.cpp

        DEPENDS
        MLIRMulberryOpsIncGen
        MLIRMulberryConversionPassIncGen

        LINK_LIBS PUBLIC
        mulberryBasic
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
