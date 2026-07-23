set(BUILD_SHARED_LIBS OFF)

set(LIB_HEADERS ${MULBERRY_SOURCE_DIR}/include/mulberry/BigInt)

add_mlir_dialect_library(mulberryBigInt
        ${LIB_HEADERS}/BigIntDialect.h
        ${LIB_HEADERS}/BigIntOps.h
        ${LIB_HEADERS}/BigIntTypes.h
        BigIntDialect.cpp
        BigIntOps.cpp

        DEPENDS
        MLIRBigIntOpsIncGen

        LINK_LIBS PUBLIC
        mulberryBasic
        MLIRInferTypeOpInterface
        MLIRIR
)
