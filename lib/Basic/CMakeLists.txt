set(LIB_HEADERS ${MULBERRY_SOURCE_DIR}/include/mulberry/Basic)

add_library(mulberryBasic STATIC
        ${LIB_HEADERS}/Builtins.h
        ${LIB_HEADERS}/MulberryResult.h
        ${LIB_HEADERS}/Types.h
        Types.cpp
)

target_include_directories(mulberryBasic PUBLIC
        ${LIB_HEADERS}
)
