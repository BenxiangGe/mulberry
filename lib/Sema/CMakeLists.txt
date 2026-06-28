set(LIB_HEADERS ${MULBERRY_SOURCE_DIR}/include/mulberry/Sema)

add_library(mulberrySema STATIC
        ${LIB_HEADERS}/Sema.h
        Sema.cpp
        Symbols.h
        ${LIB_HEADERS}/DiagnosticsSema.h
        ${LIB_HEADERS}/DiagnosticsSema.def
)

target_include_directories(mulberrySema PUBLIC
        ${LIB_HEADERS}
)

target_link_libraries(mulberrySema PUBLIC
        mulberryBasic
)
