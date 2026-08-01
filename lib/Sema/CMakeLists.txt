set(LIB_HEADERS ${MULBERRY_SOURCE_DIR}/include/mulberry/Sema)

add_library(mulberrySema STATIC
        ${LIB_HEADERS}/Sema.h
        SemaData.h
        SemaImpl.h
        SemaSupport.h
        SemaSupport.cpp
        SemaComptime.h
        SemaComptime.cpp
        SemaBuiltins.cpp
        SemaDeclarations.h
        SemaDeclarations.cpp
        SemaExpressions.h
        SemaExpressions.cpp
        SemaGenerics.cpp
        SemaStatements.h
        SemaStatements.cpp
        SemaTypes.cpp
        SemaTraits.h
        SemaTraits.cpp
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
