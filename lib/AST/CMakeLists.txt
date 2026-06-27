set(LIB_HEADERS ${MULBERRY_SOURCE_DIR}/include/mulberry/AST)

add_library(mulberryAST STATIC
        ${LIB_HEADERS}/AST.h
        ${LIB_HEADERS}/Node.h
        ${LIB_HEADERS}/Module.h
        ${LIB_HEADERS}/Decl.h
        ${LIB_HEADERS}/Expr.h
        ${LIB_HEADERS}/TensorExpr.h
        ${LIB_HEADERS}/Stat.h
        ASTDumper.cpp
)

target_include_directories(mulberryAST PUBLIC
        ${LIB_HEADERS}
)

target_link_libraries(mulberryAST PUBLIC
        mulberryBasic
)
