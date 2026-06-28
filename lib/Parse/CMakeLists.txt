set(LIB_HEADERS ${MULBERRY_SOURCE_DIR}/include/mulberry/Parse)

add_library(mulberryParse STATIC
        ${LIB_HEADERS}/TokenKinds.def
        ${LIB_HEADERS}/Token.h
        Token.cpp
        ${LIB_HEADERS}/Lexer.h
        Lexer.cpp
        ${LIB_HEADERS}/Parser.h
        Parser.cpp
        ${LIB_HEADERS}/DiagnosticsParse.h
        ${LIB_HEADERS}/DiagnosticsParse.def
)

target_include_directories(mulberryParse PUBLIC
        ${LIB_HEADERS}
)
