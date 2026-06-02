set(LIB_HEADERS ${CHERRY_SOURCE_DIR}/include/cherry/Driver)

add_library(cherryDriver STATIC
        ${LIB_HEADERS}/Compilation.h
        Compilation.cpp
)

target_link_libraries(cherryDriver
        ${targets}
)

target_include_directories(cherryDriver PUBLIC
        ${LIB_HEADERS}
)
