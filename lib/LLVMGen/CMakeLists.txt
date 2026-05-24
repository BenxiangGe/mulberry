set(LIB_HEADERS ${CHERRY_SOURCE_DIR}/include/cherry/LLVMGen)

add_library(cherryLLVMGen STATIC
        ${LIB_HEADERS}/LLVMGen.h
        ${LIB_HEADERS}/TypeConverter.h
        LLVMGen.cpp
        TypeConverter.cpp
)

target_include_directories(cherryLLVMGen PUBLIC
        ${LIB_HEADERS}
)

target_link_libraries(cherryLLVMGen PUBLIC
        cherryBasic
)
