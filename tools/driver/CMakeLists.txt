add_llvm_executable(cherry-driver driver.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

target_link_libraries(cherry-driver
        cherryBasic
        cherryDriver
        cherryParse
        cherryLLVMGen
        cherryMLIRGen
        cherrySema
        cherryAST
        ${dialect_libs}
        ${conversion_libs}
        ${extension_libs}
        LLVMSupport
)
