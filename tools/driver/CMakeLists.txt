add_llvm_executable(mulberry-driver driver.cpp)

# ORC resolves generated extern calls from the current process. Embed every
# runtime object and export executable symbols so no runtime .so is needed.
target_sources(mulberry-driver PRIVATE
        $<TARGET_OBJECTS:mulberryRuntimeObjects>
)
set_target_properties(mulberry-driver PROPERTIES ENABLE_EXPORTS ON)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

target_link_libraries(mulberry-driver PUBLIC
        mulberryBasic
        mulberryDriver
        mulberryParse
        mulberryMLIRGen
        mulberrySema
        mulberryAST
        mulberryRuntime
        ${dialect_libs}
        ${conversion_libs}
        ${extension_libs}
        LLVMSupport
)
