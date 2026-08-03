add_llvm_executable(imb imb.cpp)

# Keep the REPL executable self-contained so generated extern calls resolve
# from the current process, just like the batch driver.
target_sources(imb PRIVATE
        $<TARGET_OBJECTS:mulberryRuntimeObjects>
)
set_target_properties(imb PROPERTIES ENABLE_EXPORTS ON)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

target_link_libraries(imb PUBLIC
        mulberryBasic
        mulberryDriver
        mulberryParse
        mulberryMLIRGen
        mulberrySema
        mulberryAST
        mulberryRuntime
        isocline
        ${dialect_libs}
        ${conversion_libs}
        ${extension_libs}
        LLVMSupport
)
