set(LIB_HEADERS ${CHERRY_SOURCE_DIR}/include/cherry/Driver)

add_library(cherryDriver STATIC
        ${LIB_HEADERS}/Compilation.h
        Compilation.cpp
)

target_compile_definitions(cherryDriver PRIVATE
        CHERRY_MLIR_C_RUNNER_UTILS="${LLVM_BUILD_LIBRARY_DIR}/libmlir_c_runner_utils.so"
        CHERRY_RUNTIME_LIB="$<TARGET_FILE:cherryRuntime>"
)

target_link_libraries(cherryDriver
        MLIRBuiltinToLLVMIRTranslation
        MLIRExecutionEngine
        MLIRExecutionEngineUtils
        MLIRLLVMIRTransforms
        MLIRLLVMToLLVMIRTranslation
        MLIRRegisterAllExtensions
        MLIRTargetLLVMIRExport
        ${targets}
)

target_include_directories(cherryDriver PUBLIC
        ${LIB_HEADERS}
)
