set(LIB_HEADERS ${CHERRY_SOURCE_DIR}/include/cherry/Driver)
set(CHERRY_BUNDLED_PACKAGE_REGISTRY
        "mulberry.nn|${LLVM_LIBRARY_OUTPUT_INTDIR}/MulberryNNPackage${CMAKE_SHARED_MODULE_SUFFIX}|prepare-mulberry-nn-calls|prepare-mulberry-nn-calls,lower-mulberry-nn|1"
)

add_library(cherryDriver STATIC
        ${LIB_HEADERS}/Compilation.h
        Compilation.cpp
)

target_compile_definitions(cherryDriver PRIVATE
        CHERRY_STDLIB_DIR="${CHERRY_SOURCE_DIR}/stdlib"
        CHERRY_MLIR_C_RUNNER_UTILS="${LLVM_BUILD_LIBRARY_DIR}/libmlir_c_runner_utils.so"
        CHERRY_RUNTIME_LIB="$<TARGET_FILE:cherryRuntime>"
        CHERRY_BUNDLED_PACKAGE_REGISTRY="${CHERRY_BUNDLED_PACKAGE_REGISTRY}"
)

target_link_libraries(cherryDriver
        MLIRBuiltinToLLVMIRTranslation
        MLIRExecutionEngine
        MLIRExecutionEngineUtils
        MLIRLLVMIRTransforms
        MLIRPluginsLib
        MLIRLLVMToLLVMIRTranslation
        MLIRRegisterAllExtensions
        MLIRTargetLLVMIRExport
        ${targets}
)

target_include_directories(cherryDriver PUBLIC
        ${LIB_HEADERS}
)
