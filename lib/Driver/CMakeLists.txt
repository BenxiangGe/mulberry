set(LIB_HEADERS ${CHERRY_SOURCE_DIR}/include/cherry/Driver)
set(CHERRY_BUNDLED_PACKAGE_REGISTRY
        "mulberry.nn|${LLVM_LIBRARY_OUTPUT_INTDIR}/MulberryNNPackage${CMAKE_SHARED_MODULE_SUFFIX}|prepare-mulberry-nn-calls|prepare-mulberry-nn-calls,lower-mulberry-nn|1"
)
set(CHERRY_MLIR_RUNTIME_LIBS
        "$<TARGET_FILE:mlir_c_runner_utils>|$<TARGET_FILE:mlir_apfloat_wrappers>|$<TARGET_FILE:mlir_float16_utils>"
)

add_library(cherryDriver STATIC
        ${LIB_HEADERS}/Compilation.h
        Compilation.cpp
)

target_compile_definitions(cherryDriver PRIVATE
        CHERRY_STDLIB_DIR="${CHERRY_SOURCE_DIR}/stdlib"
        CHERRY_MLIR_C_RUNNER_UTILS="$<TARGET_FILE:mlir_c_runner_utils>"
        CHERRY_MLIR_RUNTIME_LIBS="${CHERRY_MLIR_RUNTIME_LIBS}"
        CHERRY_RUNTIME_LIB="$<TARGET_FILE:cherryRuntime>"
        CHERRY_BDWGC_LINK_LIB="$<TARGET_LINKER_FILE:gc>"
        CHERRY_BDWGC_RUNTIME_LIBS="$<TARGET_SONAME_FILE:gc>"
        CHERRY_BDWGC_LIBRARY_DIR="$<TARGET_FILE_DIR:gc>"
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
