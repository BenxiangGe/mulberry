set(LIB_HEADERS ${MULBERRY_SOURCE_DIR}/include/mulberry/Driver)
set(MULBERRY_BUNDLED_PACKAGE_REGISTRY
        "mulberry.nn|${LLVM_LIBRARY_OUTPUT_INTDIR}/MulberryNNPackage${CMAKE_SHARED_MODULE_SUFFIX}|prepare-mulberry-nn-calls|prepare-mulberry-nn-calls,lower-mulberry-nn|1"
)
set(MULBERRY_MLIR_RUNTIME_LIBS
        "$<TARGET_FILE:mlir_c_runner_utils>|$<TARGET_FILE:mlir_apfloat_wrappers>|$<TARGET_FILE:mlir_float16_utils>"
)

add_library(mulberryDriver STATIC
        ${LIB_HEADERS}/Compilation.h
        Compilation.cpp
)

target_compile_definitions(mulberryDriver PRIVATE
        MULBERRY_STDLIB_DIR="${MULBERRY_SOURCE_DIR}/stdlib"
        MULBERRY_MLIR_C_RUNNER_UTILS="$<TARGET_FILE:mlir_c_runner_utils>"
        MULBERRY_MLIR_RUNTIME_LIBS="${MULBERRY_MLIR_RUNTIME_LIBS}"
        MULBERRY_RUNTIME_LIB="$<TARGET_FILE:mulberryRuntime>"
        MULBERRY_BDWGC_LINK_LIB="$<TARGET_LINKER_FILE:gc>"
        MULBERRY_BDWGC_RUNTIME_LIBS="$<TARGET_SONAME_FILE:gc>"
        MULBERRY_BDWGC_LIBRARY_DIR="$<TARGET_FILE_DIR:gc>"
        MULBERRY_BUNDLED_PACKAGE_REGISTRY="${MULBERRY_BUNDLED_PACKAGE_REGISTRY}"
)

target_link_libraries(mulberryDriver
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

target_include_directories(mulberryDriver PUBLIC
        ${LIB_HEADERS}
)
