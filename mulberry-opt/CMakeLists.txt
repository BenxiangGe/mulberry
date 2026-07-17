get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
        MLIROptLib
        mulberryMLIRGen
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRAnalysis
        MLIRRegisterAllDialects
        MLIRRegisterAllExtensions
        MLIRRegisterAllPasses
        MLIRSupport
        MLIRTransformDialect
        MLIRTransformDialectTransforms
        MLIRTransforms
        ${dialect_libs}
        ${conversion_libs}
        )

add_llvm_executable(mulberry-opt mulberry-opt.cpp)
set_target_properties(mulberry-opt PROPERTIES ENABLE_EXPORTS ON)

llvm_update_compile_flags(mulberry-opt)
target_link_libraries(mulberry-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(mulberry-opt)
