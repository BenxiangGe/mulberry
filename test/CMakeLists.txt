llvm_canonicalize_cmake_booleans(
        MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/mulberry/Unit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/mulberry/Unit/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/mulberry/Unit/lit.cfg.py
)

set(MULBERRY_TEST_DEPENDS
        FileCheck count not
        MulberryUnitTests
        mulberry-capi-test
        mulberry-opt
        mulberry-translate
        mulberry-driver
        mulberryRuntime
        MulberryNNPackage
)

if(MLIR_ENABLE_BINDINGS_PYTHON)
    list(APPEND MULBERRY_TEST_DEPENDS MulberryPythonModules)
endif()

add_lit_testsuite(check-mulberry "Running the Mulberry regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${MULBERRY_TEST_DEPENDS}
        )
set_target_properties(check-mulberry PROPERTIES FOLDER "Tests")

add_lit_testsuites(mulberry ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${MULBERRY_TEST_DEPENDS})

add_subdirectory(CAPI)
