include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `mlir_mulberry`
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_mulberry.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(MulberryPythonSources)

################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(MulberryPythonCAPI
  INSTALL_COMPONENT MulberryPythonModules
  INSTALL_DESTINATION python_packages/mulberry/mlir_mulberry/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/mulberry/mlir_mulberry/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    MulberryPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(MulberryPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/mulberry/mlir_mulberry"
  INSTALL_PREFIX "python_packages/mulberry/mlir_mulberry"
  DECLARED_SOURCES
    MulberryPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    MulberryPythonCAPI
  )
