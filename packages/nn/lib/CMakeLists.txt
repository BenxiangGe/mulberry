add_subdirectory(IR)
add_subdirectory(Conversion)

add_llvm_library(MulberryNNPackage
        MODULE BUILDTREE_ONLY
        MulberryNNPackage.cpp

        DEPENDS
        MLIRMulberryNNOpsIncGen
        MLIRMulberryNNPassesIncGen

        PLUGIN_TOOL
        mulberry-opt
)

# The package shares LLVM, MLIR, and Mulberry core code with its host. Linking
# the static NN libraries here would pull a second LLVM copy into this module.
target_sources(MulberryNNPackage PRIVATE
        $<TARGET_OBJECTS:obj.MulberryNNIR>
        $<TARGET_OBJECTS:obj.MulberryNNConversion>
)
