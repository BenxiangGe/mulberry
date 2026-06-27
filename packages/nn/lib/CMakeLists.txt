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

        LINK_LIBS
        MulberryNNIR
        MulberryNNConversion
)
