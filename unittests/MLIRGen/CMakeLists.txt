add_cherry_unittest(CherryMLIRGenTests
        TypeConverterTest.cpp
)

target_link_libraries(CherryMLIRGenTests
        PRIVATE
        cherryMLIRGen
        MLIRIR
        LLVMSupport
        LLVMDemangle
)
