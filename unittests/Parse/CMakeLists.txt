add_mulberry_unittest(MulberryParseTests
        LexerTest.cpp
)

target_link_libraries(MulberryParseTests
        PRIVATE
        mulberryParse
        LLVMSupport
        LLVMDemangle
)
