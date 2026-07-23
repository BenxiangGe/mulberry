add_mulberry_unittest(MulberryBigIntTests
        BigIntRuntimeTest.cpp
)

target_link_libraries(MulberryBigIntTests
        PRIVATE
        mulberryRuntime
)
