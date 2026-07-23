add_custom_target(MulberryUnitTests)
set_target_properties(MulberryUnitTests PROPERTIES FOLDER "Mulberry Tests")

function(add_mulberry_unittest test_dirname)
    add_unittest(MulberryUnitTests ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(Parse)
add_subdirectory(BigInt)
