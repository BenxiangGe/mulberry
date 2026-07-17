set(MULBERRY_RUNTIME_SOURCES
        BoehmGC.cpp
        Core.cpp
        File.cpp
        Print.cpp
)

add_library(mulberryRuntimeObjects OBJECT
        ${MULBERRY_RUNTIME_SOURCES}
)

add_library(mulberryRuntime STATIC
        $<TARGET_OBJECTS:mulberryRuntimeObjects>
)

set_target_properties(mulberryRuntime PROPERTIES
        OUTPUT_NAME mulberry_runtime
)

target_link_libraries(mulberryRuntimeObjects PRIVATE gc)
target_link_libraries(mulberryRuntime PUBLIC gc)
