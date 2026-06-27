add_library(mulberryRuntime SHARED
        BoehmGC.cpp
        Core.cpp
        File.cpp
        Print.cpp
)

set_target_properties(mulberryRuntime PROPERTIES
        OUTPUT_NAME mulberry_runtime
)

target_link_libraries(mulberryRuntime PRIVATE gc)
