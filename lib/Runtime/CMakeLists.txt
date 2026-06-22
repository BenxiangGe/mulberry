add_library(cherryRuntime SHARED
        BoehmGC.cpp
        File.cpp
        Safetensors.cpp
)

set_target_properties(cherryRuntime PROPERTIES
        OUTPUT_NAME cherry_runtime
)

target_link_libraries(cherryRuntime PRIVATE gc)
