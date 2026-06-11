add_library(cherryRuntime SHARED
        Safetensors.cpp
)

set_target_properties(cherryRuntime PROPERTIES
        OUTPUT_NAME cherry_runtime
)
