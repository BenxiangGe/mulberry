include(cmake/CPM.cmake)

CPMAddPackage(
        NAME bdwgc
        GITHUB_REPOSITORY ivmai/bdwgc
        GIT_TAG v8.2.10
        OPTIONS
          "enable_docs OFF"
          "enable_gcj_support OFF"
          "enable_tests OFF"
)
