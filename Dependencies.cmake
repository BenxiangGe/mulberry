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

set(FETCHCONTENT_UPDATES_DISCONNECTED_ISOCLINE ON CACHE BOOL
        "Do not update the pinned Isocline checkout" FORCE)

CPMAddPackage(
        NAME isocline
        GITHUB_REPOSITORY daanx/isocline
        GIT_TAG 8d6dc1ef95b1b46711e66eb23d39d4467a0fcdac
        PATCH_COMMAND
          ${CMAKE_COMMAND}
          -DPATCH_FILE=${CMAKE_CURRENT_SOURCE_DIR}/cmake/isocline-history-search.patch
          -DSOURCE_DIR=<SOURCE_DIR>
          -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/apply-isocline-patch.cmake
        OPTIONS
          "IC_BUILD_TESTS OFF"
)
