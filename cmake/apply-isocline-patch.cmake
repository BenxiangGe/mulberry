find_program(PATCH_EXECUTABLE patch REQUIRED)

execute_process(
        COMMAND ${PATCH_EXECUTABLE} --dry-run -p1
        INPUT_FILE "${PATCH_FILE}"
        WORKING_DIRECTORY "${SOURCE_DIR}"
        RESULT_VARIABLE forwardResult
        OUTPUT_QUIET
        ERROR_QUIET)

if(forwardResult EQUAL 0)
  execute_process(
          COMMAND ${PATCH_EXECUTABLE} -p1
          INPUT_FILE "${PATCH_FILE}"
          WORKING_DIRECTORY "${SOURCE_DIR}"
          RESULT_VARIABLE applyResult)
  if(NOT applyResult EQUAL 0)
    message(FATAL_ERROR "Failed to apply Isocline patch")
  endif()
  return()
endif()

# A second configure can revisit the patch step after the source is already patched.
# Accept that state only when the complete patch reverses cleanly; partial changes fail.
execute_process(
        COMMAND ${PATCH_EXECUTABLE} --dry-run -R -p1
        INPUT_FILE "${PATCH_FILE}"
        WORKING_DIRECTORY "${SOURCE_DIR}"
        RESULT_VARIABLE reverseResult
        OUTPUT_QUIET
        ERROR_QUIET)
if(NOT reverseResult EQUAL 0)
  message(FATAL_ERROR "Isocline source is neither unpatched nor fully patched")
endif()
