//===--- BoehmGC.cpp ------------------------------------------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include <cstdint>

#include <gc.h>

namespace {

auto initializeRuntime() -> void {
  static bool initialized = false;
  if (initialized)
    return;

  // Boehm recommends the GC_INIT() macro for explicit process initialization;
  // it carries platform-specific stack discovery details that GC_init() may
  // not have when called late from JIT-generated code.
  GC_INIT();
  initialized = true;
}

} // namespace

extern "C" void mulberry_runtime_init() {
  initializeRuntime();
}

extern "C" void* mulberry_boehm_malloc(uint64_t size) {
  initializeRuntime();
  return GC_malloc(size);
}
