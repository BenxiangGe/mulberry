//===--- BoehmGC.cpp ------------------------------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include <cstdint>

#include <gc.h>

extern "C" void* mulberry_boehm_malloc(uint64_t size) {
  static bool initialized = false;
  if (!initialized) {
    GC_init();
    initialized = true;
  }

  return GC_malloc(size);
}
