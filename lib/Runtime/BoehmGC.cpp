//===--- BoehmGC.cpp ------------------------------------------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

#include <gc.h>

namespace {

struct TensorStorage {
  void* allocated;
  void* data;
  bool disposed;
};

struct ReplSlot {
  void* address = nullptr;
  uint64_t size = 0;
};

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

auto replSlots() -> std::map<uint64_t, ReplSlot>& {
  static std::map<uint64_t, ReplSlot> slots;
  return slots;
}

auto disposeTensorStorage(TensorStorage* storage) -> void {
  if (!storage || storage->disposed)
    return;

  std::free(storage->allocated);
  storage->allocated = nullptr;
  storage->data = nullptr;
  storage->disposed = true;
}

void finalizeTensorStorage(void* object, void*) {
  disposeTensorStorage(static_cast<TensorStorage*>(object));
}

} // namespace

extern "C" void mulberry_runtime_init() {
  initializeRuntime();
}

extern "C" void* mulberry_boehm_malloc(uint64_t size) {
  initializeRuntime();
  return GC_malloc(size);
}

extern "C" void* mulberry_repl_get_slot(uint64_t slot, uint64_t size) {
  initializeRuntime();
  auto &slots = replSlots();
  auto found = slots.find(slot);
  if (found != slots.end()) {
    if (found->second.size != size) {
      std::fputs("fatal: REPL slot type changed", stderr);
      std::abort();
    }
    return found->second.address;
  }

  auto *address = GC_malloc(size);
  if (!address) {
    std::fputs("fatal: failed to allocate REPL slot", stderr);
    std::abort();
  }
  std::memset(address, 0, size);
  slots.emplace(slot, ReplSlot{address, size});
  return address;
}

extern "C" void mulberry_boehm_free(void* pointer) {
  if (pointer)
    GC_free(pointer);
}

extern "C" void mulberry_tensor_storage_register(void* storage) {
  initializeRuntime();
  GC_register_finalizer_no_order(storage, finalizeTensorStorage, nullptr,
                                 nullptr, nullptr);
}

extern "C" void mulberry_tensor_storage_dispose(void* storage) {
  disposeTensorStorage(static_cast<TensorStorage*>(storage));
}

extern "C" void mulberry_tensor_use_after_dispose() {
  std::fputs("fatal: access to disposed Tensor\n", stderr);
  std::abort();
}
