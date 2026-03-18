#ifndef CHERRY_LOGGING_H
#define CHERRY_LOGGING_H

#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

// #ifndef DEBUG_TYPE
// #define DEBUG_TYPE "cherry"
// #endif // DEBUG_TYPE

#define DBG(...) \
    LLVM_DEBUG( \
        llvm::dbgs() << "[" << __FILE__ << ":" << __LINE__ << " | " << __func__ << "]" \
                     __VA_OPT__(<< " " << llvm::formatv(__VA_ARGS__)) << "\n" \
    )

#define ERR(...) \
    LLVM_DEBUG( \
        llvm::errs() << "[" << __FILE__ << ":" << __LINE__ << " | " << __func__ << "]" \
                     __VA_OPT__(<< " " << llvm::formatv(__VA_ARGS__)) << "\n" \
    )

#endif // CHERRY_LOGGING_H