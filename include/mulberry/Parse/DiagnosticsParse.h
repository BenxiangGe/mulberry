//===--- DiagnosticsParse.h - Diagnostic Definitions ------------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_DIAGNOSTICSPARSE_H
#define MULBERRY_DIAGNOSTICSPARSE_H

namespace mulberry {
namespace diag {
#define ERROR(ID, TEXT) const char *const ID = TEXT;
#include "DiagnosticsParse.def"
} // end namespace diag
} // end namespace mulberry

#endif // MULBERRY_DIAGNOSTICSPARSE_H
