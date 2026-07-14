//===--- MulberryNNPackage.cpp - Mulberry NN package entry ----------------===//

#include "MulberryNN/MulberryNNDialect.h"
#include "MulberryNN/MulberryNNPasses.h"
#include "MulberryNN/MulberryNNToLinalgPatterns.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Compiler.h"

using namespace mlir;

namespace mulberry_nn = mlir::mulberry_nn;

extern "C" LLVM_ATTRIBUTE_WEAK DialectPluginLibraryInfo
mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "MulberryNNPackage", LLVM_VERSION_STRING,
          [](DialectRegistry* registry) {
            registry->insert<mulberry_nn::MulberryNNDialect>();
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "MulberryNNPackagePasses",
          LLVM_VERSION_STRING,
          []() { mulberry_nn::registerPasses(); }};
}
