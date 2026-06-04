// RUN: not cherry-opt --lower-mulberry %s 2>&1 | FileCheck %s

module {
  func.func @main() {
    %tensor = mulberry.tensor.alloc() : !mulberry.tensor<2xf32>
    return
  }
}

// CHECK: failed to legalize operation 'mulberry.tensor.alloc'
