// RUN: not cherry-opt %s 2>&1 | FileCheck %s

module {
  func.func @bad_alloc(%n: index) {
    %tensor = mulberry.tensor.alloc(%n) : !mulberry.tensor<?x?xf32>
    return
  }
}

// CHECK: error: 'mulberry.tensor.alloc' op dynamic size count must match dynamic tensor dims
