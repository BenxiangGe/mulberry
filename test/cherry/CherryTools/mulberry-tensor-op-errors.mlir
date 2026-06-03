// RUN: not cherry-opt %s 2>&1 | FileCheck %s

module {
  func.func @bad_rank(%i: index) -> f32 {
    %tensor = mulberry.tensor.alloc : !mulberry.tensor<2x3xf32>
    %loaded = mulberry.tensor.load %tensor[%i] : !mulberry.tensor<2x3xf32> -> f32
    return %loaded : f32
  }
}

// CHECK: error: 'mulberry.tensor.load' op index count must match tensor rank
