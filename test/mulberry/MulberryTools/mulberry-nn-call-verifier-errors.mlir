// RUN: not mulberry-opt --load-dialect-plugin=%mulberry_libs/MulberryNNPackage%shlibext --load-pass-plugin=%mulberry_libs/MulberryNNPackage%shlibext --pass-pipeline='builtin.module(prepare-mulberry-nn-calls)' %s 2>&1 | FileCheck %s

module {
  func.func private @mulberry.nn.__tensor.softmaxCrossEntropy(
      memref<4xf32>, memref<4xf32>) -> f32

  func.func @bad_internal_rank(
      %logits: memref<4xf32>, %expected: memref<4xf32>) -> f32 {
    %result = call @mulberry.nn.__tensor.softmaxCrossEntropy(%logits, %expected)
        : (memref<4xf32>, memref<4xf32>) -> f32
    return %result : f32
  }
}

// CHECK: error: 'mulberry_nn.softmax_cross_entropy' op logits must have rank 2
