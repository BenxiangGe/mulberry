// RUN: mulberry-opt --load-dialect-plugin=%mulberry_libs/MulberryNNPackage%shlibext --load-pass-plugin=%mulberry_libs/MulberryNNPackage%shlibext --pass-pipeline='builtin.module(lower-mulberry-nn)' %s | FileCheck %s
// RUN: mulberry-opt --load-dialect-plugin=%mulberry_libs/MulberryNNPackage%shlibext --load-pass-plugin=%mulberry_libs/MulberryNNPackage%shlibext --pass-pipeline='builtin.module(lower-mulberry-nn,func.func(convert-linalg-to-loops),lower-affine,convert-scf-to-cf)' %s | FileCheck %s --check-prefix=LOOPS

module {
  func.func @static_backward(
      %reluInput: memref<4xf32>, %reluOutputGradient: memref<4xf32>,
      %reluInputGradient: memref<4xf32>,
      %input: memref<1x1x4x4xf32>, %weight: memref<1x1x2x2xf32>,
      %convOutputGradient: memref<1x1x2x2xf32>,
      %inputGradient: memref<1x1x4x4xf32>,
      %weightGradient: memref<1x1x2x2xf32>,
      %biasGradient: memref<1xf32>, %poolInput: memref<1x1x3x3xf32>,
      %poolOutputGradient: memref<1x1x3x3xf32>,
      %poolInputGradient: memref<1x1x3x3xf32>) {
    mulberry_nn.relu_backward
        ins(%reluInput, %reluOutputGradient : memref<4xf32>, memref<4xf32>)
        outs(%reluInputGradient : memref<4xf32>)
    mulberry_nn.conv2d_backward
        ins(%input, %weight, %convOutputGradient : memref<1x1x4x4xf32>,
            memref<1x1x2x2xf32>, memref<1x1x2x2xf32>)
        outs(%inputGradient, %weightGradient, %biasGradient
             : memref<1x1x4x4xf32>, memref<1x1x2x2xf32>, memref<1xf32>) {
          dilations = array<i64: 2, 1>,
          padding = array<i64: 1, 0, 1, 0>,
          strides = array<i64: 2, 2>
        }
    mulberry_nn.max_pool2d_backward
        ins(%poolInput, %poolOutputGradient : memref<1x1x3x3xf32>,
            memref<1x1x3x3xf32>)
        outs(%poolInputGradient : memref<1x1x3x3xf32>) {
          kernel = array<i64: 2, 2>,
          padding = array<i64: 1, 0, 1, 0>,
          strides = array<i64: 1, 1>
        }
    return
  }

  func.func @dynamic_backward(
      %reluInput: memref<?x?x?x?xf32>,
      %reluOutputGradient: memref<?x?x?x?xf32>,
      %reluInputGradient: memref<?x?x?x?xf32>,
      %input: memref<?x?x?x?xf32>, %weight: memref<?x?x?x?xf32>,
      %convOutputGradient: memref<?x?x?x?xf32>,
      %inputGradient: memref<?x?x?x?xf32>,
      %weightGradient: memref<?x?x?x?xf32>,
      %biasGradient: memref<?xf32>, %poolInput: memref<?x?x?x?xf32>,
      %poolOutputGradient: memref<?x?x?x?xf32>,
      %poolInputGradient: memref<?x?x?x?xf32>) {
    mulberry_nn.relu_backward
        ins(%reluInput, %reluOutputGradient : memref<?x?x?x?xf32>,
            memref<?x?x?x?xf32>)
        outs(%reluInputGradient : memref<?x?x?x?xf32>)
    mulberry_nn.conv2d_backward
        ins(%input, %weight, %convOutputGradient : memref<?x?x?x?xf32>,
            memref<?x?x?x?xf32>, memref<?x?x?x?xf32>)
        outs(%inputGradient, %weightGradient, %biasGradient
             : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?xf32>) {
          dilations = array<i64: 1, 1>,
          padding = array<i64: 0, 0, 0, 0>,
          strides = array<i64: 1, 1>
        }
    mulberry_nn.max_pool2d_backward
        ins(%poolInput, %poolOutputGradient : memref<?x?x?x?xf32>,
            memref<?x?x?x?xf32>)
        outs(%poolInputGradient : memref<?x?x?x?xf32>) {
          kernel = array<i64: 2, 2>,
          padding = array<i64: 0, 0, 0, 0>,
          strides = array<i64: 2, 2>
        }
    return
  }
}

// CHECK-LABEL: func.func @static_backward
// CHECK: linalg.map
// CHECK: arith.cmpf ogt
// CHECK-COUNT-3: linalg.fill
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.store
// CHECK: scf.for
// CHECK: scf.if
// CHECK: arith.mulf
// CHECK: memref.store
// CHECK: linalg.fill
// CHECK: scf.for
// CHECK: scf.if
// CHECK: arith.cmpf ogt
// CHECK: scf.if

// CHECK-LABEL: func.func @dynamic_backward
// CHECK: linalg.map
// CHECK: memref.dim
// CHECK: scf.for
// CHECK: scf.if
// CHECK-NOT: mulberry_nn.

// LOOPS-NOT: mulberry_nn.
// LOOPS-NOT: linalg.
// LOOPS-NOT: scf.
// LOOPS: cf.cond_br
