// RUN: mulberry-opt --load-dialect-plugin=%mulberry_libs/MulberryNNPackage%shlibext --load-pass-plugin=%mulberry_libs/MulberryNNPackage%shlibext --pass-pipeline='builtin.module(lower-mulberry-nn,buffer-deallocation-pipeline)' %s | FileCheck %s
// RUN: mulberry-opt --load-dialect-plugin=%mulberry_libs/MulberryNNPackage%shlibext --load-pass-plugin=%mulberry_libs/MulberryNNPackage%shlibext --pass-pipeline='builtin.module(lower-mulberry-nn,func.func(convert-linalg-to-loops),convert-scf-to-cf)' %s | FileCheck %s --check-prefix=LOOPS

module {
  func.func @nielsen_block(
      %image: memref<10x1x28x28xf32>,
      %weight: memref<20x1x5x5xf32>, %bias: memref<20xf32>,
      %conv: memref<10x20x24x24xf32>,
      %pool: memref<10x20x12x12xf32>,
      %relu: memref<10x20x12x12xf32>,
      %scores: memref<10x10xf32>,
      %probabilities: memref<10x10xf32>) {
    mulberry_nn.conv2d
        ins(%image, %weight, %bias : memref<10x1x28x28xf32>,
            memref<20x1x5x5xf32>, memref<20xf32>)
        outs(%conv : memref<10x20x24x24xf32>) {
          dilations = array<i64: 1, 1>,
          padding = array<i64: 0, 0, 0, 0>,
          strides = array<i64: 1, 1>
        }
    mulberry_nn.max_pool2d
        ins(%conv : memref<10x20x24x24xf32>)
        outs(%pool : memref<10x20x12x12xf32>) {
          kernel = array<i64: 2, 2>,
          padding = array<i64: 0, 0, 0, 0>,
          strides = array<i64: 2, 2>
        }
    mulberry_nn.relu
        ins(%pool : memref<10x20x12x12xf32>)
        outs(%relu : memref<10x20x12x12xf32>)
    mulberry_nn.softmax
        ins(%scores : memref<10x10xf32>)
        outs(%probabilities : memref<10x10xf32>)
    return
  }

  func.func @padded_windows(
      %image: memref<1x1x3x3xf32>, %weight: memref<1x1x3x3xf32>,
      %bias: memref<1xf32>, %conv: memref<1x1x3x3xf32>,
      %pool: memref<1x1x4x4xf32>) {
    mulberry_nn.conv2d
        ins(%image, %weight, %bias : memref<1x1x3x3xf32>,
            memref<1x1x3x3xf32>, memref<1xf32>)
        outs(%conv : memref<1x1x3x3xf32>) {
          dilations = array<i64: 1, 1>,
          padding = array<i64: 1, 1, 1, 1>,
          strides = array<i64: 1, 1>
        }
    mulberry_nn.max_pool2d
        ins(%conv : memref<1x1x3x3xf32>)
        outs(%pool : memref<1x1x4x4xf32>) {
          kernel = array<i64: 2, 2>,
          padding = array<i64: 1, 1, 1, 1>,
          strides = array<i64: 1, 1>
    }
    return
  }

  func.func @dynamic_shapes(
      %image: memref<?x1x?x?xf32>, %weight: memref<20x1x5x5xf32>,
      %bias: memref<20xf32>, %conv: memref<?x20x?x?xf32>,
      %scores: memref<?x?xf32>, %probabilities: memref<?x?xf32>) {
    mulberry_nn.conv2d
        ins(%image, %weight, %bias : memref<?x1x?x?xf32>,
            memref<20x1x5x5xf32>, memref<20xf32>)
        outs(%conv : memref<?x20x?x?xf32>) {
          dilations = array<i64: 1, 1>,
          padding = array<i64: 1, 1, 1, 1>,
          strides = array<i64: 1, 1>
        }
    mulberry_nn.softmax
        ins(%scores : memref<?x?xf32>)
        outs(%probabilities : memref<?x?xf32>)
    return
  }

  func.func @cross_entropy(%logits: memref<?x?xf32>,
                           %expected: memref<?x?xf32>) -> f32 {
    %loss = mulberry_nn.softmax_cross_entropy
        ins(%logits, %expected : memref<?x?xf32>, memref<?x?xf32>) -> f32
    return %loss : f32
  }
}

// CHECK-LABEL: func.func @nielsen_block
// CHECK: arith.constant 0xFF800000 : f32
// CHECK: linalg.generic
// CHECK: linalg.conv_2d_nchw_fchw
// CHECK-SAME: dilations = dense<1> : tensor<2xi64>
// CHECK-SAME: strides = dense<1> : tensor<2xi64>
// CHECK: memref.alloca() : memref<2x2xf32>
// CHECK: linalg.pooling_nchw_max
// CHECK-SAME: dilations = dense<1> : tensor<2xi64>
// CHECK-SAME: strides = dense<2> : tensor<2xi64>
// CHECK: linalg.map
// CHECK: arith.maximumf
// CHECK: memref.alloc() : memref<10xf32>
// CHECK: memref.alloc() : memref<10xf32>
// CHECK: arith.maximumf
// CHECK: arith.subf
// CHECK: math.exp
// CHECK: arith.addf
// CHECK: arith.divf
// CHECK-COUNT-2: memref.dealloc

// CHECK-LABEL: func.func @padded_windows
// CHECK: arith.constant 0xFF800000 : f32
// CHECK: %[[CONV_PAD:.*]] = memref.alloc() : memref<1x1x5x5xf32>
// CHECK: linalg.fill
// CHECK: %[[CONV_INTERIOR:.*]] = memref.subview %[[CONV_PAD]][0, 0, 1, 1]
// CHECK: memref.copy %{{.*}}, %[[CONV_INTERIOR]]
// CHECK: linalg.conv_2d_nchw_fchw
// CHECK: %[[POOL_PAD:.*]] = memref.alloc() : memref<1x1x5x5xf32>
// CHECK: %[[POOL_INTERIOR:.*]] = memref.subview %[[POOL_PAD]][0, 0, 1, 1]
// CHECK: memref.copy %{{.*}}, %[[POOL_INTERIOR]]
// CHECK: linalg.pooling_nchw_max
// CHECK: memref.dealloc %[[CONV_PAD]]
// CHECK: memref.dealloc %[[POOL_PAD]]

// CHECK-LABEL: func.func @dynamic_shapes
// CHECK: memref.dim
// CHECK: arith.addi
// CHECK: arith.addi
// CHECK: memref.alloc(%{{.*}}, %{{.*}}, %{{.*}}) : memref<?x1x?x?xf32>
// CHECK: memref.subview
// CHECK: linalg.conv_2d_nchw_fchw
// CHECK: memref.dim
// CHECK: memref.alloc(%{{.*}}) : memref<?xf32>
// CHECK: memref.alloc(%{{.*}}) : memref<?xf32>
// CHECK: math.exp
// CHECK: arith.divf
// CHECK-NOT: mulberry_nn.

// CHECK-LABEL: func.func @cross_entropy
// CHECK: %[[ROWS:.*]] = memref.dim %{{.*}}, %{{.*}} : memref<?x?xf32>
// CHECK: scf.for
// CHECK: arith.maximumf
// CHECK: math.exp
// CHECK: math.log
// CHECK: arith.divf
// CHECK-NOT: memref.alloc
// CHECK-NOT: mulberry_nn.

// LOOPS-NOT: mulberry_nn.
// LOOPS-NOT: linalg.
// LOOPS: cf.cond_br
