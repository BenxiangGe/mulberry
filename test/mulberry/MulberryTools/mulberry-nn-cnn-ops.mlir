// RUN: mulberry-opt --load-dialect-plugin=%mulberry_libs/MulberryNNPackage%shlibext %s | FileCheck %s

!image = !mulberry_core.tensor<10x1x28x28xf32>
!weight = !mulberry_core.tensor<20x1x5x5xf32>
!bias = !mulberry_core.tensor<20xf32>
!conv = !mulberry_core.tensor<10x20x24x24xf32>
!pool = !mulberry_core.tensor<10x20x12x12xf32>
!scores = !mulberry_core.tensor<10x10xf32>

module {
  func.func @core_tensor_cnn(
      %image: !image, %weight: !weight, %bias: !bias,
      %conv: !conv, %pool: !pool, %relu: !pool,
      %scores: !scores, %probabilities: !scores) {
    mulberry_nn.conv2d
        ins(%image, %weight, %bias : !image, !weight, !bias)
        outs(%conv : !conv) {
          dilations = array<i64: 1, 1>,
          padding = array<i64: 0, 0, 0, 0>,
          strides = array<i64: 1, 1>
        }
    mulberry_nn.max_pool2d ins(%conv : !conv) outs(%pool : !pool) {
      kernel = array<i64: 2, 2>,
      padding = array<i64: 0, 0, 0, 0>,
      strides = array<i64: 2, 2>
    }
    mulberry_nn.relu ins(%pool : !pool) outs(%relu : !pool)
    mulberry_nn.softmax
        ins(%scores : !scores) outs(%probabilities : !scores)
    return
  }

  func.func @memref_cnn(
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

  func.func @core_cross_entropy(%logits: !scores,
                                %expected: !scores) -> f32 {
    %loss = mulberry_nn.softmax_cross_entropy
        ins(%logits, %expected : !scores, !scores) -> f32
    return %loss : f32
  }

  func.func @memref_cross_entropy(%logits: memref<10x10xf32>,
                                  %expected: memref<10x10xf32>) -> f32 {
    %loss = mulberry_nn.softmax_cross_entropy
        ins(%logits, %expected : memref<10x10xf32>, memref<10x10xf32>) -> f32
    return %loss : f32
  }
}

// CHECK-LABEL: func.func @core_tensor_cnn
// CHECK: mulberry_nn.conv2d
// CHECK: dilations = array<i64: 1, 1>
// CHECK: padding = array<i64: 0, 0, 0, 0>
// CHECK: strides = array<i64: 1, 1>
// CHECK: mulberry_nn.max_pool2d
// CHECK: kernel = array<i64: 2, 2>
// CHECK: strides = array<i64: 2, 2>
// CHECK: mulberry_nn.relu
// CHECK: mulberry_nn.softmax

// CHECK-LABEL: func.func @memref_cnn
// CHECK: mulberry_nn.conv2d
// CHECK: mulberry_nn.max_pool2d
// CHECK: mulberry_nn.relu
// CHECK: mulberry_nn.softmax

// CHECK-LABEL: func.func @core_cross_entropy
// CHECK: mulberry_nn.softmax_cross_entropy

// CHECK-LABEL: func.func @memref_cross_entropy
// CHECK: mulberry_nn.softmax_cross_entropy
