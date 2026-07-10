// RUN: mulberry-opt --load-dialect-plugin=%mulberry_libs/MulberryNNPackage%shlibext %s | FileCheck %s

!image = !mulberry_core.tensor<10x1x28x28xf32>
!weight = !mulberry_core.tensor<20x1x5x5xf32>
!bias = !mulberry_core.tensor<20xf32>
!conv = !mulberry_core.tensor<10x20x24x24xf32>
!pool = !mulberry_core.tensor<10x20x12x12xf32>

module {
  func.func @core_tensor_backward(
      %image: !image, %weight: !weight, %convGradient: !conv,
      %imageGradient: !image, %weightGradient: !weight,
      %biasGradient: !bias, %poolInput: !conv, %poolGradient: !pool,
      %poolInputGradient: !conv, %reluInput: !pool,
      %reluOutputGradient: !pool, %reluInputGradient: !pool) {
    mulberry_nn.relu_backward
        ins(%reluInput, %reluOutputGradient : !pool, !pool)
        outs(%reluInputGradient : !pool)
    mulberry_nn.max_pool2d_backward
        ins(%poolInput, %poolGradient : !conv, !pool)
        outs(%poolInputGradient : !conv) {
          kernel = array<i64: 2, 2>,
          padding = array<i64: 0, 0, 0, 0>,
          strides = array<i64: 2, 2>
        }
    mulberry_nn.conv2d_backward
        ins(%image, %weight, %convGradient : !image, !weight, !conv)
        outs(%imageGradient, %weightGradient, %biasGradient
             : !image, !weight, !bias) {
          dilations = array<i64: 1, 1>,
          padding = array<i64: 0, 0, 0, 0>,
          strides = array<i64: 1, 1>
        }
    return
  }

  func.func @dynamic_memref_backward(
      %image: memref<?x1x?x?xf32>,
      %weight: memref<20x1x5x5xf32>,
      %convGradient: memref<?x20x?x?xf32>,
      %imageGradient: memref<?x1x?x?xf32>,
      %weightGradient: memref<20x1x5x5xf32>,
      %biasGradient: memref<20xf32>,
      %poolInput: memref<?x20x?x?xf32>,
      %poolGradient: memref<?x20x?x?xf32>,
      %poolInputGradient: memref<?x20x?x?xf32>,
      %reluInput: memref<?x20x?x?xf32>,
      %reluOutputGradient: memref<?x20x?x?xf32>,
      %reluInputGradient: memref<?x20x?x?xf32>) {
    mulberry_nn.relu_backward
        ins(%reluInput, %reluOutputGradient : memref<?x20x?x?xf32>,
            memref<?x20x?x?xf32>)
        outs(%reluInputGradient : memref<?x20x?x?xf32>)
    mulberry_nn.max_pool2d_backward
        ins(%poolInput, %poolGradient : memref<?x20x?x?xf32>,
            memref<?x20x?x?xf32>)
        outs(%poolInputGradient : memref<?x20x?x?xf32>) {
          kernel = array<i64: 2, 2>,
          padding = array<i64: 0, 0, 0, 0>,
          strides = array<i64: 2, 2>
        }
    mulberry_nn.conv2d_backward
        ins(%image, %weight, %convGradient : memref<?x1x?x?xf32>,
            memref<20x1x5x5xf32>, memref<?x20x?x?xf32>)
        outs(%imageGradient, %weightGradient, %biasGradient
             : memref<?x1x?x?xf32>, memref<20x1x5x5xf32>, memref<20xf32>) {
          dilations = array<i64: 1, 1>,
          padding = array<i64: 0, 0, 0, 0>,
          strides = array<i64: 1, 1>
        }
    return
  }
}

// CHECK-LABEL: func.func @core_tensor_backward
// CHECK: mulberry_nn.relu_backward
// CHECK: mulberry_nn.max_pool2d_backward
// CHECK-SAME: kernel = array<i64: 2, 2>
// CHECK-SAME: strides = array<i64: 2, 2>
// CHECK: mulberry_nn.conv2d_backward
// CHECK-SAME: dilations = array<i64: 1, 1>
// CHECK-SAME: padding = array<i64: 0, 0, 0, 0>
// CHECK-SAME: strides = array<i64: 1, 1>

// CHECK-LABEL: func.func @dynamic_memref_backward
// CHECK: mulberry_nn.relu_backward
// CHECK: mulberry_nn.max_pool2d_backward
// CHECK: mulberry_nn.conv2d_backward
