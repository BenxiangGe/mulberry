// RUN: mulberry-opt --load-dialect-plugin=%mulberry_libs/MulberryNNPackage%shlibext --load-pass-plugin=%mulberry_libs/MulberryNNPackage%shlibext --pass-pipeline='builtin.module(prepare-mulberry-nn-calls)' %s | FileCheck %s

!list_u64 = !mulberry_core.record<ListU64 {
  length: i64,
  capacity: i64,
  data: !mulberry_core.ptr<i64>}>
!tensor_f32 = !mulberry_core.record<TensorF32 {
  data: !mulberry_core.ptr<f32>,
  rank: i64,
  numel: i64,
  sizes: !mulberry_core.ptr<!list_u64>,
  strides: !mulberry_core.ptr<!list_u64>}>

module {
  func.func private @mulberry.nn.conv2d(
      !tensor_f32, !tensor_f32, !tensor_f32) -> !tensor_f32
  func.func private @mulberry.nn.maxPool2d(
      !tensor_f32, i64, i64) -> !tensor_f32
  func.func private @mulberry.nn.relu(!tensor_f32) -> !tensor_f32
  func.func private @mulberry.nn.softmax(!tensor_f32) -> !tensor_f32
  func.func private @mulberry.nn.softmaxCrossEntropy(
      !tensor_f32, !tensor_f32) -> f32
  func.func private @mulberry.nn.__tensorAdd1d(
      !tensor_f32, !tensor_f32) -> !tensor_f32
  func.func private @mulberry.nn.__tensorSubtract1d(
      !tensor_f32, !tensor_f32) -> !tensor_f32
  func.func private @mulberry.nn.__tensorScale1d(
      !tensor_f32, f32) -> !tensor_f32
  func.func private @mulberry.nn.__tensorAdd4d(
      !tensor_f32, !tensor_f32) -> !tensor_f32
  func.func private @mulberry.nn.__tensorSubtract4d(
      !tensor_f32, !tensor_f32) -> !tensor_f32
  func.func private @mulberry.nn.__tensorScale4d(
      !tensor_f32, f32) -> !tensor_f32

  func.func @cnn(%input: !tensor_f32, %weight: !tensor_f32,
                 %bias: !tensor_f32) -> !tensor_f32 {
    %height = arith.constant 2 : i64
    %width = arith.constant 2 : i64
    %conv = call @mulberry.nn.conv2d(%input, %weight, %bias)
        : (!tensor_f32, !tensor_f32, !tensor_f32) -> !tensor_f32
    %pool = call @mulberry.nn.maxPool2d(%conv, %height, %width)
        : (!tensor_f32, i64, i64) -> !tensor_f32
    %relu = call @mulberry.nn.relu(%pool)
        : (!tensor_f32) -> !tensor_f32
    return %relu : !tensor_f32
  }

  func.func @classifier(%scores: !tensor_f32) -> !tensor_f32 {
    %result = call @mulberry.nn.softmax(%scores)
        : (!tensor_f32) -> !tensor_f32
    return %result : !tensor_f32
  }

  func.func @cross_entropy(%logits: !tensor_f32,
                           %expected: !tensor_f32) -> f32 {
    %result = call @mulberry.nn.softmaxCrossEntropy(%logits, %expected)
        : (!tensor_f32, !tensor_f32) -> f32
    return %result : f32
  }

  func.func @update_bias(%value: !tensor_f32, %gradient: !tensor_f32,
                         %scale: f32) -> !tensor_f32 {
    %sum = call @mulberry.nn.__tensorAdd1d(%value, %gradient)
        : (!tensor_f32, !tensor_f32) -> !tensor_f32
    %scaled = call @mulberry.nn.__tensorScale1d(%sum, %scale)
        : (!tensor_f32, f32) -> !tensor_f32
    %updated = call @mulberry.nn.__tensorSubtract1d(%value, %scaled)
        : (!tensor_f32, !tensor_f32) -> !tensor_f32
    return %updated : !tensor_f32
  }

  func.func @update_filter(%value: !tensor_f32, %gradient: !tensor_f32,
                           %scale: f32) -> !tensor_f32 {
    %sum = call @mulberry.nn.__tensorAdd4d(%value, %gradient)
        : (!tensor_f32, !tensor_f32) -> !tensor_f32
    %scaled = call @mulberry.nn.__tensorScale4d(%sum, %scale)
        : (!tensor_f32, f32) -> !tensor_f32
    %updated = call @mulberry.nn.__tensorSubtract4d(%value, %scaled)
        : (!tensor_f32, !tensor_f32) -> !tensor_f32
    return %updated : !tensor_f32
  }
}

// CHECK-LABEL: func.func @cnn
// CHECK: mulberry_core.tensor.view
// CHECK-SAME: -> !mulberry_core.tensor<?x?x?x?xf32>
// CHECK: mulberry_core.tensor.view
// CHECK-SAME: -> !mulberry_core.tensor<?x?x?x?xf32>
// CHECK: mulberry_core.tensor.view
// CHECK-SAME: -> !mulberry_core.tensor<?xf32>
// CHECK: call @mulberry.nn.__tensor.conv2d
// CHECK-SAME: -> !mulberry_core.tensor<?x?x?x?xf32>
// CHECK: call @mulberry.nn.__tensor.maxPool2d
// CHECK-SAME: -> !mulberry_core.tensor<?x?x?x?xf32>
// CHECK: call @mulberry.nn.__tensor.relu
// CHECK-SAME: -> !mulberry_core.tensor<?x?x?x?xf32>
// CHECK: mulberry_core.tensor.pack
// CHECK-NEXT: mulberry_core.tensor.release

// CHECK-LABEL: func.func @classifier
// CHECK: mulberry_core.tensor.view
// CHECK-SAME: -> !mulberry_core.tensor<?x?xf32>
// CHECK: call @mulberry.nn.__tensor.softmax
// CHECK-SAME: -> !mulberry_core.tensor<?x?xf32>
// CHECK: mulberry_core.tensor.pack
// CHECK-NEXT: mulberry_core.tensor.release
// CHECK-NOT: call @mulberry.nn.softmax

// CHECK-LABEL: func.func @cross_entropy
// CHECK-COUNT-2: mulberry_core.tensor.view
// CHECK: call @mulberry.nn.__tensor.softmaxCrossEntropy
// CHECK-SAME: -> f32
// CHECK-NOT: mulberry_core.tensor.pack
// CHECK-NOT: call @mulberry.nn.softmaxCrossEntropy

// CHECK-LABEL: func.func @update_bias
// CHECK: call @mulberry.nn.__tensor.__tensorAdd1d
// CHECK-SAME: -> !mulberry_core.tensor<?xf32>
// CHECK: call @mulberry.nn.__tensor.__tensorScale1d
// CHECK-SAME: -> !mulberry_core.tensor<?xf32>
// CHECK: call @mulberry.nn.__tensor.__tensorSubtract1d
// CHECK-SAME: -> !mulberry_core.tensor<?xf32>
// CHECK-NOT: call @mulberry.nn.__tensorAdd

// CHECK-LABEL: func.func @update_filter
// CHECK: call @mulberry.nn.__tensor.__tensorAdd4d
// CHECK-SAME: -> !mulberry_core.tensor<?x?x?x?xf32>
// CHECK: call @mulberry.nn.__tensor.__tensorScale4d
// CHECK-SAME: -> !mulberry_core.tensor<?x?x?x?xf32>
// CHECK: call @mulberry.nn.__tensor.__tensorSubtract4d
// CHECK-SAME: -> !mulberry_core.tensor<?x?x?x?xf32>
// CHECK-NOT: call @mulberry.nn.__tensorAdd
