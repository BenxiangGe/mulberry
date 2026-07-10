// RUN: %numpy_python %mulberry_src_root/tools/check_cnn_backward.py | FileCheck %s --check-prefix=NUMPY
// RUN: mulberry-opt --load-dialect-plugin=%mulberry_libs/MulberryNNPackage%shlibext --load-pass-plugin=%mulberry_libs/MulberryNNPackage%shlibext --pass-pipeline='builtin.module(lower-mulberry-nn,func.func(convert-linalg-to-loops),lower-affine,convert-scf-to-cf,func.func(convert-arith-to-llvm),finalize-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)' %s | mlir-runner -e main --entry-point-result=i32 | FileCheck %s --check-prefix=JIT

memref.global "private" constant @reluInput : memref<4xf32> =
    dense<[-1.0, 0.0, 2.0, 3.0]>
memref.global "private" constant @reluOutputGradient : memref<4xf32> =
    dense<[1.0, 2.0, 3.0, 4.0]>
memref.global "private" constant @reluExpected : memref<4xf32> =
    dense<[0.0, 0.0, 3.0, 4.0]>

memref.global "private" constant @convInput : memref<1x1x4x4xf32> = dense<
    [[[[1.0, 2.0, 3.0, 4.0],
       [5.0, 6.0, 7.0, 8.0],
       [9.0, 10.0, 11.0, 12.0],
       [13.0, 14.0, 15.0, 16.0]]]]>
memref.global "private" constant @convWeight : memref<1x1x2x2xf32> =
    dense<[[[[1.5, -2.0], [0.5, 3.0]]]]>
memref.global "private" constant @convOutputGradient : memref<1x1x2x2xf32> =
    dense<[[[[1.0, -2.0], [0.5, 3.0]]]]>
memref.global "private" constant @convInputGradientExpected
    : memref<1x1x4x4xf32> = dense<
        [[[[0.0, 0.0, 0.0, 0.0],
           [2.0, 3.5, -12.0, 0.0],
           [0.0, 0.0, 0.0, 0.0],
           [1.5, 1.5, 9.0, 0.0]]]]>
memref.global "private" constant @convWeightGradientExpected
    : memref<1x1x2x2xf32> = dense<[[[[18.0, 23.5], [30.0, 42.5]]]]>
memref.global "private" constant @convBiasGradientExpected : memref<1xf32> =
    dense<[2.5]>

memref.global "private" constant @poolInput : memref<1x1x3x3xf32> =
    dense<[[[[1.0, 2.0, 2.0], [3.0, 3.0, 0.0], [3.0, 1.0, 4.0]]]]>
memref.global "private" constant @poolOutputGradient : memref<1x1x3x3xf32> =
    dense<[[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]>
memref.global "private" constant @poolInputGradientExpected
    : memref<1x1x3x3xf32> =
        dense<[[[[1.0, 5.0, 0.0], [24.0, 6.0, 0.0], [0.0, 0.0, 9.0]]]]>

func.func @check1d(%actual: memref<?xf32>, %expected: memref<?xf32>,
                   %mismatches: memref<i32>) {
  %zero = arith.constant 0.0 : f32
  %tolerance = arith.constant 1.0e-5 : f32
  linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
    iterator_types = ["reduction"]
  } ins(%actual, %expected : memref<?xf32>, memref<?xf32>)
    outs(%mismatches : memref<i32>) {
  ^bb0(%actualValue: f32, %expectedValue: f32, %mismatchCount: i32):
    %difference = arith.subf %actualValue, %expectedValue : f32
    %negative = arith.negf %difference : f32
    %isNegative = arith.cmpf olt, %difference, %zero : f32
    %absolute = arith.select %isNegative, %negative, %difference : f32
    %isUnordered = arith.cmpf uno, %actualValue, %expectedValue : f32
    %isTooLarge = arith.cmpf ogt, %absolute, %tolerance : f32
    %isMismatch = arith.ori %isUnordered, %isTooLarge : i1
    %increment = arith.extui %isMismatch : i1 to i32
    %nextCount = arith.addi %mismatchCount, %increment : i32
    linalg.yield %nextCount : i32
  }
  return
}

func.func @check4d(%actual: memref<?x?x?x?xf32>,
                   %expected: memref<?x?x?x?xf32>,
                   %mismatches: memref<i32>) {
  %zero = arith.constant 0.0 : f32
  %tolerance = arith.constant 1.0e-5 : f32
  linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> ()>
    ],
    iterator_types = ["reduction", "reduction", "reduction", "reduction"]
  } ins(%actual, %expected : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>)
    outs(%mismatches : memref<i32>) {
  ^bb0(%actualValue: f32, %expectedValue: f32, %mismatchCount: i32):
    %difference = arith.subf %actualValue, %expectedValue : f32
    %negative = arith.negf %difference : f32
    %isNegative = arith.cmpf olt, %difference, %zero : f32
    %absolute = arith.select %isNegative, %negative, %difference : f32
    %isUnordered = arith.cmpf uno, %actualValue, %expectedValue : f32
    %isTooLarge = arith.cmpf ogt, %absolute, %tolerance : f32
    %isMismatch = arith.ori %isUnordered, %isTooLarge : i1
    %increment = arith.extui %isMismatch : i1 to i32
    %nextCount = arith.addi %mismatchCount, %increment : i32
    linalg.yield %nextCount : i32
  }
  return
}

func.func @main() -> i32 {
  %reluInput = memref.get_global @reluInput : memref<4xf32>
  %reluOutputGradient = memref.get_global @reluOutputGradient : memref<4xf32>
  %reluExpected = memref.get_global @reluExpected : memref<4xf32>
  %convInput = memref.get_global @convInput : memref<1x1x4x4xf32>
  %convWeight = memref.get_global @convWeight : memref<1x1x2x2xf32>
  %convOutputGradient =
      memref.get_global @convOutputGradient : memref<1x1x2x2xf32>
  %convInputGradientExpected =
      memref.get_global @convInputGradientExpected : memref<1x1x4x4xf32>
  %convWeightGradientExpected =
      memref.get_global @convWeightGradientExpected : memref<1x1x2x2xf32>
  %convBiasGradientExpected =
      memref.get_global @convBiasGradientExpected : memref<1xf32>
  %poolInput = memref.get_global @poolInput : memref<1x1x3x3xf32>
  %poolOutputGradient =
      memref.get_global @poolOutputGradient : memref<1x1x3x3xf32>
  %poolInputGradientExpected =
      memref.get_global @poolInputGradientExpected : memref<1x1x3x3xf32>

  %reluInputGradient = memref.alloc() : memref<4xf32>
  %convInputGradient = memref.alloc() : memref<1x1x4x4xf32>
  %convWeightGradient = memref.alloc() : memref<1x1x2x2xf32>
  %convBiasGradient = memref.alloc() : memref<1xf32>
  %poolInputGradient = memref.alloc() : memref<1x1x3x3xf32>
  %mismatches = memref.alloca() : memref<i32>
  %zero = arith.constant 0 : i32
  memref.store %zero, %mismatches[] : memref<i32>

  mulberry_nn.relu_backward
      ins(%reluInput, %reluOutputGradient : memref<4xf32>, memref<4xf32>)
      outs(%reluInputGradient : memref<4xf32>)
  mulberry_nn.conv2d_backward
      ins(%convInput, %convWeight, %convOutputGradient
          : memref<1x1x4x4xf32>, memref<1x1x2x2xf32>,
            memref<1x1x2x2xf32>)
      outs(%convInputGradient, %convWeightGradient, %convBiasGradient
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

  %reluInputGradientDynamic = memref.cast %reluInputGradient
      : memref<4xf32> to memref<?xf32>
  %reluExpectedDynamic = memref.cast %reluExpected
      : memref<4xf32> to memref<?xf32>
  func.call @check1d(%reluInputGradientDynamic, %reluExpectedDynamic,
                     %mismatches)
      : (memref<?xf32>, memref<?xf32>, memref<i32>) -> ()

  %convInputGradientDynamic = memref.cast %convInputGradient
      : memref<1x1x4x4xf32> to memref<?x?x?x?xf32>
  %convInputGradientExpectedDynamic = memref.cast %convInputGradientExpected
      : memref<1x1x4x4xf32> to memref<?x?x?x?xf32>
  func.call @check4d(%convInputGradientDynamic,
                     %convInputGradientExpectedDynamic, %mismatches)
      : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<i32>) -> ()

  %convWeightGradientDynamic = memref.cast %convWeightGradient
      : memref<1x1x2x2xf32> to memref<?x?x?x?xf32>
  %convWeightGradientExpectedDynamic = memref.cast %convWeightGradientExpected
      : memref<1x1x2x2xf32> to memref<?x?x?x?xf32>
  func.call @check4d(%convWeightGradientDynamic,
                     %convWeightGradientExpectedDynamic, %mismatches)
      : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<i32>) -> ()

  %convBiasGradientDynamic = memref.cast %convBiasGradient
      : memref<1xf32> to memref<?xf32>
  %convBiasGradientExpectedDynamic = memref.cast %convBiasGradientExpected
      : memref<1xf32> to memref<?xf32>
  func.call @check1d(%convBiasGradientDynamic,
                     %convBiasGradientExpectedDynamic, %mismatches)
      : (memref<?xf32>, memref<?xf32>, memref<i32>) -> ()

  %poolInputGradientDynamic = memref.cast %poolInputGradient
      : memref<1x1x3x3xf32> to memref<?x?x?x?xf32>
  %poolInputGradientExpectedDynamic = memref.cast %poolInputGradientExpected
      : memref<1x1x3x3xf32> to memref<?x?x?x?xf32>
  func.call @check4d(%poolInputGradientDynamic,
                     %poolInputGradientExpectedDynamic, %mismatches)
      : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<i32>) -> ()

  %result = memref.load %mismatches[] : memref<i32>
  memref.dealloc %reluInputGradient : memref<4xf32>
  memref.dealloc %convInputGradient : memref<1x1x4x4xf32>
  memref.dealloc %convWeightGradient : memref<1x1x2x2xf32>
  memref.dealloc %convBiasGradient : memref<1xf32>
  memref.dealloc %poolInputGradient : memref<1x1x3x3xf32>
  return %result : i32
}

// NUMPY: NumPy CNN backward expected gradients and finite differences: ok
// JIT: 0
