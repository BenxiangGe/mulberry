// RUN: mulberry-opt --load-dialect-plugin=%mulberry_libs/MulberryNNPackage%shlibext --load-pass-plugin=%mulberry_libs/MulberryNNPackage%shlibext --pass-pipeline='builtin.module(lower-mulberry-nn)' %s | FileCheck %s --check-prefix=LOWER
// RUN: mulberry-opt --load-dialect-plugin=%mulberry_libs/MulberryNNPackage%shlibext --load-pass-plugin=%mulberry_libs/MulberryNNPackage%shlibext --pass-pipeline='builtin.module(prepare-mulberry-nn-calls,lower-mulberry-nn)' %s | FileCheck %s --check-prefix=CALL

module {
  func.func @main(%lhs: memref<2x3xf32>, %rhs: memref<3x4xf32>,
                  %out: memref<2x4xf32>) -> i64 {
    mulberry_nn.matmul ins(%lhs, %rhs : memref<2x3xf32>, memref<3x4xf32>)
        outs(%out : memref<2x4xf32>)
    %result = mulberry_nn.argmax %out : memref<2x4xf32> -> i64
    return %result : i64
  }

  func.func private @mulberry.nn.__tensor.sigmoid(memref<?x?xf32>) -> memref<?x?xf32>
  func.func private @mulberry.nn.__tensor.sigmoidPrime(memref<?x?xf32>) -> memref<?x?xf32>
  func.func private @mulberry.nn.__tensor.exp(memref<?x?xf32>) -> memref<?x?xf32>
  func.func private @mulberry.nn.__tensor.transpose(memref<?x?xf32>) -> memref<?x?xf32>
  func.func private @mulberry.nn.__tensor.matmul(memref<?x?xf32>, memref<?x?xf32>) -> memref<?x?xf32>
  func.func private @mulberry.nn.__tensor.matadd(memref<?x?xf32>, memref<?x?xf32>) -> memref<?x?xf32>
  func.func private @mulberry.nn.__tensor.matsub(memref<?x?xf32>, memref<?x?xf32>) -> memref<?x?xf32>
  func.func private @mulberry.nn.__tensor.hadamard(memref<?x?xf32>, memref<?x?xf32>) -> memref<?x?xf32>
  func.func private @mulberry.nn.__tensor.matscale(memref<?x?xf32>, f32) -> memref<?x?xf32>
  func.func private @mulberry.nn.__tensor.argmax(memref<?x?xf32>) -> i64

  func.func @package_matmul(%lhs: memref<?x?xf32>, %rhs: memref<?x?xf32>)
      -> memref<?x?xf32> {
    %result = call @mulberry.nn.__tensor.matmul(%lhs, %rhs)
        : (memref<?x?xf32>, memref<?x?xf32>) -> memref<?x?xf32>
    return %result : memref<?x?xf32>
  }

  func.func @package_transpose(%input: memref<?x?xf32>) -> memref<?x?xf32> {
    %result = call @mulberry.nn.__tensor.transpose(%input)
        : (memref<?x?xf32>) -> memref<?x?xf32>
    return %result : memref<?x?xf32>
  }

  func.func @package_calls(%input: memref<?x?xf32>, %other: memref<?x?xf32>,
                           %scale: f32) -> i64 {
    %sigmoid = call @mulberry.nn.__tensor.sigmoid(%input)
        : (memref<?x?xf32>) -> memref<?x?xf32>
    %prime = call @mulberry.nn.__tensor.sigmoidPrime(%sigmoid)
        : (memref<?x?xf32>) -> memref<?x?xf32>
    %exp = call @mulberry.nn.__tensor.exp(%prime)
        : (memref<?x?xf32>) -> memref<?x?xf32>
    %transpose = call @mulberry.nn.__tensor.transpose(%exp)
        : (memref<?x?xf32>) -> memref<?x?xf32>
    %sum = call @mulberry.nn.__tensor.matadd(%transpose, %other)
        : (memref<?x?xf32>, memref<?x?xf32>) -> memref<?x?xf32>
    %diff = call @mulberry.nn.__tensor.matsub(%sum, %other)
        : (memref<?x?xf32>, memref<?x?xf32>) -> memref<?x?xf32>
    %product = call @mulberry.nn.__tensor.hadamard(%diff, %other)
        : (memref<?x?xf32>, memref<?x?xf32>) -> memref<?x?xf32>
    %scaled = call @mulberry.nn.__tensor.matscale(%product, %scale)
        : (memref<?x?xf32>, f32) -> memref<?x?xf32>
    %result = call @mulberry.nn.__tensor.argmax(%scaled) : (memref<?x?xf32>) -> i64
    return %result : i64
  }
}

// LOWER-LABEL: func.func @main
// LOWER: linalg.fill
// LOWER: linalg.generic
// LOWER: arith.cmpf ogt
// LOWER-COUNT-2: memref.dealloc
// LOWER: return
// LOWER-NOT: mulberry_nn.

// CALL-LABEL: func.func @package_matmul
// CALL: memref.dim %arg0, %c0
// CALL: memref.dim %arg1, %c1
// CALL: memref.alloc
// CALL: linalg.fill
// CALL: linalg.generic
// CALL: return

// CALL-LABEL: func.func @package_transpose
// CALL: memref.dim %arg0, %c1
// CALL: memref.dim %arg0, %c0
// CALL: memref.alloc
// CALL: linalg.transpose
// CALL: return

// CALL-LABEL: func.func @package_calls
// CALL: memref.alloc
// CALL: linalg.map
// CALL: linalg.transpose
// CALL: linalg.add
// CALL: linalg.sub
// CALL: linalg.mul
// CALL: arith.cmpf ogt
// CALL: return
// CALL-NOT: call @mulberry.nn
// CALL-NOT: mulberry_nn.
