// RUN: cherry-opt --lower-mulberry %s | FileCheck %s

module {
  func.func @main() -> i64 {
    %a = mulberry.tensor.alloc() : !mulberry.tensor<2x3xf32>
    %b = mulberry.tensor.alloc() : !mulberry.tensor<3x4xf32>
    %c = mulberry.tensor.alloc() : !mulberry.tensor<2x4xf32>
    cherry_nn.matmul ins(%a, %b : !mulberry.tensor<2x3xf32>,
                         !mulberry.tensor<3x4xf32>)
                     outs(%c : !mulberry.tensor<2x4xf32>)

    %d = mulberry.tensor.alloc() : !mulberry.tensor<2x4xf32>
    cherry_nn.sigmoid ins(%c : !mulberry.tensor<2x4xf32>)
                      outs(%d : !mulberry.tensor<2x4xf32>)

    %result = cherry_nn.argmax %d : !mulberry.tensor<2x4xf32> -> i64
    return %result : i64
  }
}

// CHECK-LABEL: func.func @main
// CHECK: memref.alloc() : memref<2x3xf32>
// CHECK: memref.alloc() : memref<3x4xf32>
// CHECK: memref.alloc() : memref<2x4xf32>
// CHECK: linalg.fill
// CHECK: linalg.matmul
// CHECK: linalg.map
// CHECK: math.exp
// CHECK: linalg.generic
// CHECK: return {{.*}} : i64
// CHECK-NOT: cherry_nn.
// CHECK-NOT: mulberry.
