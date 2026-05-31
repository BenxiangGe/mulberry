// RUN: cherry-opt --lower-mulberry-tensor %s | FileCheck %s

func.func @unpack_pack(%arg0: memref<2xf32>) -> memref<2xf32> {
  %0 = mulberry.tensor.pack %arg0
      : memref<2xf32> -> <memref<2xf32>>
  %1 = mulberry.tensor.unpack %0
      : <memref<2xf32>> -> memref<2xf32>
  return %1 : memref<2xf32>
}

// CHECK-LABEL: func.func @unpack_pack
// CHECK-SAME: %[[ARG0:.*]]: memref<2xf32>
// CHECK-NOT: mulberry.tensor
// CHECK: [[BASE:%[A-Za-z0-9_]+]], {{.*}} = memref.extract_strided_metadata %[[ARG0]]
// CHECK: [[RECONSTRUCTED:%[A-Za-z0-9_]+]] = memref.reinterpret_cast [[BASE]]
// CHECK-SAME: to memref<2xf32, strided<[?], offset: ?>>
// CHECK: [[CAST:%[A-Za-z0-9_]+]] = memref.cast [[RECONSTRUCTED]]
// CHECK-SAME: to memref<2xf32>
// CHECK: return [[CAST]] : memref<2xf32>

func.func @unpack_pack_dynamic(%arg0: memref<?x?xf32>) -> memref<?x?xf32> {
  %0 = mulberry.tensor.pack %arg0
      : memref<?x?xf32> -> <memref<?x?xf32>>
  %1 = mulberry.tensor.unpack %0
      : <memref<?x?xf32>> -> memref<?x?xf32>
  return %1 : memref<?x?xf32>
}

// CHECK-LABEL: func.func @unpack_pack_dynamic
// CHECK-SAME: %[[ARG0:.*]]: memref<?x?xf32>
// CHECK-NOT: mulberry.tensor
// CHECK: [[BASE:%[A-Za-z0-9_]+]], {{.*}} = memref.extract_strided_metadata %[[ARG0]]
// CHECK: [[RECONSTRUCTED:%[A-Za-z0-9_]+]] = memref.reinterpret_cast [[BASE]]
// CHECK-SAME: to memref<?x?xf32, strided<[?, ?], offset: ?>>
// CHECK: [[CAST:%[A-Za-z0-9_]+]] = memref.cast [[RECONSTRUCTED]]
// CHECK-SAME: to memref<?x?xf32>
// CHECK: return [[CAST]] : memref<?x?xf32>
