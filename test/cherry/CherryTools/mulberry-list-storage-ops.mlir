// RUN: cherry-opt %s | FileCheck %s

module {
  func.func @main(%length: index, %index: index, %value: i64) -> i64 {
    %storage = mulberry.list.alloc %length : !mulberry.list_storage<i64>
    mulberry.list.store %value, %storage[%index] : i64,
        !mulberry.list_storage<i64>
    %loaded = mulberry.list.load %storage[%index]
        : !mulberry.list_storage<i64> -> i64
    %size = mulberry.list.length %storage : !mulberry.list_storage<i64>
    return %loaded : i64
  }

  func.func @tensor_storage(%length: index, %index: index,
                            %value: memref<2xf32>) -> memref<2xf32> {
    %storage = mulberry.list.alloc %length
        : !mulberry.list_storage<memref<2xf32>>
    mulberry.list.store %value, %storage[%index] : memref<2xf32>,
        !mulberry.list_storage<memref<2xf32>>
    %loaded = mulberry.list.load %storage[%index]
        : !mulberry.list_storage<memref<2xf32>> -> memref<2xf32>
    %size = mulberry.list.length %storage
        : !mulberry.list_storage<memref<2xf32>>
    return %loaded : memref<2xf32>
  }
}

// CHECK: mulberry.list.alloc
// CHECK: !mulberry.list_storage<i64>
// CHECK: mulberry.list.store
// CHECK: mulberry.list.load
// CHECK: mulberry.list.length
// CHECK-LABEL: func.func @tensor_storage
// CHECK: mulberry.list.alloc
// CHECK: !mulberry.list_storage<memref<2xf32>>
// CHECK: mulberry.list.store
// CHECK-SAME: memref<2xf32>
// CHECK: mulberry.list.load
// CHECK-SAME: !mulberry.list_storage<memref<2xf32>> -> memref<2xf32>
// CHECK: mulberry.list.length
