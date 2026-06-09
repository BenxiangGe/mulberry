// RUN: cherry-opt --lower-mulberry %s | FileCheck %s

module {
  func.func @list_desc_data(%length: index, %index: index,
                            %value: i64) -> i64 {
    %storage = mulberry.list.alloc %length : !mulberry.list_storage<i64>
    mulberry.list.store %value, %storage[%index] : i64,
        !mulberry.list_storage<i64>
    %desc = mulberry.list.desc_pack %length, %storage
        : !mulberry.list_storage<i64> -> !mulberry.list_desc<i64>
    %data = mulberry.list.desc_data %desc
        : !mulberry.list_desc<i64> -> !mulberry.list_storage<i64>
    %loaded = mulberry.list.load %data[%index]
        : !mulberry.list_storage<i64> -> i64
    return %loaded : i64
  }

  func.func @list_desc_length(%length: index) -> index {
    %storage = mulberry.list.alloc %length : !mulberry.list_storage<i64>
    %desc = mulberry.list.desc_pack %length, %storage
        : !mulberry.list_storage<i64> -> !mulberry.list_desc<i64>
    %size = mulberry.list.desc_length %desc : !mulberry.list_desc<i64>
    return %size : index
  }

  func.func @tensor_list_desc_data(%length: index, %index: index,
                                   %value: memref<2xf32>) -> memref<2xf32> {
    %storage = mulberry.list.alloc %length
        : !mulberry.list_storage<memref<2xf32>>
    mulberry.list.store %value, %storage[%index] : memref<2xf32>,
        !mulberry.list_storage<memref<2xf32>>
    %desc = mulberry.list.desc_pack %length, %storage
        : !mulberry.list_storage<memref<2xf32>>
            -> !mulberry.list_desc<memref<2xf32>>
    %data = mulberry.list.desc_data %desc
        : !mulberry.list_desc<memref<2xf32>>
            -> !mulberry.list_storage<memref<2xf32>>
    %loaded = mulberry.list.load %data[%index]
        : !mulberry.list_storage<memref<2xf32>> -> memref<2xf32>
    return %loaded : memref<2xf32>
  }
}

// CHECK-LABEL: func.func @list_desc_data
// CHECK: %[[STORAGE:.*]] = memref.alloc(%{{.*}}) : memref<?xi64>
// CHECK: memref.store {{.*}}, %[[STORAGE]]
// CHECK: %[[LOADED:.*]] = memref.load %[[STORAGE]]
// CHECK: return %[[LOADED]] : i64

// CHECK-LABEL: func.func @list_desc_length
// CHECK: return %{{.*}} : index

// CHECK-LABEL: func.func @tensor_list_desc_data
// CHECK: %[[STORAGE:.*]] = memref.alloc(%{{.*}}) : memref<?xmemref<2xf32>>
// CHECK: memref.store %{{[^,]+}}, %[[STORAGE]]
// CHECK: %[[LOADED:.*]] = memref.load %[[STORAGE]]
// CHECK: return %[[LOADED]] : memref<2xf32>

// CHECK-NOT: mulberry.list.desc_pack
// CHECK-NOT: mulberry.list.desc_data
// CHECK-NOT: mulberry.list.desc_length
