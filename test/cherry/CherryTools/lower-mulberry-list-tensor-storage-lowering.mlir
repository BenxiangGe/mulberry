// RUN: cherry-opt --lower-mulberry %s | FileCheck %s

module {
  func.func @tensor_list(%index: index) -> !mulberry.tensor<2xf32> {
    %a = mulberry.tensor.alloc() : !mulberry.tensor<2xf32>
    %b = mulberry.tensor.alloc() : !mulberry.tensor<2xf32>
    %list = mulberry.list.create(%a, %b)
        : (!mulberry.tensor<2xf32>, !mulberry.tensor<2xf32>)
            -> !mulberry.list<!mulberry.tensor<2xf32>>
    %item = mulberry.list.get %list[%index]
        : !mulberry.list<!mulberry.tensor<2xf32>>
            -> !mulberry.tensor<2xf32>
    %size = mulberry.list.size %list
        : !mulberry.list<!mulberry.tensor<2xf32>>
    return %item : !mulberry.tensor<2xf32>
  }
}

// CHECK-LABEL: func.func @tensor_list
// CHECK-SAME: -> memref<2xf32>
// CHECK: %[[A:.*]] = memref.alloc() : memref<2xf32>
// CHECK: %[[B:.*]] = memref.alloc() : memref<2xf32>
// CHECK: %[[LIST:.*]] = memref.alloc(%{{.*}}) : memref<?xmemref<2xf32>>
// CHECK: memref.store %[[A]], %[[LIST]]
// CHECK: memref.store %[[B]], %[[LIST]]
// CHECK: %[[ITEM:.*]] = memref.load %[[LIST]]
// CHECK: memref.dim %[[LIST]]
// CHECK: return %[[ITEM]] : memref<2xf32>

// CHECK-NOT: mulberry.list.create
// CHECK-NOT: mulberry.list.get
// CHECK-NOT: mulberry.list.size
// CHECK-NOT: mulberry.list.alloc
// CHECK-NOT: mulberry.list.store
// CHECK-NOT: mulberry.list.load
// CHECK-NOT: mulberry.list.length
