// RUN: cherry-opt --lower-mulberry %s | FileCheck %s

module {
  func.func @scalar_list(%index: index) -> i64 {
    %a = arith.constant 10 : i64
    %b = arith.constant 20 : i64
    %list = mulberry.list.create(%a, %b)
        : (i64, i64) -> !mulberry.list<i64>
    %item = mulberry.list.get %list[%index]
        : !mulberry.list<i64> -> i64
    %size = mulberry.list.size %list : !mulberry.list<i64>
    return %item : i64
  }
}

// CHECK-LABEL: func.func @scalar_list
// CHECK: %[[SCALAR_LIST:.*]] = memref.alloc(%{{.*}}) : memref<?xi64>
// CHECK: memref.store {{.*}}, %[[SCALAR_LIST]]
// CHECK: %[[ITEM:.*]] = memref.load %[[SCALAR_LIST]]
// CHECK: %[[LENGTH:.*]] = memref.dim %[[SCALAR_LIST]]
// CHECK: arith.index_cast
// CHECK: memref.dealloc %[[SCALAR_LIST]] : memref<?xi64>

// CHECK-NOT: mulberry.list.create
// CHECK-NOT: mulberry.list.get
// CHECK-NOT: mulberry.list.size
// CHECK-NOT: mulberry.list.alloc
// CHECK-NOT: mulberry.list.store
// CHECK-NOT: mulberry.list.load
// CHECK-NOT: mulberry.list.length
