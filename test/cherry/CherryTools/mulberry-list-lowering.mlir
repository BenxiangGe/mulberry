// RUN: cherry-opt --lower-mulberry-list --lower-mulberry-tensor %s | FileCheck %s

func.func @list_tensor_get(%arg0: memref<2xf32>, %arg1: memref<2xf32>,
                           %index: index) -> memref<2xf32> {
  %0 = mulberry.tensor.pack %arg0
      : memref<2xf32> -> <memref<2xf32>>
  %1 = mulberry.tensor.pack %arg1
      : memref<2xf32> -> <memref<2xf32>>
  %2 = mulberry.list.create %0, %1
      : (!mulberry.tensor_desc<memref<2xf32>>,
         !mulberry.tensor_desc<memref<2xf32>>)
      -> !mulberry.list<!mulberry.tensor_desc<memref<2xf32>>>
  %3 = mulberry.list.get %2[%index]
      : <!mulberry.tensor_desc<memref<2xf32>>>
      -> !mulberry.tensor_desc<memref<2xf32>>
  %4 = mulberry.tensor.unpack %3
      : <memref<2xf32>> -> memref<2xf32>
  return %4 : memref<2xf32>
}

// CHECK-LABEL: func.func @list_tensor_get
// CHECK-NOT: mulberry.list
// CHECK-NOT: mulberry.tensor
// CHECK-NOT: scf.index_switch
// CHECK: memref.alloc() : memref<1xi64>
// CHECK: memref.alloc() : memref<2xmemref<f32>>
// CHECK: memref.alloc() : memref<2xindex>
// CHECK: memref.alloc() : memref<2x1xindex>
// CHECK: memref.extract_strided_metadata
// CHECK: memref.store
// CHECK: memref.load
// CHECK: memref.reinterpret_cast
// CHECK: return

func.func @list_tensor_get_twice(%arg0: memref<2xf32>, %arg1: memref<2xf32>,
                                 %i: index, %j: index)
    -> (memref<2xf32>, memref<2xf32>) {
  %0 = mulberry.tensor.pack %arg0
      : memref<2xf32> -> <memref<2xf32>>
  %1 = mulberry.tensor.pack %arg1
      : memref<2xf32> -> <memref<2xf32>>
  %2 = mulberry.list.create %0, %1
      : (!mulberry.tensor_desc<memref<2xf32>>,
         !mulberry.tensor_desc<memref<2xf32>>)
      -> !mulberry.list<!mulberry.tensor_desc<memref<2xf32>>>
  %3 = mulberry.list.get %2[%i]
      : <!mulberry.tensor_desc<memref<2xf32>>>
      -> !mulberry.tensor_desc<memref<2xf32>>
  %4 = mulberry.tensor.unpack %3
      : <memref<2xf32>> -> memref<2xf32>
  %5 = mulberry.list.get %2[%j]
      : <!mulberry.tensor_desc<memref<2xf32>>>
      -> !mulberry.tensor_desc<memref<2xf32>>
  %6 = mulberry.tensor.unpack %5
      : <memref<2xf32>> -> memref<2xf32>
  return %4, %6 : memref<2xf32>, memref<2xf32>
}

// CHECK-LABEL: func.func @list_tensor_get_twice
// CHECK-NOT: mulberry.list
// CHECK-NOT: mulberry.tensor
// CHECK-COUNT-5: memref.alloc
// CHECK: memref.extract_strided_metadata
// CHECK: memref.extract_strided_metadata
// CHECK: memref.reinterpret_cast
// CHECK: memref.reinterpret_cast
// CHECK: return

func.func @list_dynamic_tensor_get(%arg0: memref<?xf32>, %arg1: memref<?xf32>,
                                   %index: index) -> memref<?xf32> {
  %0 = mulberry.tensor.pack %arg0
      : memref<?xf32> -> <memref<?xf32>>
  %1 = mulberry.tensor.pack %arg1
      : memref<?xf32> -> <memref<?xf32>>
  %2 = mulberry.list.create %0, %1
      : (!mulberry.tensor_desc<memref<?xf32>>,
         !mulberry.tensor_desc<memref<?xf32>>)
      -> !mulberry.list<!mulberry.tensor_desc<memref<?xf32>>>
  %3 = mulberry.list.get %2[%index]
      : <!mulberry.tensor_desc<memref<?xf32>>>
      -> !mulberry.tensor_desc<memref<?xf32>>
  %4 = mulberry.tensor.unpack %3
      : <memref<?xf32>> -> memref<?xf32>
  return %4 : memref<?xf32>
}

// CHECK-LABEL: func.func @list_dynamic_tensor_get
// CHECK-NOT: mulberry.list
// CHECK-NOT: mulberry.tensor
// CHECK-NOT: scf.index_switch
// CHECK: memref.alloc() : memref<1xi64>
// CHECK: memref.alloc() : memref<2xmemref<f32>>
// CHECK: memref.alloc() : memref<2xindex>
// CHECK: memref.alloc() : memref<2x1xindex>
// CHECK: memref.extract_strided_metadata
// CHECK: memref.load
// CHECK: memref.reinterpret_cast
// CHECK: return

func.func @list_i64_get(%index: index) -> i64 {
  %0 = arith.constant 10 : i64
  %1 = arith.constant 20 : i64
  %2 = mulberry.list.create %0, %1
      : (i64, i64) -> !mulberry.list<i64>
  %3 = mulberry.list.get %2[%index] : <i64> -> i64
  return %3 : i64
}

// CHECK-LABEL: func.func @list_i64_get
// CHECK-NOT: mulberry.list
// CHECK-NOT: scf.index_switch
// CHECK: memref.alloc() : memref<1xi64>
// CHECK: memref.alloc() : memref<2xi64>
// CHECK: memref.store
// CHECK: memref.store
// CHECK: memref.load
// CHECK: return

func.func @list_i64_size() -> i64 {
  %0 = arith.constant 10 : i64
  %1 = arith.constant 20 : i64
  %2 = mulberry.list.create %0, %1
      : (i64, i64) -> !mulberry.list<i64>
  %3 = mulberry.list.size %2 : !mulberry.list<i64> -> i64
  return %3 : i64
}

// CHECK-LABEL: func.func @list_i64_size
// CHECK-NOT: mulberry.list
// CHECK: memref.alloc() : memref<1xi64>
// CHECK: memref.alloc() : memref<2xi64>
// CHECK: memref.store
// CHECK: memref.load
// CHECK: return

func.func @list_tensor_size(%arg0: memref<2xf32>,
                            %arg1: memref<2xf32>) -> i64 {
  %0 = mulberry.tensor.pack %arg0
      : memref<2xf32> -> <memref<2xf32>>
  %1 = mulberry.tensor.pack %arg1
      : memref<2xf32> -> <memref<2xf32>>
  %2 = mulberry.list.create %0, %1
      : (!mulberry.tensor_desc<memref<2xf32>>,
         !mulberry.tensor_desc<memref<2xf32>>)
      -> !mulberry.list<!mulberry.tensor_desc<memref<2xf32>>>
  %3 = mulberry.list.size %2
      : !mulberry.list<!mulberry.tensor_desc<memref<2xf32>>> -> i64
  return %3 : i64
}

// CHECK-LABEL: func.func @list_tensor_size
// CHECK-NOT: mulberry.list
// CHECK-NOT: mulberry.tensor
// CHECK: memref.alloc() : memref<1xi64>
// CHECK: memref.store
// CHECK: memref.load
// CHECK: return
