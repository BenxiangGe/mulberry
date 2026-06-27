// RUN: mulberry-opt --lower-mulberry %s | FileCheck %s

module {
  func.func @alloc_scalar() -> !mulberry.ptr<i64> {
    %count = arith.constant 1 : index
    %ptr = mulberry.heap.alloc i64, %count : !mulberry.ptr<i64>
    return %ptr : !mulberry.ptr<i64>
  }

  func.func @alloc_record(%count: index)
      -> !mulberry.ptr<!mulberry.record<Person {age: i64, active: i1}>> {
    %ptr = mulberry.heap.alloc !mulberry.record<Person {age: i64, active: i1}>, %count
        : !mulberry.ptr<!mulberry.record<Person {age: i64, active: i1}>>
    return %ptr : !mulberry.ptr<!mulberry.record<Person {age: i64, active: i1}>>
  }
}

// CHECK-DAG: llvm.func @mulberry_boehm_malloc(i64) -> !llvm.ptr
// CHECK-LABEL: func.func @alloc_scalar() -> !llvm.ptr
// CHECK: %[[I64_NULL:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK: %[[I64_NEXT:.*]] = llvm.getelementptr %[[I64_NULL]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK: %[[I64_SIZE:.*]] = llvm.ptrtoint %[[I64_NEXT]] : !llvm.ptr to i64
// CHECK: %[[I64_COUNT:.*]] = arith.index_cast %{{.*}} : index to i64
// CHECK: %[[I64_BYTES:.*]] = arith.muli %[[I64_SIZE]], %[[I64_COUNT]] : i64
// CHECK: %[[I64_PTR:.*]] = llvm.call @mulberry_boehm_malloc(%[[I64_BYTES]]) : (i64) -> !llvm.ptr
// CHECK: return %[[I64_PTR]] : !llvm.ptr

// CHECK-LABEL: func.func @alloc_record(
// CHECK-SAME: %[[COUNT:.*]]: index) -> !llvm.ptr
// CHECK: %[[REC_NULL:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK: %[[REC_NEXT:.*]] = llvm.getelementptr %[[REC_NULL]][1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i1)>
// CHECK: %[[REC_SIZE:.*]] = llvm.ptrtoint %[[REC_NEXT]] : !llvm.ptr to i64
// CHECK: %[[REC_COUNT:.*]] = arith.index_cast %[[COUNT]] : index to i64
// CHECK: %[[REC_BYTES:.*]] = arith.muli %[[REC_SIZE]], %[[REC_COUNT]] : i64
// CHECK: %[[REC_PTR:.*]] = llvm.call @mulberry_boehm_malloc(%[[REC_BYTES]]) : (i64) -> !llvm.ptr
// CHECK: return %[[REC_PTR]] : !llvm.ptr
// CHECK-NOT: mulberry.
