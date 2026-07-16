// RUN: mulberry-opt --lower-mulberry %s | FileCheck %s

!key = !mulberry_core.data<"LookupKey">
!key_ref = !mulberry_core.ptr<!key>
!tree = !mulberry_core.data<"Tree__UInt64">
!tree_ref = !mulberry_core.ptr<!tree>

module {
  func.func @by_id(%id: i64) -> !key_ref {
    %result = mulberry_core.data.construct "ById"[0] (%id)
        : (i64) -> !key_ref
    return %result : !key_ref
  }

  func.func @by_name(%name: !mulberry_core.ptr<i8>) -> !key_ref {
    %result = mulberry_core.data.construct "ByName"[1] (%name)
        : (!mulberry_core.ptr<i8>) -> !key_ref
    return %result : !key_ref
  }

  func.func @empty() -> !tree_ref {
    %result = mulberry_core.data.construct "Empty"[0] () : () -> !tree_ref
    return %result : !tree_ref
  }

  func.func @node(%value: i64, %left: !tree_ref, %right: !tree_ref)
      -> !tree_ref {
    %result = mulberry_core.data.construct "Node"[1]
        (%value, %left, %right) : (i64, !tree_ref, !tree_ref) -> !tree_ref
    return %result : !tree_ref
  }
}

// CHECK-DAG: llvm.func @mulberry_boehm_malloc(i64) -> !llvm.ptr
// CHECK-LABEL: func.func @by_id(
// CHECK-SAME: %[[ID:.*]]: i64) -> !llvm.ptr
// CHECK: llvm.call @mulberry_boehm_malloc
// CHECK: llvm.insertvalue {{.*}}[0] : !llvm.struct<(i64, i64)>
// CHECK: llvm.insertvalue %[[ID]], {{.*}}[1] : !llvm.struct<(i64, i64)>
// CHECK: llvm.store {{.*}} : !llvm.struct<(i64, i64)>, !llvm.ptr

// CHECK-LABEL: func.func @by_name(
// CHECK-SAME: %[[NAME:.*]]: !llvm.ptr) -> !llvm.ptr
// CHECK: llvm.store {{.*}} : !llvm.struct<(i64, ptr)>, !llvm.ptr

// CHECK-LABEL: func.func @empty() -> !llvm.ptr
// CHECK: llvm.store {{.*}} : !llvm.struct<(i64)>, !llvm.ptr

// CHECK-LABEL: func.func @node(
// CHECK-SAME: %[[VALUE:.*]]: i64, %[[LEFT:.*]]: !llvm.ptr,
// CHECK-SAME: %[[RIGHT:.*]]: !llvm.ptr) -> !llvm.ptr
// CHECK: llvm.store {{.*}} : !llvm.struct<(i64, i64, ptr, ptr)>, !llvm.ptr
// CHECK-NOT: mulberry_core.
