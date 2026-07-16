// RUN: mulberry-opt --lower-mulberry %s | FileCheck %s

!key = !mulberry_core.data<"LookupKey">
!key_ref = !mulberry_core.ptr<!key>
!tree = !mulberry_core.data<"Tree__UInt64">
!tree_ref = !mulberry_core.ptr<!tree>

module {
  func.func @tag(%value: !key_ref) -> i64 {
    %tag = mulberry_core.data.tag %value : !key_ref -> i64
    return %tag : i64
  }

  func.func @by_id(%value: !key_ref) -> i64 {
    %id = mulberry_core.data.unpack "ById"[0] %value
        : (!key_ref) -> (i64)
    return %id : i64
  }

  func.func @by_name(%value: !key_ref) -> !mulberry_core.ptr<i8> {
    %name = mulberry_core.data.unpack "ByName"[1] %value
        : (!key_ref) -> (!mulberry_core.ptr<i8>)
    return %name : !mulberry_core.ptr<i8>
  }

  func.func @empty(%value: !tree_ref) {
    mulberry_core.data.unpack "Empty"[0] %value : (!tree_ref) -> ()
    return
  }

  func.func @node(%tree: !tree_ref) -> !tree_ref {
    %value, %left, %right = mulberry_core.data.unpack "Node"[1] %tree
        : (!tree_ref) -> (i64, !tree_ref, !tree_ref)
    return %left : !tree_ref
  }
}

// CHECK-LABEL: func.func @tag(
// CHECK-SAME: %[[VALUE:.*]]: !llvm.ptr) -> i64
// CHECK: %[[TAG:.*]] = llvm.load %[[VALUE]] : !llvm.ptr -> i64
// CHECK: return %[[TAG]] : i64

// CHECK-LABEL: func.func @by_id(
// CHECK-SAME: %[[VALUE:.*]]: !llvm.ptr) -> i64
// CHECK: %[[FIELD:.*]] = llvm.getelementptr %[[VALUE]][0, 1]
// CHECK-SAME: !llvm.struct<(i64, i64)>
// CHECK: %[[ID:.*]] = llvm.load %[[FIELD]] : !llvm.ptr -> i64
// CHECK: return %[[ID]] : i64

// CHECK-LABEL: func.func @by_name(
// CHECK-SAME: %[[VALUE:.*]]: !llvm.ptr) -> !llvm.ptr
// CHECK: llvm.getelementptr %[[VALUE]][0, 1]
// CHECK-SAME: !llvm.struct<(i64, ptr)>
// CHECK: llvm.load {{.*}} : !llvm.ptr -> !llvm.ptr

// CHECK-LABEL: func.func @empty(%{{.*}}: !llvm.ptr)
// CHECK-NEXT: return

// CHECK-LABEL: func.func @node(
// CHECK-SAME: %[[TREE:.*]]: !llvm.ptr) -> !llvm.ptr
// CHECK: llvm.getelementptr %[[TREE]][0, 1]
// CHECK-SAME: !llvm.struct<(i64, i64, ptr, ptr)>
// CHECK: llvm.getelementptr %[[TREE]][0, 2]
// CHECK-SAME: !llvm.struct<(i64, i64, ptr, ptr)>
// CHECK: llvm.getelementptr %[[TREE]][0, 3]
// CHECK-SAME: !llvm.struct<(i64, i64, ptr, ptr)>
// CHECK-NOT: mulberry_core.
