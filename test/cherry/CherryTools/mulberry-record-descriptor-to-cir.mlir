// RUN: cherry-opt --convert-mulberry-record-to-cir %s | FileCheck %s

module {
  cir.func @tensor_descriptor() {
    %descriptor = mulberry.alloca !mulberry.record<TensorDescriptor {allocated: !mulberry.ptr<!cir.int<u, 64>>, aligned: !mulberry.ptr<!cir.int<u, 64>>, offset: !cir.int<u, 64>, sizes: !mulberry.record<TensorSizes2D {rows: !cir.int<u, 64>, cols: !cir.int<u, 64>}>, strides: !mulberry.record<TensorStrides2D {rows: !cir.int<u, 64>, cols: !cir.int<u, 64>}>}> : !mulberry.ptr<!mulberry.record<TensorDescriptor {allocated: !mulberry.ptr<!cir.int<u, 64>>, aligned: !mulberry.ptr<!cir.int<u, 64>>, offset: !cir.int<u, 64>, sizes: !mulberry.record<TensorSizes2D {rows: !cir.int<u, 64>, cols: !cir.int<u, 64>}>, strides: !mulberry.record<TensorStrides2D {rows: !cir.int<u, 64>, cols: !cir.int<u, 64>}>}>>
    %allocated = mulberry.record.get_field %descriptor["allocated"] : !mulberry.ptr<!mulberry.record<TensorDescriptor {allocated: !mulberry.ptr<!cir.int<u, 64>>, aligned: !mulberry.ptr<!cir.int<u, 64>>, offset: !cir.int<u, 64>, sizes: !mulberry.record<TensorSizes2D {rows: !cir.int<u, 64>, cols: !cir.int<u, 64>}>, strides: !mulberry.record<TensorStrides2D {rows: !cir.int<u, 64>, cols: !cir.int<u, 64>}>}>> -> !mulberry.ptr<!mulberry.ptr<!cir.int<u, 64>>>
    %sizes = mulberry.record.get_field %descriptor["sizes"] : !mulberry.ptr<!mulberry.record<TensorDescriptor {allocated: !mulberry.ptr<!cir.int<u, 64>>, aligned: !mulberry.ptr<!cir.int<u, 64>>, offset: !cir.int<u, 64>, sizes: !mulberry.record<TensorSizes2D {rows: !cir.int<u, 64>, cols: !cir.int<u, 64>}>, strides: !mulberry.record<TensorStrides2D {rows: !cir.int<u, 64>, cols: !cir.int<u, 64>}>}>> -> !mulberry.ptr<!mulberry.record<TensorSizes2D {rows: !cir.int<u, 64>, cols: !cir.int<u, 64>}>>
    %rows = mulberry.record.get_field %sizes["rows"] : !mulberry.ptr<!mulberry.record<TensorSizes2D {rows: !cir.int<u, 64>, cols: !cir.int<u, 64>}>> -> !mulberry.ptr<!cir.int<u, 64>>
    cir.return
  }
}

// CHECK: !rec_TensorSizes2D = !cir.record<struct "TensorSizes2D" {!u64i, !u64i}>
// CHECK: !rec_TensorStrides2D = !cir.record<struct "TensorStrides2D" {!u64i, !u64i}>
// CHECK: !rec_TensorDescriptor = !cir.record<struct "TensorDescriptor" {!cir.ptr<!u64i>, !cir.ptr<!u64i>, !u64i, !rec_TensorSizes2D, !rec_TensorStrides2D}>
// CHECK-LABEL: cir.func @tensor_descriptor()
// CHECK: cir.alloca !rec_TensorDescriptor, !cir.ptr<!rec_TensorDescriptor>
// CHECK: cir.get_member {{.*}}[0] {name = "allocated"} : !cir.ptr<!rec_TensorDescriptor> -> !cir.ptr<!cir.ptr<!u64i>>
// CHECK: cir.get_member {{.*}}[3] {name = "sizes"} : !cir.ptr<!rec_TensorDescriptor> -> !cir.ptr<!rec_TensorSizes2D>
// CHECK: cir.get_member {{.*}}[0] {name = "rows"} : !cir.ptr<!rec_TensorSizes2D> -> !cir.ptr<!u64i>
// CHECK-NOT: mulberry.
