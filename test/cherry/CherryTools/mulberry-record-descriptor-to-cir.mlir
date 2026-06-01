// RUN: cherry-opt --convert-mulberry-record-to-cir %s | FileCheck %s

module {
  cir.func @tensor_descriptor() {
    %descriptor = mulberry.alloca !mulberry.record<TensorDescriptor {data: !mulberry.ptr<!cir.int<u, 64>>, shape: !mulberry.record<TensorShape2D {rows: !cir.int<u, 64>, cols: !cir.int<u, 64>}>}> : !mulberry.ptr<!mulberry.record<TensorDescriptor {data: !mulberry.ptr<!cir.int<u, 64>>, shape: !mulberry.record<TensorShape2D {rows: !cir.int<u, 64>, cols: !cir.int<u, 64>}>}>>
    %data = mulberry.record.get_field %descriptor["data"] : !mulberry.ptr<!mulberry.record<TensorDescriptor {data: !mulberry.ptr<!cir.int<u, 64>>, shape: !mulberry.record<TensorShape2D {rows: !cir.int<u, 64>, cols: !cir.int<u, 64>}>}>> -> !mulberry.ptr<!mulberry.ptr<!cir.int<u, 64>>>
    %shape = mulberry.record.get_field %descriptor["shape"] : !mulberry.ptr<!mulberry.record<TensorDescriptor {data: !mulberry.ptr<!cir.int<u, 64>>, shape: !mulberry.record<TensorShape2D {rows: !cir.int<u, 64>, cols: !cir.int<u, 64>}>}>> -> !mulberry.ptr<!mulberry.record<TensorShape2D {rows: !cir.int<u, 64>, cols: !cir.int<u, 64>}>>
    %rows = mulberry.record.get_field %shape["rows"] : !mulberry.ptr<!mulberry.record<TensorShape2D {rows: !cir.int<u, 64>, cols: !cir.int<u, 64>}>> -> !mulberry.ptr<!cir.int<u, 64>>
    cir.return
  }
}

// CHECK: !rec_TensorShape2D = !cir.record<struct "TensorShape2D" {!u64i, !u64i}>
// CHECK: !rec_TensorDescriptor = !cir.record<struct "TensorDescriptor" {!cir.ptr<!u64i>, !rec_TensorShape2D}>
// CHECK-LABEL: cir.func @tensor_descriptor()
// CHECK: cir.alloca !rec_TensorDescriptor, !cir.ptr<!rec_TensorDescriptor>
// CHECK: cir.get_member {{.*}}[0] {name = "data"} : !cir.ptr<!rec_TensorDescriptor> -> !cir.ptr<!cir.ptr<!u64i>>
// CHECK: cir.get_member {{.*}}[1] {name = "shape"} : !cir.ptr<!rec_TensorDescriptor> -> !cir.ptr<!rec_TensorShape2D>
// CHECK: cir.get_member {{.*}}[0] {name = "rows"} : !cir.ptr<!rec_TensorShape2D> -> !cir.ptr<!u64i>
// CHECK-NOT: mulberry.
