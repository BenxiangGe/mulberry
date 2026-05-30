// RUN: cherry-opt --verify-diagnostics %s

func.func @list_create_element_type_mismatch(
    %arg0: memref<2xf32>,
    %arg1: memref<3xf32>) {
  // expected-error@+1 {{'mulberry.list.create' op element type does not match list element type}}
  %0 = mulberry.list.create %arg0, %arg1
      : (memref<2xf32>, memref<3xf32>) -> !mulberry.list<memref<2xf32>>
  return
}

func.func @list_get_result_type_mismatch(
    %arg0: !mulberry.list<memref<2xf32>>,
    %index: index) {
  // expected-error@+1 {{'mulberry.list.get' op result type does not match list element type}}
  %0 = mulberry.list.get %arg0[%index]
      : <memref<2xf32>> -> memref<3xf32>
  return
}
