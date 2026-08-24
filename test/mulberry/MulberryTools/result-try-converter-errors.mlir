// RUN: not mulberry-opt --split-input-file %s 2>&1 | FileCheck %s

module {
  func.func @missing_converter(
      %input: !mulberry_core.ptr<!mulberry_core.data<"Result">>)
      -> !mulberry_core.ptr<!mulberry_core.data<"Result">> {
    mulberry_core.result.try %input
        source_error !mulberry_core.ptr<!mulberry_core.data<"SourceError">>
        target_error !mulberry_core.ptr<!mulberry_core.data<"TargetError">>
        : (!mulberry_core.ptr<!mulberry_core.data<"Result">>) -> ()
    return %input : !mulberry_core.ptr<!mulberry_core.data<"Result">>
  }
}

// -----

module {
  func.func private @wrong_converter(
      !mulberry_core.ptr<!mulberry_core.data<"OtherError">>)
      -> !mulberry_core.ptr<!mulberry_core.data<"TargetError">>

  func.func @wrong_converter_signature(
      %input: !mulberry_core.ptr<!mulberry_core.data<"Result">>)
      -> !mulberry_core.ptr<!mulberry_core.data<"Result">> {
    mulberry_core.result.try %input
        source_error !mulberry_core.ptr<!mulberry_core.data<"SourceError">>
        target_error !mulberry_core.ptr<!mulberry_core.data<"TargetError">>
        converter @wrong_converter
        : (!mulberry_core.ptr<!mulberry_core.data<"Result">>) -> ()
    return %input : !mulberry_core.ptr<!mulberry_core.data<"Result">>
  }
}

// CHECK: error: 'mulberry_core.result.try' op different source and target error types require a converter
// CHECK: error: 'mulberry_core.result.try' op converter must have signature source error type -> target error type
