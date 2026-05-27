# Mulberry

Mulberry 是一个个人编译器学习项目。目标是把一个小型静态语言 lowering 到
MLIR、Clang CIR、Linalg/MemRef 和 LLVM dialect，并逐步加入深度学习推理与训练能力。

项目基于 [Nicola Lancellotti 的 Cherry 项目](https://github.com/NicolaLancellotti/cherry)
继续开发。内部目录、namespace、dialect 和工具名目前仍大量保留 `cherry` 命名；
语言和项目方向已经逐步转向 Mulberry。

当前使用 `llvmorg-22.1.0`，并开启 ClangIR/CIR 支持。

## Current Status

Mulberry 目前已经具备一个可工作的端到端 MLIR compiler pipeline：

- Lexer / Parser / AST / Sema / Driver。
- 结构化 semantic type system，不再使用旧的 string-based type system。
- 基础类型：`Unit`、`Bool`、`UInt64`、`Float32`。
- 复合类型：`struct` 和静态 shape 的 tensor，例如 `Float32[2, 3]`。
- `const` tensor 绑定检查。
- Struct literal 语法：`A { ... }`。
- Struct member read/write 使用独立 AST node，不再伪装成普通 call/binary expr。
- Tensor literal、tensor access，以及 `size(xs)` builtin。
- 普通语言结构 lowering 到 Clang CIR、SCF、arith、cf、func、memref。
- `cherry_nn` 深度学习 dialect，并支持 lowering 到 Linalg/Math/Arith/MemRef。
- 通过 MLIR LLVM dialect 支持 x64 Linux JIT 和 object file 输出。

## Pipeline

```text
Mulberry source
  -> Lexer / Parser / AST
  -> Sema / semantic type checking
  -> MLIRGen
       - Clang CIR: scalar values, functions, control flow, structs
       - memref: tensor storage
       - cherry_nn: neural-network operations
  -> optional lowering
       - cherry_nn -> linalg / math / arith / memref
       - CIR / SCF / linalg -> LLVM dialect
  -> JIT execution or object file
```

## Language Snapshot

```cherry
struct Point {
  x: UInt64,
  y: UInt64
}

fn main(): UInt64 {
  var p: Point = Point { 10, 20 };
  var xs: UInt64[3] = [1, 2, 3];

  p.x = size(xs);
  p.x
}
```

## Neural Network Builtins

The internal `cherry_nn` dialect currently supports:

- `matmul`
- `matadd`
- `transpose`
- `exp`
- `sigmoid`
- `argmax`

These ops lower to standard MLIR dialects such as `linalg`, `math`, `arith`,
and `memref`.

## Build And Test

Build LLVM and Mulberry:

```sh
make all
```

If LLVM is already built, build only this project:

```sh
make cherry-build
```

The makefile currently defaults to the `release` CMake preset.

Common smoke tests:

```sh
./build/release/bin/cherry-driver --dump=mlir test/cherry/Language/structs.cherry
./build/release/bin/cherry-driver --dump=linalg test/cherry/Language/argmax.cherry
./build/release/bin/cherry-driver examples/dl/inference_mnist1.cherry
```

Run lit tests directly:

```sh
cmake --build build/release --target check-cherry
```

Note: build targets and executable names still use `cherry`, for example
`cherry-driver` and `check-cherry`.

## MNIST Example

The repository contains a generated single-sample MNIST inference example based
on Michael Nielsen's neural network tutorial data:

- Weights and bias: `data/mnist-784-30-10.json`
- Test data: `data/mnist.pkl.gz`
- Generator: `tools/generate_inference_mnist1.py`
- Generated source: `examples/dl/inference_mnist1.cherry`

Regenerate the example:

```sh
/usr/bin/python3 tools/generate_inference_mnist1.py
```

Run it:

```sh
./build/release/bin/cherry-driver examples/dl/inference_mnist1.cherry
```

Expected result for `test_data[0]`:

```text
7
```

## Known Limitations

- Internal naming is still mostly `cherry`.
- The language has no standard library or namespace system yet.
- Large model data is still expanded into source literals.
- `cherry_nn` ops still need stronger verifiers and diagnostics.
- This is a learning compiler, not a production compiler.

## Docs

- [Grammar](docs/Grammar.md)
- [Builtins](docs/Builtins.md)
- Driver examples: `test/cherry/Driver`
