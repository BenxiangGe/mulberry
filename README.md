# Mulberry

Mulberry 是一个个人编译器学习项目。目标是把一个小型静态语言 lowering 到
MLIR，并逐步加入深度学习推理与训练能力。

项目基于 [Nicola Lancellotti 的 Cherry 项目](https://github.com/NicolaLancellotti/cherry)
继续开发。内部目录、namespace、dialect 和工具名目前仍大量保留 `cherry` 命名；
语言和项目方向已经逐步转向 Mulberry。

当前使用 `llvmorg-22.1.0`。

## 当前状态

Mulberry 目前已经具备一个可工作的前端和高层 MLIR pipeline：

- Lexer / Parser / AST / Sema / Driver。
- 结构化 semantic type system，不再使用旧的 string-based type system。
- 基础类型：`Unit`、`Bool`、`UInt64`、`Float32`。
- 复合类型：`struct` 和静态 shape 的 tensor，例如 `Float32[2, 3]`。
- `const` tensor 绑定检查。
- Struct literal 语法：`A { ... }`。
- Struct member read/write 使用独立 AST node，不再伪装成普通 call/binary expr。
- Tensor literal、tensor access，以及 `size(xs)` builtin。
- 普通语言结构 codegen 到 `func`、`arith`、`scf` 和高层 `mulberry` dialect。
- `cherry_nn` 深度学习 dialect，并支持 lowering 到 Linalg/Math/Arith/MemRef。
- `--dump=lowered-mlir` 可以把 Mulberry Tensor 和 `cherry_nn` ops lower 到
  storage-level MLIR，例如 `memref`、`linalg`、`math` 和 `llvm` dialect。
- LLVM/JIT/object file pipeline 暂时关闭，等待新的 Mulberry-to-LLVM lowering
  设计。

## Pipeline

```text
Mulberry source
  -> Lexer / Parser / AST
  -> Sema / semantic type checking
  -> MLIRGen
       - func / arith / scf: functions, scalar values, control flow
       - mulberry.record / mulberry.ptr: structs and addressable values
       - mulberry.tensor: writable tensor values
       - cherry_nn: neural-network operations
  -> optional lowering
       - mulberry.tensor -> memref
       - cherry_nn -> linalg / math / arith / memref
       - scalar and record storage -> LLVM dialect where currently supported
  -> LLVM/JIT/object pipeline is temporarily disabled
```

## 语言快照

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

## 神经网络 Builtin

内部 `cherry_nn` dialect 目前支持：

- `matmul`
- `matadd`
- `transpose`
- `exp`
- `sigmoid`
- `argmax`

这些 ops 会 lower 到标准 MLIR dialect，例如 `linalg`、`math`、`arith` 和
`memref`。

## 构建与测试

构建 LLVM 和 Mulberry：

```sh
make all
```

如果 LLVM 已经构建完成，只构建本项目：

```sh
make cherry-build
```

makefile 当前默认使用 `release` CMake preset。

常用 smoke tests：

```sh
./build/release/bin/cherry-driver --dump=mlir test/cherry/Language/structs.cherry
./build/release/bin/cherry-driver --dump=lowered-mlir test/cherry/Language/argmax.cherry
./build/release/bin/cherry-driver --dump=mlir examples/dl/inference_mnist1.cherry
```

直接运行 lit 测试：

```sh
cmake --build build/release --target check-cherry
```

注意：构建 target 和可执行文件名称仍然使用 `cherry`，例如 `cherry-driver`
和 `check-cherry`。

## MNIST 示例

仓库中包含一个生成出来的单样本 MNIST 推理示例，数据来自 Michael Nielsen 的
神经网络教程：

- 权重和 bias：`data/mnist-784-30-10.json`
- 测试数据：`data/mnist.pkl.gz`
- 生成脚本：`tools/generate_inference_mnist1.py`
- 生成源码：`examples/dl/inference_mnist1.cherry`

重新生成示例：

```sh
/usr/bin/python3 tools/generate_inference_mnist1.py
```

查看生成出的高层 MLIR：

```sh
./build/release/bin/cherry-driver --dump=mlir examples/dl/inference_mnist1.cherry
```

当执行能力重新启用时，`test_data[0]` 的期望预测结果是：

```text
7
```

## 已知限制

- 内部命名仍然大多是 `cherry`。
- 语言还没有标准库和 namespace 系统。
- 大模型数据仍然展开到源码 literal 里。
- End-to-end JIT 暂时关闭，等待 Mulberry lowering 重新设计完成。
- `cherry_nn` ops 还需要更强的 verifier 和诊断。
- 这是一个学习用编译器，不是生产编译器。

## 文档

- [Grammar](docs/Grammar.md)
- [Builtins](docs/Builtins.md)
- [Mulberry Lowering](docs/MulberryLowering.md)
- Driver 示例：`test/cherry/Driver`
