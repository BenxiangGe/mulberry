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
- `--dump=mlir-llvm`、`--dump=llvm` 和 JIT 已经对当前正向子集打开。
- object file pipeline 暂时关闭，等待后续单独设计。

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
  -> LLVM dialect / LLVM IR / JIT for the currently supported positive path
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
- `matsub`
- `hadamard`
- `matscale`
- `transpose`
- `exp`
- `sigmoid`
- `sigmoidPrime`
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
./build/release/bin/cherry-driver --dump=lowered-mlir examples/dl/inference_mnist1.cherry
./build/release/bin/cherry-driver --dump=llvm test/cherry/Driver/driver.cherry
./build/release/bin/cherry-driver examples/dl/inference_mnist1.cherry
./build/release/bin/cherry-driver examples/dl/inference_mnist_safetensors.cherry
./build/release/bin/cherry-driver examples/dl/inference_mnist_raw.cherry
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
- raw tensor 导出脚本：`tools/export_mnist_raw_tensors.py`
- safetensors 导出脚本：`tools/export_mnist_safetensors.py`
- training safetensors 导出脚本：`tools/export_mnist_training_safetensors.py`
- 生成源码：`examples/dl/inference_mnist1.cherry`
- safetensors 文件推理源码：`examples/dl/inference_mnist_safetensors.cherry`
- raw 文件推理源码：`examples/dl/inference_mnist_raw.cherry`
- safetensors training smoke 源码：`examples/dl/training_mnist_safetensors.cherry`

重新生成示例：

```sh
/usr/bin/python3 tools/generate_inference_mnist1.py
```

导出当前最小 raw tensor 文件：

```sh
python3 tools/export_mnist_raw_tensors.py
```

默认输出目录是 `data/mnist-784-30-10-raw/`。文件只包含连续 element bytes，type
和 shape 由 Cherry 源码里的 tensor 类型决定。详细约定见
[Raw Tensor Files](docs/RawTensorFiles.md)。

raw `.f32` 是 bootstrap/debug 格式。日常 MNIST 推理优先使用 safetensors：它用
单个文件保存多个 tensor，并通过 expected-type `readTensor(file, name)` 读取。
详细约定见 [Safetensors](docs/Safetensors.md)。

导出 safetensors 单文件：

```sh
python3 tools/export_mnist_safetensors.py
```

导出后可以直接运行 safetensors 文件版本：

```sh
./build/release/bin/cherry-driver examples/dl/inference_mnist_safetensors.cherry
```

导出一个小的 training safetensors 子集：

```sh
python3 tools/export_mnist_training_safetensors.py
```

当前 training 导出是 bootstrap 布局：每个样本独立保存为
`train_x_0`、`train_y_0`、...、`train_x_9`、`train_y_9` 这样的 named tensor。这样后续
training script 可以继续使用已经跑通的 `readTensor(file, name)`，不需要先引入
dataset iterator 或 tensor slice。

导出后可以运行一个最小真实数据 training smoke：

```sh
./build/release/bin/cherry-driver examples/dl/training_mnist_safetensors.cherry
```

当前 training smoke 走 Nielsen `network2.py` 默认的 CrossEntropy output delta：
`delta = a - y`，用默认导出的 `10` 个 training 样本跑 `30` 个 epoch。训练后会读取
`data/mnist-784-30-10.safetensors` 里的 `x` 做一次 inference，期望输出是 `7`。
mini-batch、shuffle、L2 regularization 和保存训练结果还没实现。

查看生成出的高层 MLIR：

```sh
./build/release/bin/cherry-driver --dump=mlir examples/dl/inference_mnist1.cherry
```

查看 lowering 后的 MLIR：

```sh
./build/release/bin/cherry-driver --dump=lowered-mlir examples/dl/inference_mnist1.cherry
```

当前 lowered IR 会把 for-loop 推理降到 `scf`、`memref`、`linalg`、`arith` 和
`math`，不应再残留 `mulberry` 或 `cherry_nn` op。

直接执行 JIT：

```sh
./build/release/bin/cherry-driver examples/dl/inference_mnist1.cherry
```

如果已经导出 raw tensor 文件，也可以运行不包含巨大 literal 的 raw 文件版本：

```sh
./build/release/bin/cherry-driver examples/dl/inference_mnist_raw.cherry
```

`test_data[0]` 的期望预测结果是：

```text
7
```

## 已知限制

- 内部命名仍然大多是 `cherry`。
- 语言还没有标准库和 namespace 系统。
- `examples/dl/inference_mnist1.cherry` 仍然把数据展开到源码 literal 里；
  safetensors 和 raw 文件版本已经可以从外部 tensor 文件读取数据。
- JIT 只覆盖当前正向子集；object file generation 仍然关闭。
- `cherry_nn` ops 还需要更强的 verifier 和诊断。
- 这是一个学习用编译器，不是生产编译器。

## 文档

- [Grammar](docs/Grammar.md)
- [Builtins](docs/Builtins.md)
- [Mulberry Lowering](docs/MulberryLowering.md)
- [Raw Tensor Files](docs/RawTensorFiles.md)
- [Safetensors](docs/Safetensors.md)
- Driver 示例：`test/cherry/Driver`
