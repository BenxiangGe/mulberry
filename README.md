# Mulberry

Mulberry 是一个个人编译器学习项目。目标是把一个小型静态语言 lowering 到
MLIR，并逐步加入深度学习推理与训练能力。

项目基于 [Nicola Lancellotti 的 Cherry 项目](https://github.com/NicolaLancellotti/cherry)
继续开发。当前项目、工具、namespace 和核心 IR 命名已经迁移到 Mulberry；
源码文件扩展名暂时仍使用 `.mulberry`。

当前使用 `llvmorg-22.1.0`。

## 当前状态

Mulberry 目前已经具备一个可工作的前端和高层 MLIR pipeline：

- Lexer / Parser / AST / Sema / Driver。
- 结构化 semantic type system，不再使用旧的 string-based type system。
- 基础类型：`Unit`、`Bool`、`UInt64`、`Float32`。
- 复合类型：`struct`、`Array<T, N>`、`List<T>` 和 `Tensor<T>` object。
- 下一阶段的类型分层已经确定：普通 `[]` literal 应默认是语言 `Array`；
  `List<T>` 是 growable container；`Tensor<T>` 类似 NumPy `ndarray`，只在显式需要
  dense numeric buffer、memref/linalg 或 `mulberry.nn` interop 时使用。
- 当前一维静态 `T[N]` 已是 fixed-size Array；`Array<T, N>` 现在是
  `{ length, data }` 风格的普通 object，元素 buffer 由 heap 分配，支持函数
  参数/返回、`struct` field 和 nested Array。底层 storage 是 compiler/stdlib
  内部实现细节，不是用户语法。多维/动态 `T[...]`
  源码语法已经删除；ndarray-style value 统一写 `Tensor<T>`。
- Struct literal 语法：`A { ... }`。
- Struct member read/write 使用独立 AST node，不再伪装成普通 call/binary expr。
- `Tensor<T>` header、`tensor.zeros([shape])`、`tensor.from(array)`、直接
  `tensor[i, j]` 元素访问，以及 `Tensor<T>.numel()` method；普通数组转 Tensor
  已有显式入口。
- 普通语言结构 codegen 到 `func`、`arith`、`scf` 和高层 `mulberry` dialect。
- `mulberry.nn` 已经作为独立 NN package 从 core 拆出，相关 lit 测试已经删除或
  改写成 core Tensor/List 正向测试。
- `--dump=lowered-mlir` 可以把当前 core Tensor 和对象模型 lower 到 storage-level
  MLIR，例如 `memref`、`arith` 和 `llvm` dialect。
- `--dump=mlir-llvm`、`--dump=llvm`、JIT、object file 和 executable pipeline
  已经对当前正向子集打开。

## Pipeline

```text
Mulberry source
  -> Lexer / Parser / AST
  -> Sema / semantic type checking
  -> MLIRGen / high-level MLIR
  -> optional storage lowering
  -> LLVM dialect / LLVM IR / JIT / object file / executable
```

用户层只需要理解 Mulberry source、stdlib API、`mulberry.nn` package 和 driver
输出层级。compiler-owned core dialect、storage object layout 和 lowering ABI 都是
实现细节，集中记录在 [Mulberry Lowering](docs/MulberryLowering.md)。

## 语言快照

```mulberry
struct Point {
  x: UInt64,
  y: UInt64
}

fn main(): UInt64 {
  var p: Point = Point { 10, 20 };
  var xs: UInt64[3] = [1, 2, 3];

  xs[1] = 40;
  p.x = 3;
  return p.x;
}
```

## 神经网络 Package

`mulberry.nn` package 目前是独立 NN package 的源码入口。NN dialect/backend
位于 `packages/nn`；`stdlib/mulberry/nn.mulberry` 只保留普通 extern package
function 声明。下面这些 primitives 不再由 core Sema/MLIRGen/LowerMulberry
硬编码处理：

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

当前 `packages/nn` 已经能编译出 `MulberryNNPackage`。它提供
`prepare-mulberry-nn-calls` 和 `lower-mulberry-nn` 两段 pass：前者把
source-level `nn.matmul(...)` 这类普通 package call 桥到 private tensor-form
call，并在 core `lower-mulberry` 之后改写成 `mulberry_nn.*` op；后者把
`mulberry_nn.*` op lower 到 `linalg` / `arith` / `math`。

driver 通过 bundled package registry 检测 `import mulberry.nn`，自动加载
`MulberryNNPackage`，并把 package pipeline 接入默认 lowering/JIT 路径。这里
只注册 package 边界和 pipeline，不把 NN primitive 重新塞回 core compiler。
`mulberry.nn` inference 示例可以直接用普通 `mulberry-driver file.mulberry` 运行。

## 构建与测试

构建 LLVM 和 Mulberry：

```sh
make all
```

如果 LLVM 已经构建完成，只构建本项目：

```sh
make mulberry-build
```

makefile 当前默认使用 `release` CMake preset。

常用 smoke tests：

```sh
./build/release/bin/mulberry-driver --dump=mlir test/mulberry/Language/structs.mulberry
./build/release/bin/mulberry-driver --dump=llvm test/mulberry/Driver/driver.mulberry
./build/release/bin/mulberry-driver -c=/tmp/driver.o test/mulberry/Driver/driver.mulberry
./build/release/bin/mulberry-driver -o /tmp/driver test/mulberry/Driver/driver.mulberry
```

`-c=<file>` 只生成 native object file。`-o <file>` 会先生成临时 object file，
再调用系统 `clang`/`cc` 链接 executable。当前 executable 会链接构建树里的
Mulberry runtime、MLIR runner utils 和源码编译出的 Boehm GC，并写入对应
runtime rpath。

如果需要把 runtime `.so` 放到 executable 同目录，可以使用
`--bundle-runtime`：

```sh
./build/release/bin/mulberry-driver -o /tmp/driver --bundle-runtime test/mulberry/Driver/driver.mulberry
```

这个模式会把当前需要的 runtime shared libraries 复制到输出文件所在目录，并
使用 `$ORIGIN` rpath。

直接运行 lit 测试：

```sh
cmake --build build/release --target check-mulberry
```

## MNIST 示例

仓库中包含一个生成出来的单样本 MNIST 推理示例，数据来自 Michael Nielsen 的
神经网络教程：

- 权重和 bias：`data/mnist-784-30-10.json`
- 测试数据：`data/mnist.pkl.gz`
- 生成脚本：`tools/generate_inference_mnist1.py`
- raw tensor 导出脚本：`tools/export_mnist_raw_tensors.py`
- safetensors 导出脚本：`tools/export_mnist_safetensors.py`
- training safetensors 导出脚本：`tools/export_mnist_training_safetensors.py`
- 生成源码：`examples/dl/inference_mnist1.mulberry`
- safetensors 文件推理源码：`examples/dl/inference_mnist_safetensors.mulberry`
- raw 文件推理源码：`examples/dl/inference_mnist_raw.mulberry`
- safetensors training smoke 源码：`examples/dl/training_mnist_safetensors.mulberry`
- safetensors mini-batch training smoke 源码：
  `examples/dl/training_mnist_minibatch_safetensors.mulberry`

重新生成示例：

```sh
/usr/bin/python3 tools/generate_inference_mnist1.py
```

导出当前最小 raw tensor 文件：

```sh
python3 tools/export_mnist_raw_tensors.py
```

默认输出目录是 `data/mnist-784-30-10-raw/`。文件只包含连续 element bytes，type
和 shape 由 Mulberry 源码里的 tensor 类型决定。详细约定见
[Raw Tensor Files](docs/RawTensorFiles.md)。

raw `.f32` 是 bootstrap/debug 格式。日常 MNIST 推理优先使用 safetensors：它用
单个文件保存多个 tensor，并通过 `safetensors.open(path)` / `safetensors.read(file, name)`
按名字读取 Tensor header。
详细约定见 [Safetensors](docs/Safetensors.md)。

导出 safetensors 单文件：

```sh
python3 tools/export_mnist_safetensors.py
```

这些导出脚本和示例现在作为外部 NN package 的回归素材保留。MNIST inference
路径已经重新接入默认 JIT smoke：

```sh
./build/release/bin/mulberry-driver examples/dl/inference_mnist_safetensors.mulberry
```

也可以直接生成 native executable：

```sh
./build/release/bin/mulberry-driver examples/dl/inference_mnist_safetensors.mulberry -o /tmp/mnist
/tmp/mnist
```

导出一个小的 training safetensors 子集：

```sh
python3 tools/export_mnist_training_safetensors.py
```

当前 training 导出是 batch tensor 布局：`train_x` / `test_x` 的 shape 是
`[N, 784, 1]`，`train_y` / `test_y` 的 shape 是 `[N, 10, 1]`。training script
先用 `safetensors.TensorFile` cache 读取整个 batch，再通过 `nn.TensorDataset`
把 batch tensor 转成成对的样本 view；这样暂时不需要先引入完整 dataset iterator。

training smoke 走 Nielsen `network2.py` 默认的 CrossEntropy output delta：
`delta = a - y`，用默认导出的 `10` 个 training 样本跑 `30` 个 epoch。训练后会读取
`data/mnist-784-30-10.safetensors` 里的 `x` 做一次 inference。training 示例会先输出
bootstrap training/test 的 `correct`、`total` 和 `accuracyBasisPoints`，最后输出
inference label，当前期望是 `10`、`10`、`10000`、`9`、`10`、`9000` 和 `7`。
per-sample 和 mini-batch 示例当前复用 `mulberry.nn` 里的 `nn.twoLayerGradient()`，
把 2-layer FCN 的 backprop 从示例代码里收进 stdlib helper；它仍然只是当前
Nielsen 784-30-10 正向路径需要的普通 helper，不是通用 autograd。
当前已接入固定小 lambda 的 L2 regularization 正向路径，并会把训练后的权重写成一个
safetensors checkpoint 再读回验证。per-sample 和 mini-batch 示例都会在每个 epoch
前调用 `TensorDataset.shuffle()` 做 deterministic paired shuffle。mini-batch 示例从
`dataset.size()` 和 `batchSize` 派生 batch 数，最后一个不满 batch 也会按实际大小更新；
lit 里有一个 `--sample-count 11` 的 safetensors tail-batch 回归专门覆盖这条路径。
`TensorDataset` 只是当前 `Tensor<Float32>` input/label pair 的轻量视图，完整 dataset
iterator 仍是后续任务。

```sh
./build/release/bin/mulberry-driver examples/dl/training_mnist_safetensors.mulberry
./build/release/bin/mulberry-driver examples/dl/training_mnist_minibatch_safetensors.mulberry
./build/release/bin/mulberry-driver --dump=mlir examples/dl/inference_mnist1.mulberry
./build/release/bin/mulberry-driver --dump=lowered-mlir examples/dl/inference_mnist1.mulberry
./build/release/bin/mulberry-driver examples/dl/inference_mnist1.mulberry
./build/release/bin/mulberry-driver examples/dl/inference_mnist_raw.mulberry
```

`test_data[0]` 的期望预测结果是：

```text
7
```

## 已知限制

- 源码文件扩展名仍然是 `.mulberry`，后续是否改为 `.mulberry` 需要单独决定。
- 标准库、namespace 和 import 仍然是早期实现。
- `examples/dl/inference_mnist1.mulberry` 仍然把数据展开到源码 literal 里；
  safetensors 和 raw 文件版本已经可以从外部 tensor 文件读取数据。
- JIT 只覆盖当前正向子集；object file generation 仍然关闭。
- `mulberry.nn` primitives 已经从 core 拆出；当前已有独立 NN package，driver 通过
  bundled package registry 按需加载 package plugin 并接入 NN lowering pipeline。
- 这是一个学习用编译器，不是生产编译器。

## 文档

- [Grammar](docs/Grammar.md)
- [Builtins](docs/Builtins.md)
- [Mulberry Lowering](docs/MulberryLowering.md)
- [Mulberry NN External Package](docs/MulberryNNExternal.md)
- [Raw Tensor Files](docs/RawTensorFiles.md)
- [Safetensors](docs/Safetensors.md)
- Driver 示例：`test/mulberry/Driver`
