# Mulberry

Mulberry 是一个个人编译器学习项目。目标是实现一门小型静态类型语言，
把 source lowering 到 MLIR / LLVM，并逐步加入数值计算、深度学习推理和训练能力。

项目基于 [Nicola Lancellotti 的 Cherry 项目](https://github.com/NicolaLancellotti/cherry)
继续开发。当前项目名、工具名、namespace、核心 IR 和源码扩展名都已经迁移到
Mulberry；用户程序使用 `.mulberry`。

当前使用 `llvmorg-22.1.0`。

## 当前状态

Mulberry 目前已经具备一条可工作的 frontend 到 MLIR / LLVM pipeline：

- Frontend：Lexer、Parser、AST、Sema、Driver。
- 类型系统：结构化 semantic type system，不再使用早期 string-based type system。
- 基础类型：`Unit`、`Bool`、`UInt8`、`UInt64`、`Float32`、`Char`。
- Source object：`String`、`File`、user `struct`、`Array<T, N>`、`List<T>`、
  `Tensor<T>`。
- 对象模型：scalar 按 value 传递；非 scalar object 按 reference 传递，由 GC
  管理内部对象生命周期。
- 可变性模型：函数参数和 method receiver 默认 readonly；需要修改 object 时显式写
  `mut value: T` 或 `mut self: T`。`const` 只用于 local binding。
- 指针模型：`Ptr<T>` 不再是普通用户 surface，只保留在 stdlib/compiler/runtime
  内部，用于对象布局、FFI/runtime handle 和 lowering。
- 泛型/编译期能力：支持当前 stdlib 需要的 `comptime` 类型别名和泛型函数实例化，
  例如 `List<T>`、`Tensor<T>`、`Array<T, N>`。
- 控制流：`if` statement、`while`、`break` / `continue`、`for i in start .. end`。
- Module surface：`package`、`import std.foo`、`import mulberry.nn`、package alias。

## 数据模型

Mulberry 当前把语言数组、动态容器和数值 Tensor 分开处理：

- `Array<T, N>` 是普通语言数组，一维静态 `T[N]` 是它的语法糖。
- `[]` literal 默认构造 `Array`。如果要得到 `List<T>` 或 `Tensor<T>`，需要显式调用
  `list.from(...)` 或 `tensor.from(...)`。
- `List<T>` 是 growable container，定义在 `stdlib/std/list.mulberry`。
- `Tensor<T>` 是类似 NumPy `ndarray` 的数值 buffer object，定义在
  `stdlib/std/tensor.mulberry`。它包含 runtime `rank`、`numel`、`sizes` 和 `strides`
  元数据。
- 多维 / 动态 `Float32[?, ?]` 这类旧 source spelling 已删除；数值 buffer 统一写
  `Tensor<T>`。

示例：

```mulberry
struct Point {
  x: UInt64,
  y: UInt64
}

fn main(): UInt64 {
  var p: Point = Point { 10, 20 };
  var xs: UInt64[3] = [1, 2, 3];
  var values: List<UInt64> = list.from([10, 20, 30]);
  var matrix: Tensor<Float32> = tensor.from([[1.0, 2.0], [3.0, 4.0]]);

  xs[1] = 40;
  values.push(40);
  io.print(matrix.numel());

  return p.x + xs[1] + values.size();
}
```

## Pipeline

```text
Mulberry source
  -> Lexer / Parser / AST
  -> Sema / semantic type checking
  -> MLIRGen / high-level MLIR
  -> Mulberry core lowering
  -> package lowering, e.g. mulberry.nn
  -> LLVM dialect / LLVM IR
  -> JIT / object file / executable
```

Core compiler 仍然有 compiler-owned `mulberry_core` dialect，用于 record、ptr、heap
object、Tensor view/pack 等 lowering bridge。这是内部 IR，不是用户 surface。

## 标准库

当前 stdlib 主要放在 `stdlib/std`：

- `core`：基础转换函数，例如 `toUInt64()`、`toUInt8()`、`toFloat32()`。
- `io`：`print()`、`File`、`open()` / `close()`、Tensor byte read/write。
- `string`：source-level `String` object。
- `list`：`List<T>`、`withCapacity()`、`from()`、`push()`、`size()`、`shufflePair()`。
- `tensor`：`Tensor<T>`、`zeros()`、`from()`、`sliceFirst()`、`slicesFirst()`。
- `json`：cursor-style JSON parser，用于 safetensors metadata。
- `safetensors`：safetensors open/read/write/checkpoint helper。
- `random`：当前训练示例使用的 deterministic LCG。

`stdlib/prelude.mulberry` 默认 import 常用 stdlib package 和类型，所以普通用户程序可以
直接写 `List<T>`、`Tensor<T>`、`String`、`File`、`io.print(...)`、`tensor.zeros(...)` 等。

## 神经网络 Package

`mulberry.nn` 已经从 core compiler 中拆出，作为 bundled package 存在：

- source 入口：`stdlib/mulberry/nn.mulberry`
- dialect / pass / lowering：`packages/nn`
- package library：`MulberryNNPackage`

`stdlib/mulberry/nn.mulberry` 只声明普通 extern package function 和少量 helper。
Sema、MLIRGen 和 core `LowerMulberry` 不再硬编码 NN primitive。

当前支持的 NN primitive：

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

driver 检测到 `import mulberry.nn` 后，会通过 bundled package registry 加载
`MulberryNNPackage`，并把 NN package pipeline 接入默认 lowering / JIT / executable
路径。当前 NN lowering 主要覆盖 Nielsen MNIST FCN 推理和训练需要的 rank-2 Tensor
路径；CNN、ReLU、Softmax、conv2d、max-pool 和通用 autograd 还没有实现。

## 构建与测试

构建 LLVM 和 Mulberry：

```sh
make all
```

如果 LLVM 已经构建完成，只构建并测试本项目：

```sh
make mulberry-build
```

makefile 当前默认使用 `release` CMake preset。

常用 smoke tests：

```sh
./build/release/bin/mulberry-driver --dump=mlir test/mulberry/Language/structs.mulberry
./build/release/bin/mulberry-driver --dump=lowered-mlir test/mulberry/Language/mulberry_nn_tensor_bridge.mulberry
./build/release/bin/mulberry-driver --dump=llvm test/mulberry/Driver/driver.mulberry
./build/release/bin/mulberry-driver -c=/tmp/driver.o test/mulberry/Driver/driver.mulberry
./build/release/bin/mulberry-driver -o /tmp/driver test/mulberry/Driver/driver.mulberry
```

`-c=<file>` 生成 native object file。`-o <file>` 会先生成临时 object file，再调用系统
`clang`/`cc` 链接 executable。executable 会链接构建树里的 Mulberry runtime、MLIR
runner utils 和 Boehm GC。

如果需要把 runtime `.so` 放到 executable 同目录，可以使用 `--bundle-runtime`：

```sh
./build/release/bin/mulberry-driver -o /tmp/driver --bundle-runtime test/mulberry/Driver/driver.mulberry
```

直接运行 lit 测试：

```sh
cmake --build build/release --target check-mulberry
```

## MNIST 示例

仓库包含 Michael Nielsen MNIST 784-30-10 网络的推理和训练 smoke：

- 展开到源码 literal 的推理示例：`examples/dl/inference_mnist1.mulberry`
- raw tensor 文件推理：`examples/dl/inference_mnist_raw.mulberry`
- safetensors 推理：`examples/dl/inference_mnist_safetensors.mulberry`
- safetensors per-sample training smoke：
  `examples/dl/training_mnist_safetensors.mulberry`
- safetensors mini-batch training smoke：
  `examples/dl/training_mnist_minibatch_safetensors.mulberry`
- 小型纯源码训练回归：`examples/dl/training_tiny.mulberry`

相关脚本：

- `tools/generate_inference_mnist1.py`
- `tools/export_mnist_raw_tensors.py`
- `tools/export_mnist_safetensors.py`
- `tools/export_mnist_training_safetensors.py`

常用运行方式：

```sh
python3 tools/export_mnist_safetensors.py
python3 tools/export_mnist_training_safetensors.py

./build/release/bin/mulberry-driver examples/dl/inference_mnist_safetensors.mulberry
./build/release/bin/mulberry-driver examples/dl/training_mnist_safetensors.mulberry
./build/release/bin/mulberry-driver examples/dl/training_mnist_minibatch_safetensors.mulberry
```

也可以直接生成 native executable：

```sh
./build/release/bin/mulberry-driver examples/dl/inference_mnist_safetensors.mulberry -o /tmp/mnist
/tmp/mnist
```

safetensors training 文件当前使用 batch tensor 布局：

- `train_x` / `test_x`：`[N, 784, 1]`
- `train_y` / `test_y`：`[N, 10, 1]`

training 示例通过 `nn.TensorDataset` 把 batch tensor 切成 input/label pair view。
当前 backprop helper 是 `nn.twoLayerGradient()`，只覆盖 Nielsen 784-30-10 这个
2-layer FCN 正向路径需要的手写梯度，不是通用 autograd。

## 文档

当前保留的公开文档：

- [Builtins](docs/Builtins.md)
- [Grammar](docs/Grammar.md)
- [Deep learning examples](examples/dl/README.md)

更多 IR/lowering 细节目前仍在快速变化中，暂时以源码和 lit tests 为准。

## 已知限制

- 这是学习用编译器，不是生产编译器。
- stdlib、namespace、import、package registry 仍然是够用优先的早期实现。
- `Ptr<T>` 仍存在于 stdlib/compiler 内部；普通用户 surface 已经隐藏它，但 FFI 设计还没完成。
- `tensor.from(...)` 仍是 compiler-known stdlib entry，不是完全由 Mulberry stdlib
  自己实现的普通函数。干净实现需要更强的 comptime / reflection / overload 能力。
- `mulberry.nn` 当前主要覆盖 rank-2 Tensor 的 FCN 路径；CNN、broadcasting、batch matmul、
  GPU backend 和自动求导还没实现。
- 当前 safetensors / JSON 支持只覆盖项目需要的子集，不是完整通用库。
- JIT、object file 和 executable generation 已对当前正向子集打开，但 ABI、FFI 和
  runtime 打包还很早期。
