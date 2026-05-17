# Mulberry

Mulberry 是一个个人编译器学习项目，目标是把一个小型源语言 lowering 到
MLIR、Clang CIR、Linalg/MemRef 和 LLVM，并逐步加入深度学习推理能力。

当前版本基于 [Nicola Lancellotti 的 Cherry 项目](https://github.com/NicolaLancellotti/cherry)
继续开发。原项目提供了 lexer、parser、AST、Sema、driver、MLIR dialect、
LLVM IR codegen 和 lit/FileCheck 测试等基础设施；Mulberry 在此基础上把主线
pipeline 改造成更接近 MLIR/AI compiler 的方向。

项目使用 `llvmorg-22.1.0`，并开启 ClangIR/CIR 支持。

## 当前状态

这是一个早期但已经可以工作的里程碑版本。

已经完成：

- 支持一个 Cherry-like 小语言的 lexer、parser、AST、Sema 和 driver。
- 支持 `UInt64`、`Bool`、`Float32`、`struct`、`const` 和多维 list/tensor 语法。
- 普通语言结构主要 lowering 到 MLIR 标准 dialect 和 Clang CIR。
- `struct` lowering 到 Clang CIR 的 `cir.record`、`cir.get_member`、`cir.alloca`、
  `cir.load/store`。
- list/tensor 数据使用标准 MLIR `memref` 表达，可直接交给 Linalg。
- 新增 `cherry_nn` 深度学习 dialect，并支持 lowering 到 `linalg`、`math`、
  `arith` 和 `memref`。
- 打通 MLIR/CIR/Linalg 到 LLVM 的 lowering，并支持 x64 Linux JIT 执行。
- 包含基于 Michael Nielsen MNIST 数据的单样本推理示例，当前输出结果为 `7`。
- 使用 lit/FileCheck 覆盖部分语法、IR 生成和 lowering 行为。

已知限制：

- 当前分支仍保留旧的 string-based type system。
- 内部工具名、namespace、dialect 名和测试目录仍大量保留 `cherry` 命名。
- MNIST 示例目前把权重、bias 和输入数据展开为源代码 literal。
- 这是学习型 compiler，不是工业级生产编译器。

## 编译 Pipeline

```text
Mulberry/Cherry source
  -> Lexer / Parser / AST
  -> Sema / type checking
  -> MLIRGen
       - Clang CIR: scalar, control flow, function, record/struct
       - memref: list/tensor storage
       - cherry_nn: neural-network operations
  -> cherry_nn to linalg/math/arith/memref
  -> linalg to loops
  -> CIR to LLVM
  -> LLVM dialect / LLVM IR
  -> x64 Linux JIT or object file
```

## 构建与测试

构建 LLVM 和 Mulberry：

```sh
make all
```

如果 LLVM 已经构建好，只构建当前项目：

```sh
make cherry-build
```

常用 smoke test：

```sh
./build/debug/bin/cherry-driver --dump=mlir test/cherry/Language/structs.cherry
```

运行 lit 测试：

```sh
cmake --build build/debug --target check-cherry
```

注意：当前构建目标和工具名仍保留 `cherry` 命名。

## 文档

- [语法文档](docs/Grammar.md)
- [内建函数文档](docs/Builtins.md)
- Driver 参数和更多示例可以参考 `test/cherry/Driver`。

## 深度学习支持

当前版本包含内部深度学习 dialect：`cherry_nn`。

目前支持：

- `matmul`：矩阵乘法，lowering 到 `linalg.fill` + `linalg.matmul`。
- `matadd`：矩阵逐元素加法，lowering 到 `linalg.add`。
- `transpose`：矩阵转置，lowering 到 `linalg.transpose`。
- `exp`：逐元素指数函数，lowering 到 `linalg.map` + `math.exp`。
- `sigmoid`：逐元素 sigmoid，lowering 到 `linalg.map`、`math.exp` 和 `arith`。
- `argmax`：返回最大元素的 flat index，lowering 到 `linalg.generic`。
- `cherry_nn.cast`：在标准 MLIR scalar 类型和 CIR scalar 类型之间做临时桥接。

## MNIST 推理示例

仓库包含一个基于 Michael Nielsen 深度学习教程数据的 MNIST 单样本推理示例：

- 权重和 bias：`data/mnist-784-30-10.json`
- MNIST 测试数据：`data/mnist.pkl.gz`
- 数据生成脚本：`tools/generate_inference_mnist1.py`
- 生成后的推理程序：`examples/dl/inference_mnist1.cherry`

重新生成推理程序：

```sh
/usr/bin/python3 tools/generate_inference_mnist1.py
```

执行推理：

```sh
./build/debug/bin/cherry-driver examples/dl/inference_mnist1.cherry
```

当前示例会执行两层全连接网络的 forward pass：

```text
z1 = w1 * x + b1
a1 = sigmoid(z1)
z2 = w2 * a1 + b2
a2 = sigmoid(z2)
pred = argmax(a2)
```

对 `test_data[0]` 的推理结果应输出：

```text
7
```

## 后续方向

- 将当前字符串编码的类型系统重构为结构化 type system。
- 为 `cherry_nn` ops 补充 verifier、traits 和更完整的错误诊断。
- 支持从外部文件加载 tensor/weights，避免把大型数据展开为源代码 literal。
- 分阶段把内部命名从 `cherry` 迁移到 `mulberry`。
- 探索更完整的 CPU 优化 pipeline，以及后续可能的 GPU lowering。
