# Cherry 编程语言

Cherry 是一个小型编译器实验项目。当前版本基于
[Nicola Lancellotti 的 Cherry 项目](https://github.com/NicolaLancellotti/cherry)
继续开发，目标是把 Cherry 源码 lowering 到标准 MLIR、Clang CIR Dialect、
Linalg/MemRef 和 LLVM，并逐步加入深度学习相关能力。

当前项目使用 LLVM 22.1.0，并开启 ClangIR/CIR 支持。

## 项目来源

原始 Cherry 项目实现了一个教学性质的小语言，包括 lexer、parser、AST、sema、
LLVM IR codegen、MLIR dialect、driver、lit/FileCheck 测试等基础设施。

本仓库是在该项目基础上的改造版本。主要方向是：

- 通用语言部分尽量输出标准 MLIR 和 Clang CIR Dialect，而不是用 Cherry 自定义 dialect 表达所有语言结构。
- 张量/list 数据使用标准 `memref`，方便交给 MLIR `linalg`、`math`、`arith` 等 dialect 做 lowering 和优化。
- 新增 `cherry_nn` 深度学习 dialect，用于表达神经网络相关 operation，并 lowering 到 MLIR Linalg。

## 依赖

需要安装以下依赖：

- CMake
- Ninja
- Make
- Clang
- Python 3

## 构建 Cherry

构建 LLVM 和 Cherry：

```sh
make all
```

如果 LLVM 已经构建好，只构建 Cherry：

```sh
make cherry-build
```

## 语法和内建函数

- [Cherry 语法文档](/docs/Grammar.md)
- [内建函数文档](/docs/Builtins.md)

## Driver 参数

可以参考 `test/cherry/Driver` 里的调用示例。

| 含义 | 参数 |
| --- | --- |
| dump token | `-dump=tokens` |
| parse 并 dump AST | `-dump=parse` |
| parse、type-check 并 dump AST | `-dump=ast` |
| dump Cherry MLIR | `-dump=mlir` |
| dump Cherry + SCF lowering 后的 MLIR | `-dump=mlir1` |
| dump Cherry + SCF + Arith + CF + Func lowering 后的 MLIR | `-dump=mlir2` |
| dump Cherry NN lowering 到 Linalg 后的 MLIR | `-dump=linalg` |
| dump LLVM Dialect MLIR | `-dump=mlir-llvm` |
| dump LLVM IR | `-dump=llvm` |
| 选择 LLVM backend | `-b=llvm` |
| 选择 MLIR backend，默认值 | `-b=mlir` |
| parse 并 type-check | `-typecheck` |
| 开启优化 | `-opt` |
| 生成目标文件 | `-c[=<FILE_PATH>]` |

## 语法示例

```cherry
# 这是注释

struct A { }

struct B {
  x: Bool,
  y: A
}

fn bar(x: UInt64, y: Bool): B {
  var k: Bool = y;

  k = if k {
    print(18446744073709551615);
    false
  } else {
    print(1);
    true
  };

  var unit: () = while k {
    k = false;
    ()
  };

  B(k, A())
}

fn baz(): () {
  ()
}

fn main(): UInt64 {
  0 % 3 * 8 / 4 + 3 - 1;
  3 lt 1; 3 le 1; 3 gt 1; 3 ge 1;
  true and false or true eq false neq true;

  var structValue: B = bar(18446744073709551615, false);
  print(boolToUInt64(structValue.x));
  baz();
  0
}
```

## 通用语言部分的改进

相对于原始 Cherry 项目，当前版本在通用语言部分主要做了这些改进：

- 将 MLIRGen 的通用语言结构迁移到标准 MLIR 和 Clang CIR Dialect，例如函数、返回、调用、条件、循环、基础标量类型和结构体相关 IR。
- 支持 `Float32` 类型，包括 lexer、parser、sema、MLIRGen、LLVM lowering，以及负数 Float32 literal。
- 支持 `struct`，并使用 Clang CIR 的 `cir.record`、`cir.get_member`、`cir.alloca`、`cir.store`、`cir.load` 等机制表达结构体。
- 支持多维 list/tensor 类型语法，例如 `UInt64[2, 3]` 和 `Float32[30, 784]`。
- list/tensor 的底层数据使用标准 MLIR `memref`，而不是自定义 Cherry 容器类型。
- 支持 list literal、嵌套 list literal、多维 list 访问和赋值。
- 在 MLIR lowering pipeline 中补齐了从 CIR、MemRef、Linalg、Math、Arith 到 LLVM Dialect 的路径，使 MLIR backend 可以 JIT 执行和生成目标文件。
- 增加了针对新语法和 lowering 行为的 lit/FileCheck 测试，例如 Float32、负 Float32、list、matmul、sigmoid、argmax 等。

## 深度学习方面的工作

当前项目已经加入一个新的深度学习 dialect：`cherry_nn`。

`cherry_nn` 目前支持：

- `matmul`：矩阵乘法，lowering 到 `linalg.fill` + `linalg.matmul`。
- `matadd`：矩阵逐元素加法，lowering 到 `linalg.add`。
- `transpose`：矩阵转置，lowering 到 `linalg.transpose`。
- `exp`：逐元素指数函数，lowering 到 `linalg.map` + `math.exp`。
- `sigmoid`：逐元素 sigmoid，lowering 到 `linalg.map`、`math.exp` 和 `arith`。
- `argmax`：返回最大元素的 flat index，lowering 到 `linalg.generic`。
- `cherry_nn.cast`：在标准 MLIR scalar 类型和 CIR scalar 类型之间做临时桥接。

张量数据使用 `memref`，这是为了让数据可以直接交给 MLIR `linalg` 等标准 dialect
处理。Cherry 的通用语言类型目前仍大量使用 Clang CIR 类型，因此 `cherry_nn.cast`
用于处理 `i64` 和 `!cir.int<u, 64>` 等类型边界。

## MNIST 推理示例

当前仓库包含一个基于 Michael Nielsen 深度学习教程数据的 MNIST 单样本推理示例：

- 数据文件：`data/mnist-784-30-10.json`
- MNIST 测试数据：`data/mnist.pkl.gz`
- 数据生成脚本：`tools/generate_inference_mnist1.py`
- Cherry 推理脚本：`examples/dl/inference_mnist1.cherry`

重新生成推理脚本：

```sh
/usr/bin/python3 tools/generate_inference_mnist1.py
```

执行推理：

```sh
./build/debug/bin/cherry-driver examples/dl/inference_mnist1.cherry
```

当前示例会执行：

```cherry
var z1: Float32[30, 1] = matadd(matmul(w1, x), b1);
var a1: Float32[30, 1] = sigmoid(z1);
var z2: Float32[10, 1] = matadd(matmul(w2, a1), b2);
var a2: Float32[10, 1] = sigmoid(z2);
var pred: UInt64 = argmax(a2);
print(pred);
```

对 `test_data[0]` 的推理结果应输出：

```text
7
```

目前该示例把 weights、bias 和输入数据全部展开为 Cherry 源码中的 Float32 literal，
因此编译和 JIT 执行很慢。后续更合理的方向是支持从外部文件加载 tensor，或使用
MLIR global/constant data 表达权重。

## 测试

项目使用 LLVM/MLIR 风格的 lit + FileCheck 测试。常用 smoke test：

```sh
./build/debug/bin/cherry-driver --dump=mlir test/cherry/Language/structs.cherry
```

也可以针对深度学习 lowering 做检查：

```sh
./build/debug/bin/cherry-driver --dump=linalg test/cherry/Language/matmul.cherry
./build/debug/bin/cherry-driver --dump=linalg test/cherry/Language/sigmoid.cherry
./build/debug/bin/cherry-driver --dump=linalg test/cherry/Language/argmax.cherry
```
