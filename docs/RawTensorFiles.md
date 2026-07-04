# Raw Tensor 文件

本文档记录 Mulberry 读取和写入 raw Tensor 文件时采用的最小约定。

这不是一个通用 tensor 文件格式。它没有 magic number、version、dtype、rank 或
shape header。文件只保存连续的 tensor element bytes；element type 和 shape 由
Mulberry 源码里的类型声明决定。

raw `.f32` 文件只是 bootstrap/debug 格式。日常深度学习推理和后续训练数据优先
使用 [Safetensors](Safetensors.md)，用单个文件保存多个 Tensor，并在 header 中
记录 dtype、shape 和 payload offsets。

例如：

```mulberry
var w1: Tensor<Float32> = tensor.zeros([30, 784]);
const file: File = io.open("data/mnist-784-30-10-raw/w1.f32", "rb");
io.read(file, w1);
io.close(file);
```

这里 `tensor.zeros([30, 784])` 决定读取多少数据，以及如何解释这些 bytes。文件本身
只包含 `30 * 784` 个 `Float32` element。

## Layout

当前约定：

- element 按 row-major 顺序连续保存。
- `UInt8` 直接保存 byte。
- `UInt64` 保存 little-endian unsigned 64-bit integer。
- `Float32` 保存 little-endian IEEE-754 binary32。
- `read(file, tensor)` 和 `write(file, tensor)` 返回 byte count，不返回 element
  count。

因此 shape 为 `[4]` 的 `Tensor<Float32>` 文件大小是：

```text
4 elements * 4 bytes = 16 bytes
```

shape 为 `[30, 784]` 的 `Tensor<Float32>` 文件大小是：

```text
30 * 784 * 4 = 94080 bytes
```

## 为什么不加 Header

raw 格式的目标是先让训练和推理数据能从真实文件进入 Mulberry 程序，而不是先设计
完整 data format。现在 safetensors 路径已经可用，所以 raw 格式主要保留给最小
IO bootstrap、调试和对照测试。

把 type 和 shape 放在源码里有几个好处：

- 实现简单，直接复用已有 `read(file, tensor)`。
- `Tensor<T>` header 明确保存 runtime rank、shape、numel 和 strides；更完整的
  header/object 约定见 `docs/StdlibObjects.md`。
- 文件内容可以直接映射到连续 tensor storage。
- safetensors 已经提供 dtype、shape 和 payload offsets。后续正常功能应优先扩展
  safetensors，而不是在 raw tensor 之上继续发明 manifest 或 header。

目前不处理：

- endian 自动识别。
- dtype/shape runtime 校验。
- 短读、文件不存在等错误处理。
- 跨平台 ABI padding 问题。当前只支持 plain numeric tensor element。

## MNIST 导出

可以用下面的脚本把当前 Nielsen MNIST JSON 和 sample 数据导出成 raw tensor 文件：

```sh
python3 tools/export_mnist_raw_tensors.py
```

默认输出目录：

```text
data/mnist-784-30-10-raw/
```

输出文件：

```text
w1.f32  Tensor<Float32>, shape [30, 784]
b1.f32  Tensor<Float32>, shape [30, 1]
w2.f32  Tensor<Float32>, shape [10, 30]
b2.f32  Tensor<Float32>, shape [10, 1]
x.f32   Tensor<Float32>, shape [784, 1]
label.txt
```

`label.txt` 只是给人和测试脚本看的辅助文件，不属于 raw tensor 文件格式。

这个 raw 文件版本示例现在作为外部 NN package 的回归素材保留。由于
`mulberry.nn` 已从 core 拆出，driver 通过 bundled package registry 按需加载
`MulberryNNPackage` 并接入 NN package pipeline。下面的命令已经重新成为默认
JIT smoke：

```sh
./build/release/bin/mulberry-driver examples/dl/inference_mnist_raw.mulberry
```

这个示例先用 `tensor.zeros([shape])` 创建 `Tensor<Float32>` header，然后用
`io.read(file, tensor)` 把 raw bytes 读入对应 payload。推理部分直接用
`List<Tensor<Float32>>` 保存 layer 权重和 bias。普通元素访问使用 `tensor[i, j]`。
