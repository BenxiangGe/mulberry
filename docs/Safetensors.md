# Safetensors 数据文件

本文档记录 Mulberry 当前推荐的数据文件方向。raw `.f32` 文件只是
bootstrap/debug 格式；深度学习推理和后续训练数据优先使用 safetensors。

官方格式说明见：

- https://github.com/huggingface/safetensors
- https://huggingface.co/docs/safetensors/metadata_parsing

## 为什么选择 Safetensors

safetensors 很适合当前 Mulberry 的需求：

- 单文件可以保存多个 tensor，例如 `w1`、`b1`、`w2`、`b2`、`x`。
- 文件开头有 JSON header，记录 tensor 名字、dtype、shape 和 byte offsets。
- tensor payload 是连续 raw bytes，没有 pickle opcode，也不需要 zip/gzip。
- 可以只读取某个 tensor，不需要把整个文件都读入内存。
- 文件格式足够简单，当前 header parser 已经可以用 Mulberry stdlib 的 cursor JSON
  parser 实现；C++ runtime 只保留低层 file IO。

## 文件 Layout

safetensors 文件由三段组成：

```text
8-byte little-endian header length
JSON header
contiguous tensor payload bytes
```

JSON header 里的每个 tensor entry 类似：

```json
{
  "w1": {
    "dtype": "F32",
    "shape": [30, 784],
    "data_offsets": [0, 94080]
  }
}
```

`data_offsets` 是相对于 payload 开始位置的 offset。真实文件 offset 是：

```text
8 + headerLength + data_offsets[0]
```

## MNIST 导出

当前可以用下面的脚本导出 Nielsen MNIST 推理所需的单文件 safetensors：

```sh
python3 tools/export_mnist_safetensors.py
```

默认输出文件：

```text
data/mnist-784-30-10.safetensors
```

文件包含：

```text
w1  F32[30, 784]
b1  F32[30, 1]
w2  F32[10, 30]
b2  F32[10, 1]
x   F32[784, 1]
```

这个导出工具直接写 safetensors layout，不依赖 Python `safetensors` package。
Mulberry runtime 读取同一个文件格式。

这些文件和示例现在作为外部 NN package 的回归素材保留。当前已经有独立
`MulberryNNPackage`，driver 通过 bundled package registry 检测
`import mulberry.nn`，自动加载 package，并接入 source-level `nn.matmul(...)`
到 `mulberry_nn.*` / `linalg` 的 lowering。
下面的命令已经重新成为默认 JIT smoke：

```sh
./build/release/bin/mulberry-driver examples/dl/inference_mnist_safetensors.mulberry
```

`test_data[0]` 的期望预测结果是：

```text
7
```

也可以导出一个小的 training 子集：

```sh
python3 tools/export_mnist_training_safetensors.py
```

默认输出文件：

```text
data/mnist-784-30-10-training.safetensors
```

文件包含初始网络参数，以及 batch 形式的 training / test 样本：

```text
w1         F32[30, 784]
b1         F32[30, 1]
w2         F32[10, 30]
b2         F32[10, 1]
train_x    F32[10, 784, 1]
train_y    F32[10, 10, 1]
test_x     F32[10, 784, 1]
test_y     F32[10, 10, 1]
```

`train_y` 和 `test_y` 是 one-hot label batch。当前 Mulberry 还没有完整 dataset
iterator，所以 training bootstrap 先用 `nn.TensorDataset` 从 batch tensor 生成
成对的样本 view。后续如果语言补齐 dataset 抽象，可以把这层 lightweight view
收进去。

这个 smoke 使用默认导出的 `10` 个 training 样本跑 `30` 个 epoch 的 per-sample SGD，
然后先输出 bootstrap training/test 的 `correct`、`total` 和 `accuracyBasisPoints`，
最后读取 `data/mnist-784-30-10.safetensors` 里的 `x` 做一次 inference。训练后的权重
会写成 safetensors checkpoint 再读回验证。当前期望输出是 `10`、`10`、`10000`、`9`、
`10`、`9000` 和 `7`。
输出层 delta 采用 Nielsen `network2.py` 默认的 CrossEntropy 形式：
`delta = a - y`。当前已接入固定小 lambda 的 L2 regularization 正向路径，并会写出
safetensors checkpoint。per-sample 和 mini-batch 示例复用 `nn.twoLayerGradient()`
计算当前 2-layer FCN 的梯度，避免示例里重复手写 backprop。这个 helper 只覆盖当前
Nielsen-style 两层网络，不是通用训练框架。per-sample 和 mini-batch 示例都会在
每个 epoch 前调用 `TensorDataset.shuffle()` 同步打乱 input/label。mini-batch 示例从
`dataset.size()` 和 `batchSize` 派生 batch 数，最后一个不满 batch 也会按实际大小更新；
完整 dataset iterator 后续再补。`nn_minibatch_tail_safetensors_jit.mulberry` 会导出
11 个 training 样本来覆盖非整除 batch 的正向路径。

```sh
./build/release/bin/mulberry-driver examples/dl/training_mnist_safetensors.mulberry
```

## Mulberry API

第一版固定返回 `Tensor<Float32>`，不做裸类型推断：

```mulberry
const file: safetensors.TensorFile =
    safetensors.open("data/mnist-784-30-10.safetensors");

var w1: Tensor<Float32> = safetensors.read(file, "w1");
var b1: Tensor<Float32> = safetensors.read(file, "b1");
var w2: Tensor<Float32> = safetensors.read(file, "w2");
var b2: Tensor<Float32> = safetensors.read(file, "b2");
var x: Tensor<Float32> = safetensors.read(file, "x");

safetensors.close(file);
```

safetensors header 提供 dtype、concrete shape 和 payload offset。当前推荐直接使用
`Tensor<Float32>` header：

```mulberry
var w1: Tensor<Float32> = safetensors.read(file, "w1");
```

含义是：

- stdlib 签名要求返回 `Tensor<Float32>`。
- stdlib 从 header 读出实际 shape，例如 `[30, 784]`。
- stdlib 用 `tensor.zeros(shape)` 分配 payload、sizes 和 strides buffer。
- stdlib 调用 `io.read(file, value)` 读取 payload，最终返回 `Tensor<Float32>` header。

## 分层设计

`safetensors.read()` 现在由 Mulberry stdlib 中的 cursor JSON parser 解析
safetensors header。当前阶段的分层是：

```text
Mulberry source
  -> stdlib: safetensors.open(path) / safetensors.read(file, name)
  -> stdlib json parser: parse header / shape / offsets
  -> stdlib tensor.zeros/metadata helpers: build Tensor<Float32> header
  -> stdlib io.read: read payload bytes
  -> Runtime: file seek/read and byte zeroing helpers only
```

`safetensors.open(path)` 返回 `TensorFile`，其中缓存了 header bytes 和 header length。
后续多次 `safetensors.read(file, name)` 会复用这个缓存，只重新创建 cursor parser。
旧的 `safetensors.readTensor(File, name)` 仍作为兼容入口保留，但每次调用都会重新读
header。

```mulberry
const tensors: safetensors.TensorFile =
    safetensors.open("data/mnist-784-30-10.safetensors");
var w1: Tensor<Float32> = safetensors.read(tensors, "w1");
```

## Runtime Helper 责任

Mulberry stdlib parser 负责：

- `seek(file, 0)`。
- 读前 8 bytes，得到 JSON header length。
- 读 JSON header。
- 查找 tensor name。
- 检查 dtype。
- 读取 `shape` 和 `data_offsets`。
- 定位 payload offset。

runtime helper 只负责：

- file open/close/seek/read/write。
- byte-level zero fill。

compiler/lowering 不再插入 safetensors-specific 隐藏调用。

## 当前限制

- 只支持 `Float32` / safetensors `F32`。
- 只支持 Tensor，不支持 List 或 struct 直接从 safetensors 读取。
- `safetensors.read()` / `safetensors.readTensor()` 统一返回 `Tensor<Float32>` header。
- 不支持 `var w = safetensors.read(file, "w1");` 这种裸推断。
- training 数据当前使用 `train_x` / `train_y` / `test_x` / `test_y` batch tensor
  bootstrap 布局，还不是最终 dataset API。
- shape mismatch、dtype mismatch、找不到 tensor 先 fail-fast。
- `TensorFile` 已缓存 header bytes；还没有 Map/dataset iterator 级别的索引缓存。
- 支持把 `List<String>` + `List<Tensor<Float32>>` 写成 safetensors checkpoint；当前
  writer 只覆盖可信 ASCII tensor name 和 `Float32` tensor 正向路径。

## 后续方向

当前 safetensors reader 已经开始使用 Mulberry script 实现 header parsing。后续如果
继续增强，会自然推动这些真实能力：

- byte / string indexing
- slice
- string compare
- integer parsing
- dynamic List
- Map 或 record-like dictionary
- error handling
- 精确 file seek/read

当前 C++ runtime 只保留低层 file IO 和 byte helper，避免把 JSON/safetensors 语义继续
塞在 lowering 或 runtime bridge 里。
