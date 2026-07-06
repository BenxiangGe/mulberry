# 标准库对象模型

本文档记录标准库对象的 source-level 设计：`String`、`File`、`Array`、`List`、
`Tensor` 和用户 `struct` 都按 object reference 使用。内部 storage、GC allocation、
record/field lowering 和 runtime ABI 是 compiler/stdlib/runtime 实现细节，普通用户
不需要学习这些拼写。

更底层的实现细节只放在 internal 文档里：

- `docs/MulberryLowering.md`：core dialect / lowering / runtime ABI。
- `docs/PtrAndHeapObjectModel.md`：内部 typed address 和 heap object 模型。

## 动机

标准库对象不应该通过一组 source-visible dialect op 来表达。旧路径会把每个对象都拆成
type/op、verifier、MLIRGen 特判、lowering pattern 和 ABI descriptor，这会让语言语义
和后端实现缠在一起。

新的方向是：

```text
source type/API
  -> stdlib/comptime object
  -> generic compiler lowering
  -> runtime helper when needed
```

用户层只关心对象和 API；compiler 内部可以继续使用必要的 record、heap object 和
lowering bridge，但这些不是 Mulberry source surface。

## 基本原则

- Scalar 按 value 传递：`Bool`、`UInt8`、`UInt64`、`Float32`、`Char`。
- Object 按 reference 传递：`String`、`File`、`Array<T, N>`、`List<T>`、
  `Tensor<T>` 和用户 `struct`。
- 赋值、传参和返回 object 时复制 reference，不做隐式 deep copy。
- 通过一个 reference 修改对象，别的 alias 可以观察到同一个修改。
- `const` / mutable receiver 是第一层约束机制；当前不做 Rust-style borrow checking。
- 显式复制对象后续应通过 `clone()` / `copy()` 这类 API 表达。
- 遇到缺失能力时补语言/stdlib 能力，不在 MLIRGen 或 lowering 里继续堆
  object-specific workaround。

## List

`List<T>` 是 growable container，类似简化版 `std::vector<T>` / Rust `Vec<T>`：

```mulberry
var xs: List<UInt64> = list.from([1, 2, 3]);
xs.push(4);
io.print(xs.size());
io.print(xs[0]);
```

`List<T>` 不再有 `mulberry.list.*` source/dialect path。用户代码通过 `List<T>`、
`list.from(...)`、`list.withCapacity(...)`、method call 和 index syntax 操作它。
底层容量、元素 buffer 和增长策略属于 `std.list` 的实现细节。

## String

`String` 是标准库 object，不是 builtin string descriptor，也不是裸 runtime handle。
字符串字面量直接构造一个 `String` object：

```mulberry
var name: String = "mnist";
io.print(name);
```

`String` 的实现会保存长度和 byte storage；第一版字面量数据仍保持 NUL 结尾，便于
复用 C runtime API。用户层不需要直接接触 byte storage。需要字符串操作时，应通过
stdlib API 暴露，例如后续的 `len()`、`slice()`、`concat()` 等。

## File

`File` 是标准库 object，包装 runtime file handle：

```mulberry
var file: File = io.open("data/model.safetensors", "rb");
io.close(file);
```

`io.open/read/write/close` 是用户 API。真实 `FILE*`、seek/read/write helper 和 byte
view 都留在 runtime / stdlib implementation 里，不作为用户语言概念。

## Array / List / Tensor 分层

当前 source-level 规则：

- 普通 `[]` literal 默认是 `Array`，它是 Mulberry 的通用固定长度数组。
- `List<T>` 是 growable container，适合 push/pop 和动态长度数据。
- `Tensor<T>` 类似 NumPy `ndarray`，是显式 numeric buffer / bufferization object。

示例：

```mulberry
var data = [[1.0, 2.0], [3.0, 4.0]];        // Array
var tensor2d: Tensor<Float32> = tensor.from(data);
var zeros: Tensor<Float32> = tensor.zeros([30, 784]);
```

也就是说，`[]` 本身不因为 expected type 自动变成 `Tensor<T>` 或 `List<T>`。需要
Tensor 时显式调用 `tensor.from(...)` / `tensor.zeros(...)`；需要 List 时显式调用
`list.from(...)` / `list.withCapacity(...)`。

## Tensor

`Tensor<T>` 是 ndarray-style object，保存 runtime rank、shape、元素总数和 stride
metadata。它没有 compile-time rank 参数，旧的 `Float32[?, ?]` 这类 ranked tensor
source spelling 已删除。

用户层推荐写法：

```mulberry
var batch: Tensor<Float32> = tensor.zeros([10, 784]);
var x: Tensor<Float32> = tensor.from([[1.0], [2.0]]);
var value: Float32 = x[1, 0];
io.print(x.numel());
```

metadata 通过 API 访问：`ndim()`、`numel()`、`shape()`、`dim(i)`、`stride(i)`。
元素访问使用 `tensor[i]` 或 `tensor[i, j]`。如果后续支持 slice/view，也应该保持
`Tensor<T>` 的 source 形态稳定，只改变内部实现。

`safetensors.readTensor(file, name)` 返回 `Tensor<Float32>`：

```mulberry
var file: File = io.open("data/mnist-784-30-10.safetensors", "rb");
var x: Tensor<Float32> = safetensors.readTensor(file, "x");
io.close(file);
```

## 迁移状态

- `List<T>` 已迁到 stdlib/comptime path，旧 `mulberry.list.*` 已删除。
- `String` 已是 stdlib object，旧 `mulberry.string.*` 已删除。
- `File` 已是 stdlib object，旧 `mulberry.file.*` 已删除。
- safetensors reader/writer 走 stdlib JSON cursor parser 和 Tensor object path。
- `Tensor<T>` source surface 已统一成 ndarray-style object；旧 ranked tensor source
  spelling 已删除。
- `mulberry_core.tensor.*` 仍作为 internal Tensor lowering mechanism 存在，后续随着
  NN/linalg 直接消费 `Tensor<T>` 再继续收缩。

## 非目标

- 不把内部 storage spelling 暴露成用户 API。
- 不恢复 string/file/list/safetensors 的 source-level dialect op。
- 不在当前阶段引入 allocator trait、异常、borrow checker 或复杂 ownership。
- 不急着删除必要的 internal Tensor lowering bridge；先保持正向训练/推理路径稳定。

当前阶段的目标是让标准库对象模型变简单、可解释、可迁移。
