# Mulberry Lowering

本文档记录当前 Mulberry lowering 的边界、ABI 形状和仍然保留的限制。核心原则是：
MLIRGen 只生成自洽的高层 Mulberry IR；函数边界和后端 ABI 由独立 lowering pass
处理，不能把 workaround 塞回 MLIRGen。

## 当前边界

`--lower-mulberry` 现在做的是 storage-level lowering，不是最终 object/JIT ABI：

- `mulberry.tensor` lower 成 MLIR `memref`，方便 `cherry_nn` 继续 lower 到
  `linalg`、`math`、`arith` 和 `memref`。
- `mulberry.list<T>` 在函数内部 lower 成本地 list storage。
- `List<Tensor>` 穿过受支持的函数边界时，会先 rewrite 成 descriptor 语义，再
  lower 成 LLVM descriptor。
- scalar 和 record storage 在能直接映射的地方使用 LLVM dialect。
- `-dump=mlir-llvm`、`-dump=llvm`、JIT 和 object generation 仍然关闭，等完整
  backend ABI 单独设计。

## IR 分层

`!mulberry.list<T>` 是源语言语义层的 `List<T>` value。MLIRGen 为用户可见的
list 生成这个类型。

`!mulberry.list_storage<T>` 是 lowering 中使用的连续元素 storage。它是实现细节，
不是函数 ABI。

`!mulberry.list_desc<T>` 是 descriptor-stage value。它表示一个可以穿过函数边界的
first-class list descriptor。

逻辑 descriptor 是：

```text
{ length: index, data: !mulberry.list_storage<T> }
```

后端 ABI descriptor 是：

```text
{ length: i64, data: ptr }
```

注意 `length` 永远是运行时元素个数，不是 element type。因此
`List<UInt64>`、`List<Float32>`、`List<Float32[?, ?]>` 的 descriptor 外壳都是：

```mlir
!llvm.struct<(i64, ptr)>
```

element type 的差异体现在 `data` 指向的连续 element ABI storage 里。由于 LLVM
dialect 使用 opaque pointer，`data` 字段打印出来统一是 `ptr`。

## Tensor ABI

Mulberry Tensor 是可写数组，rank 静态，dimension 可以是静态或动态。Tensor ABI
descriptor 当前按 rank 固定：

```text
TensorABI<T, rank> = {
  data: ptr<#llvm.address_space<0>>,
  sizes: array<rank x i64>,
  strides: array<rank x i64>
}
```

`data` 指向连续元素 storage；`sizes` 是每个维度的运行时大小；`strides` 是按
row-major layout 访问元素时的步长。这个 layout 借鉴 MLIR memref descriptor，
但没有直接把完整 memref ABI 暴露成 Mulberry 语言 ABI。

`tensor.desc_pack` 把本地 Tensor/memref 打包成这个 ABI descriptor。当前 lowering
会从 memref 中提取 data pointer、sizes 和 strides。

`tensor.desc_unpack` 做反方向：从 Tensor ABI descriptor 重建可被后续 Tensor /
`cherry_nn` op 使用的 memref view。当前实现使用：

- `llvm.extractvalue` 取出 descriptor 的 data、sizes 和 strides。
- `ptr.from_ptr` 把 ABI data pointer 变成 MLIR ptr dialect value。
- `memref.reinterpret_cast` 用 metadata 重建 ranked memref view。
- `memref.memory_space_cast` 回到普通 tensor lowering 使用的 memref type。

这里有一个重要约束：`desc_unpack` 不拥有 data 的生命周期。它只重建 view；data
必须由函数参数、返回 descriptor、heap list storage 或未来 runtime 保证仍然有效。

`!mulberry.tensor_handle` 和 `tensor.handle_from_desc` 仍然存在，但现在不是
`List<Tensor>` boundary 的主路径。当前主路径是：

```text
list.desc_get -> tensor.desc_unpack -> Tensor/memref value
```

`tensor_handle` 继续作为 future reconstruction/runtime handle 的实验性 IR，普通
`--lower-mulberry` 仍然对它 fail-fast。

## List ABI

`List<T>` 的 ABI 外壳固定为：

```text
ListABI<T> = { length: i64, data: ptr }
```

`data` 指向连续的 element ABI storage：

```text
List<UInt64>        -> data: ptr<i64>
List<Float32[?, ?]> -> data: ptr<TensorABI<f32, 2>>
List<List<UInt64>>  -> data: ptr<ListABI<i64>>
```

当前实现已经支持 `List<Tensor>` 需要的 descriptor-backed storage：

```text
!mulberry.list_storage<!mulberry.tensor_desc<...>>
  -> ptr to contiguous Tensor ABI descriptors
```

普通函数内部的本地 `List<Tensor>` 仍可以用 memref-backed storage 作为 lowering
实现细节，以服务 for-loop 推理和 linalg lowering。这条本地 storage 路径不是函数
ABI。

## Boundary Preparation

`prepare-mulberry-boundaries` 是函数边界 rewrite pass。它负责把 source-level
`List<Tensor>` 边界改写成 descriptor 边界，避免 MLIRGen 关心 ABI。

函数边界包括：

- `func.func` 参数。
- `func.func` 返回值。
- `func.call` 参数。
- `func.call` 返回值。
- `func.return` operand。

当前支持的 rewrite 形态是：

```mlir
func.func @f(%xs: !mulberry.list<!mulberry.tensor<?x?xf32>>)
```

变成：

```mlir
func.func @f(
    %xs: !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>)
```

callee body 里的 list 操作同步改写：

```text
list.size %xs
  -> list.desc_length %xs -> arith.index_cast

list.get %xs[%i]
  -> list.desc_get %xs[%i] -> tensor.desc_unpack
```

call site 同步插入 descriptor materialization：

```text
call @f(%list)
  -> %desc = mulberry.list.to_desc %list
  -> call @f(%desc)
```

这个 pass 只在能同步改写 callee signature、callee body、call site 和 return
operands 时动手。否则 fail-fast，避免生成半转换 IR。

## 当前正向支持

当前已经支持的核心路径：

- 函数内部 Tensor、List、`cherry_nn` lowering。
- Tensor 函数参数和返回值 lower 成 memref function boundary。
- 显式 `list_desc<TensorDesc>` 的 ABI projection 和 lowering。
- source-level `List<Tensor>` 参数边界，在所有 symbol use 都是 direct
  `func.call` 且 body use 只需要 `list.size/list.get` 时 rewrite。
- source-level `List<Tensor>` 返回边界，在 return value 直接来自本地
  `mulberry.list.create`，且 caller result use 可 rewrite 时支持。
- caller 侧拿到 `List<Tensor>` return 后，可以执行 `size(result)`，也可以执行
  `result[i]` 并把得到的 Tensor 继续交给 `cherry_nn` op。

源码级正向例子：

```cherry
fn make(): List<Float32[?, ?]> {
  const w: Float32[2, 2] = [[1.0, 0.0], [0.0, 1.0]];
  [w]
}

fn main(): UInt64 {
  const x: Float32[2, 1] = [[0.2], [0.8]];
  argmax(matmul(make()[0], x))
}
```

lowering 后的关键路径是：

```text
make() -> !llvm.struct<(i64, ptr)>
main(): call @make
main(): extract data pointer
main(): load TensorABI descriptor
main(): tensor.desc_unpack -> memref
main(): linalg.matmul / cherry_nn lowering
```

## Ownership

返回 `List<Tensor>` 时，descriptor 的 `data` 不能指向 callee-local stack storage。
因此 boundary rewrite 会插入：

```mlir
mulberry.list.escape_storage
```

当前采用明确标注的临时策略：

- 只支持 `List<TensorDesc>` escaping storage。
- lowering 使用 `malloc` 分配 heap storage。
- 当前不生成 dealloc。
- 这适合当前小程序和推理脚本；training 如果频繁返回 list，需要补 dealloc、arena
  或 GC。

这不是最终 ownership 方案，只是为了避免返回悬空 pointer，同时让当前正向路径可以
完整 lower。

## 仍然限制

以下场景仍然不应该硬凑：

- 任意来源的 `List<Tensor>` value 直接 materialize 成 descriptor。当前
  `list.to_desc` 仍然主要支持本地 `list.create`。
- `List<UInt64>`、nested list、record field 里的 list 穿过函数边界。
- external function 的 `List<Tensor>` boundary。
- indirect symbol use、function pointer 或跨 module boundary。
- 完整 ownership，包括 dealloc、arena、GC、move/copy 语义。
- `tensor_handle` lowering。
- 最终 `mlir-llvm`、LLVM IR、JIT 和 object generation。

这些限制是有意保留的。缺少底层 ABI 或 ownership 设计时，继续 fail-fast 比生成
表面可用、实际语义不成立的 IR 更好。

## 借鉴 `std::vector<T>`

Mulberry 不需要完整 C++ template 语言特性，但 `List<T>` 的 lowering 可以借鉴
monomorphization 思想：

- `List<T>` 在语义层是泛型容器。
- `List<UInt64>`、`List<Float32[?, ?]>`、`List<List<UInt64>>` 在 lowering/ABI
  层是不同 concrete type。
- 每个 concrete `List<T>` 都递归计算自己的 element ABI、descriptor layout 和
  load/store lowering。

可以把当前 `List<T>` 理解成语言内建的 `std::vector<T>` 简化版本：

```text
用户层：List<T>
语义层：泛型容器类型
lowering 层：按 T 计算 element ABI
ABI 层：{ length, data: ptr<element ABI> }
```

当前不引入 `capacity`、allocator、异常安全、move/copy 语义或用户级 generic
function。等 training 需要更完整的容器语义时，再单独设计。

## 下一步

R3.8.88：继续扩大正向源码路径，优先让 Nielsen for-loop 推理脚本通过
`--dump=lowered-mlir` 的完整 lowering。

R3.8.89：如果 training 开始频繁创建/返回 list，补 heap dealloc、arena 或更明确的
runtime ownership 策略。

R3.8.90：在 `List<Tensor>` 正向路径稳定后，再考虑 scalar list boundary、nested
list、record field list 和最终 LLVM/JIT pipeline。
