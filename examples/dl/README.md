# 深度学习示例

这个目录下的文件现在主要作为独立 `mulberry.nn` package 的回归和设计素材。

Core Mulberry 目前仍然保留 tensor literal、`Tensor<T>`、`List<T>`、raw file IO
和 safetensors IO。神经网络 primitive，例如 `nn.matmul`、`nn.sigmoid`、
`nn.argmax` 等，已经从 core compiler 中移到独立 `mulberry.nn` package。

现在的路径是：

- 示例通过 `import mulberry.nn` 使用 NN package。
- `stdlib/mulberry/nn.cherry` 里只声明普通 extern package function。
- `packages/nn` 里提供 `mulberry_nn` dialect、op、pass 和 lowering。
- Sema、MLIRGen 和 LowerMulberry 不再硬编码 NN primitive。

当前 `MulberryNNPackage` 已经能通过 package pass 处理 source-level
`nn.matmul(...)` 这类普通 package call，并把最终的 `mulberry_nn.*` op lower 到
`linalg` / `arith` / `math`。driver 通过 bundled package registry 检测
`import mulberry.nn`，自动加载 `MulberryNNPackage` 并接入 NN package pipeline，
所以 inference 示例可以作为普通 `mulberry-driver file.cherry` 的默认 JIT smoke。

`matmul` 和 `transpose` 当前只实现 rank-2 tensor，刚好覆盖 Nielsen MNIST
推理/训练需要的路径。后续如果要支持更通用的 broadcasting、batch matmul 或
GPU backend，应该继续放在 `mulberry.nn` package 边界后面，而不是塞回
Sema、MLIRGen 或 LowerMulberry。
