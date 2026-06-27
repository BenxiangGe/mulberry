# Mulberry NN Package

`mulberry.nn` 不再是 core compiler feature。它是一个独立 NN package：
源码层通过普通 import 使用它，MLIR 层由 package 自己提供 dialect、op、pass
和 lowering。

```mulberry
import mulberry.nn;

var z: Tensor<Float32> = nn.matmul(w, x);
```

`stdlib/mulberry/nn.mulberry` 现在只声明 `mulberry.nn.matmul`、
`mulberry.nn.sigmoid`、`mulberry.nn.argmax` 等 package 函数。Sema 和 MLIRGen
把它们当作普通 extern function；core lowering 不知道 NN primitive 的存在。

这样分层更干净：

- Sema 把 `nn.matmul` 当作普通 import function 来 typecheck。
- MLIRGen 只生成普通 `func.call`。
- Core lowering 不知道神经网络 primitive 的存在。
- NN package 自己注册 `mulberry_nn` dialect 和 `lower-mulberry-nn` pass。
- source-call 到 `mulberry_nn.*` op 的 rewrite 放在 NN package pass 里，
  不塞回 core compiler。

当前实现状态：

- `packages/nn` 已经能编译出 `MulberryNNPackage`。
- plugin 可以通过 `--load-dialect-plugin` 注册 `mulberry_nn` dialect。
- plugin 可以通过 `--load-pass-plugin` 注册 `prepare-mulberry-nn-calls`
  和 `lower-mulberry-nn` pass。
- `prepare-mulberry-nn-calls` 先把 source-level
  `func.call @mulberry.nn.*` 从 `Tensor<Float32>` header 形式桥到 private
  `@mulberry.nn.__tensor.*` tensor-form call。
- `lower-mulberry` 把 core Tensor/object IR lower 成 memref 以后，再跑一次
  `prepare-mulberry-nn-calls`，把 `@mulberry.nn.__tensor.*` memref-form call
  改写成 `mulberry_nn.*` op。
- `lower-mulberry-nn` 最后把 `mulberry_nn.*` op lower 到
  `linalg` / `arith` / `math`。

手动测试 pipeline：

```text
prepare-mulberry-nn-calls,
lower-mulberry,
prepare-mulberry-nn-calls,
lower-mulberry-nn
```

driver 已经通过 bundled package registry 检测 `import mulberry.nn`，自动加载
`MulberryNNPackage`，并把这段 package pipeline 接入默认 lowering/JIT 路径。
这个 registry 只知道 package 名、动态库路径和 pipeline 名，不知道 `matmul` /
`sigmoid` 这些具体 primitive。使用 `import mulberry.nn` 后，`nn.matmul(...)`
可以走普通 `mulberry-driver file.mulberry` 正向路径。
