# Weft source corpus

`source/` 保存端到端编译使用的输入源码，不参与根 CMake 的默认构建。目录按 source
language、upstream/project、operator 和 variant 组织，同一变体的 source 与 runtime 相邻。

```text
source/
├── weft/weft/<operator>/<variant>/
└── c/
    ├── weft/<operator>/<variant>/
    └── ggml/quantization/block_dot/<format-pair>/
```

GGML-derived source 的许可和抽取边界见 [`c/ggml/README.md`](c/ggml/README.md)。
