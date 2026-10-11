# 神经表征流形动力学 · 第一阶段

实际运行的完整神经激活演化数据：作者完整规模架构的本地训练变体，不能视为严格论文复现。

ZIP 内的全部文件夹与文件原样位于 [phase1_delivery/](phase1_delivery/)，含完整原始张量、模型权重、代码、配置、随机种子、验证结果、SHA256 和统一坐标系 3D 投影。

- 核心：`H[21, 8192, 4096]`，初态加 20 个递归步骤，四类受控运动。
- 探索：`H[161, 1024, 4096]`，与核心按 sample_id 配对；40/80/160 步为训练时域外探索。
- 独立测试平均定位误差 8.70cm，终点 5.95cm；完整重放验证通过。
- 激活强稀疏，长时域性能下降；未建立环面或统一大流形存在性的证据。

## 获取完整数据

大型 HDF5 和模型权重使用 Git LFS，安装 Git LFS 后克隆会取得原始文件。没有 Release 附件。

```bash
git lfs install
git clone https://github.com/Andante-Assai/neural-manifold-dynamics-phase1.git
cd neural-manifold-dynamics-phase1
git lfs pull
```

[完整说明和复算命令](phase1_delivery/README.md) · [数据方法](phase1_delivery/METHODS.md) · [数据清单](phase1_delivery/datasets/full4096_fast/manifest.json) · [结果](phase1_delivery/RESULTS.json)

交互图 `phase1_delivery/analysis/full4096_fast/shared_pca_3d.html` 可在本地浏览器打开；仅为统一 PCA 投影，非真实流形形变证据。

来源：[Schøyen et al. (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10589895/) · [作者 RNN 代码](https://github.com/bioAI-Oslo/emergent-grid-cells)。原作者 MIT 许可证保留在交付目录。
