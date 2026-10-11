# 神经表征流形动力学：第一阶段数据

已实际训练并采集，主数据可用于构造候选大流形。**这是作者完整规模架构的本地训练变体，不是论文的严格复现；环面和统一大流形尚未证实。** 激活由实际 RNN 运算产生；运动轨迹为模拟数据。

作者仓库 main/v1.0、发布附件和 Zenodo 元数据中未发现公开权重或轨迹数组。直接导入作者未修改的 `SorscherRNN`、`PlaceCells` 和损失函数。完整规模为 4096 隐藏单元、512 位置细胞；本轮学习率 1e-3、递归 L2 1e-6、单环境、4750 个批次，区别于论文的 1e-4、1e-4、主实验三环境和每环境 100000 批次。缺失的默认位置细胞布局改用固定种子均匀布局，实际中心另存。另保留成功的 256 单元试验及未学会任务的短训练试验。

## 主要文件

- [数据清单](datasets/full4096_fast/manifest.json)：72 个无损 gzip HDF5 分块，共约 2.11GB；每个文件包含完整 H、输入、位置、速度、位置细胞目标、模型 logits、输出概率和逐步误差。
- [核心 HDF5 索引](datasets/full4096_fast/core_T20.h5)：`H[21,8192,4096]`，第 0 步为实际线性编码初态；静止、匀速、闭合圆路径、随机运动各 2048 条，配对使用 2048 个不同初始位置。
- [探索 HDF5 索引](datasets/full4096_fast/exploratory_T160.h5)：`H[161,1024,4096]`；核心轨迹的分散空间子集，前 21 个状态完全对应。40、80、160 步是同一连续运行的三个前缀视图，均标为训练时域外。
- [样本信息](datasets/full4096_fast/samples.csv)、[全部计划轨迹](datasets/full4096_fast/trajectory_plan.npz)、[权重和优化器状态](runs/full4096_fast/last.pt)、[配置](runs/full4096_fast/config.json)、[位置细胞中心](runs/full4096_fast/place_cell_centers.npy)。配置、检查点包含随机种子与随机数生成器状态。
- [验证结果](datasets/full4096_fast/verification.json)、[机器可读结果](RESULTS.json)、[质量检查](analysis/full4096_fast/full_population_quality.json)、[统一 3D 投影](analysis/full4096_fast/shared_pca_3d.html)、[静态图](analysis/full4096_fast/shared_pca_3d.png)、[所有文件的 SHA256](SHA256SUMS)。

HDF5 索引使用相对路径引用分块；搬运时保留整个 `datasets/full4096_fast` 目录。原始数组为 float32，全部神经元始终保留。`sample_id` 是跨文件、跨步骤对应的依据；扩展的列号与核心列号不同，须按 ID 对齐。

## 已测结果与限制

独立 1024 条随机运动轨迹：平均定位误差 **8.70cm**，第 20 步 **5.95cm**；初位不动基线 19.94cm、零输入 23.21cm、打乱输入 29.88cm，位置细胞解码误差下限 4.26cm。全部分块、全部隐藏状态及读出重放通过；在本机固定环境下，独立递归计算和跨时域前缀最大差为 0。

第 20 步，四条件在 20×20 空间网格的覆盖率为 96%–99%。但约 **97.8% 激活值为零，76%–78% 神经元在该条件的全部样本上恒为零**；没有整条神经状态全零的样本，仍有跨位置的表征差异。应把强稀疏及大量不活跃单元作为研究限制，不能据此宣布健康的环面结构。160 步终点误差分别约 16.4、64.3、17.1、24.2cm，不能宣称长期稳定积分。

3D 图只使用一套合并 PCA 均值和基向量，前三轴解释约 26.1% 方差。页面分别呈现固定步骤的样本集合及单条轨迹；原始空间的配对变化另存。未分别拟合各步骤 UMAP。内在维度和 H0/H1/H2 持久同调只做初筛，未建立环面证据。

## 复算

Python 3.11；安装 [固定依赖](requirements.txt)，CUDA torch 使用 `https://download.pytorch.org/whl/cu128` 作为额外索引。当前实测 Python 3.11.9、torch 2.7.1+cu128、RTX 5060 Laptop 8GB；环境详情在 `runs/full4096_fast/environment.json`。

```powershell
# 在 outputs 目录；原始 HDF5 读取只需要 numpy 和 h5py
python code/read_data.py
python code/phase1.py verify --run full4096_fast
python code/analyze.py --run full4096_fast

# 从固定种子重训到同一预先定义的验证门槛；使用新名字保留已交付模型
python code/phase1.py train --config configs/full_fast.json --steps 10000 --run-name full4096_recomputed
python code/phase1.py collect --run full4096_recomputed --per-condition 2048 --ext-per-condition 256 --chunk 128
python code/phase1.py index --run full4096_recomputed
python code/phase1.py verify --run full4096_recomputed
```

跨设备或库版本允许 float32 数值差异；哈希验证当前交付文件的完整性，不承诺跨平台重训文件字节相同。模型只在初态接收位置编码，此后只接收位移输入，没有逐步位置提示或状态重置。作者把已经乘 dt 的位移命名为 `velocities`；这里 `model_input` 单位为米/步，`velocity_m_s=model_input/0.02`。第 0 步无运动输入，其读出未作为训练目标。

来源：[论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10589895/)、[作者代码](https://github.com/bioAI-Oslo/emergent-grid-cells)、[轨迹模拟器](https://github.com/bioAI-Oslo/rat-simulator)。源码提交、发布核查、缺失文件和方法差异见 [来源审计](audit/source_availability.json)，进一步说明见 [数据方法](METHODS.md)。本轮未拟合 Ricci Flow、测地流、Hamiltonian 或惯性算子。
