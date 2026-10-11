# 人工 RNN 的环面拓扑验证

**得到的是有条件的环面证据：公开 Sorscher RNN 的 483 单元子群，在静止输入、20 步终态时，原始激活空间出现与二维环面一致的持久拓扑。随机运动原始状态和全部 4096 单元整体均未通过同样验收。** 不宣称严格复现 Schøyen 三环境实验，也不宣称与生物机制等价或已证明数学同胚。

| 独立留出集 | 初始位置数 | PH 代表点 | 两个 H1＋一个 H2 | 原检验 Holm p | 999 次精度复核，保守 10 倍 p |
|---|---:|---:|---|---:|---:|
| 验证 | 8192 | 768，1024 敏感性通过 | 持久 `(1,2,1)`，域 47／2 一致 | 0.05 | 0.01 |
| 确认 | 8192 | 768，1024 敏感性通过 | 持久 `(1,2,1)`，域 47／2 一致 | 0.05 | 0.01 |

两组的 H2 归一化持久长度为 0.304／0.343，`(1,2,1)` 区间长度为 0.116／0.105。999 次独立随机化均无超越观测统计量，单次检验 p=0.001；真实尾概率的单侧 95% 上限约 0.003。静止的 IID TwoNN 为 2.33／2.36，kNN 小尺度估计约 2.02；随机运动估计约 4.9–5.3。不是只凭 PCA/UMAP 图片判断。

## 交付

- [完整结果](RESULTS.json)、[统计表](STATISTICAL_SUMMARY.csv)、[原始 PH 证据图](figures/raw_torus_evidence.png)及 SVG。
- [模型](models/)：公开完整推理权重及转换版本；本地旧模型作为重新封装的失败基线，含完整训练检查点。两个公开项目的示例权重相同，不计作独立训练种子。公开权重没有优化器、原训练种子和历史预算，不假称完整训练状态。
- [原始张量](data/)：575 个分块，始终保留全部 4096 单元、初态编码、位置与完整 20 步输入历史。各实验实际记录步骤见 manifest：发现集存末 10 步及初态，验证／确认按实验存 0–20、0／20 或终态，不能混淆轴。
- [逐步完整 H 索引](data/official_ganguli_pretrained/timewise/H_global.h5)：`H[21,2048,4096]` 和固定子群 `candidate_H[21,2048,483]`；静止／随机运动各 1024 条、配对初始位置。索引使用相对链接，搬运时保留分块目录。
- [统一坐标 3D 页面](figures/matched_step_projection.html)、[静态科研图](figures/matched_step_projection.png)：同一发现集 PCA 基和相同样本 ID 用于全部步骤，前三轴解释约 41.8% 方差。仅为投影。
- [源码](code/)、版本化协议 `protocol*.json`、[方法与失败记录说明](METHODS.md)、[哈希清单](SHA256SUMS)。所有 575 块均通过哈希、有限值、输入对齐和全部隐藏状态重放；本机 H 最大重放差为 0，读出差约 1.37e-6。

## 防止选择偏差

原论文的三个手选簇编号不可直接移用。第一版宽泛 Fourier 群在原空间失败；第二版只按发现集旋转聚类、条带方向与频率一致性选择，得到唯一的三个合格簇，483 单元。低采样数的失败保留。最终使用未见过的两组新样本确认，掩码及判据均冻结；十个来源／方法／条件尝试统一做 Holm 校正。发现集、验证集、确认集以整条轨迹或配对初始位置组分离，未把同一轨迹的不同步骤拆成“独立”样本。

## 复算

安装 Git LFS 后克隆完整数据。Python 3.11，CUDA 版 torch 2.7.1；实测 RTX 5060 Laptop 8GB。安装 `requirements.txt` 时使用 `https://download.pytorch.org/whl/cu128` 作为额外索引。旧第一阶段目录不再需要。

```bash
git lfs install
git clone https://github.com/Andante-Assai/neural-manifold-dynamics-phase1.git
cd neural-manifold-dynamics-phase1/phase2_delivery
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt
python code/verify_data.py
python code/rebuild_cache.py
python code/v3.py test --tag independent_repeat
```

最后一条在新目录保存复算统计，不覆盖原结果。来源：[Sorscher 官方仓库](https://github.com/ganguli-lab/grid-pattern-formation)、[Schøyen et al. (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10589895/)、[生物网格细胞环面研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC8810387/)。未重新训练新随机种子模型；已验证的公开模型足以回答本轮的条件性存在问题，不能将数据采样种子冒充训练种子。本轮未拟合测地流或其他几何动力学算子。
