# BirdCLEF+ 2026 — Silver Medal Solution 🥈

**Kaggle Public LB: 0.948 | Silver Medal**

English README: [README_EN.md](README_EN.md)

---

### 比赛背景

[BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026) 是一项聚焦于**被动声学监测（PAM）**的 Kaggle 竞赛。核心目标是利用机器学习从连续音频中识别南美洲巴西潘塔纳尔湿地的野生鸟类物种，推动生物多样性监测与生态保护。赛方提供了覆盖全巴西潘塔纳尔湿地 1,000 台声学记录仪的连续录音数据集。

**关键限制条件：**
- 输入：5秒窗口连续音频，多物种声景
- 评估：跳过无真实正样本类别的宏平均 ROC-AUC
- 提交：仅限 Notebook，**仅 CPU，90分钟时限** → 模型须转换为 ONNX 或使用轻量架构

### 核心挑战

- **长尾分布**：234个物种，大量物种训练样本极少
- **嘈杂声景**：多物种叫声重叠，混有风声、雨声、虫鸣
- **推理无 GPU**：所有重型模型必须蒸馏或导出为 ONNX
- **序列上下文**：单个5秒窗口信号不足，跨窗口的时序上下文至关重要

### 解决方案总览

我们的方案是**双管道架构**，融合两个互补模型，并配合多项后处理策略。

**最核心的优化**：重新训练 SED 模型，用新训练的权重替换 baseline 的 `.pt` 文件，这一改动贡献了绝大部分分数提升。

```
音频（5秒窗口）
        │
        ├── 管道一：ProtoSSMv5（序列感知，基于 Perch 嵌入）
        │       └── 预测矩阵（N窗口 × 234类）
        │
        └── 管道二：Distilled SED（逐帧 CNN，EfficientNet-B0）
                └── 预测矩阵（N窗口 × 234类）
                        │
                基于排名融合：0.60 × ProtoSSM + 0.40 × SED
                        │
                后处理（阈值、平滑、镜像）
                        │
                submission.csv
```

### 管道一 — ProtoSSMv5（序列模型）

利用冻结的 **Google Perch v2** 鸟声嵌入模型提取1536维特征向量。

- **Perch 嵌入**：对所有训练声景提取（59个完整标注文件，708个窗口）
- **PCA压缩**：1536维 → 64维（保留81.47%方差）
- **MLP探针**：针对63个类别分别训练 MLP 分类器
- **ResidualSSM**：轻量级状态空间模型（d_model=128，d_state=16，2层），在 MLP 输出基础上学习序列校正
  - OOF 搜索最优校正权重：`0.40`（OOF macro-AUC = 0.99402）
- **物种映射**：234类中203类有 Perch logit；31个未映射物种通过属级代理映射处理
- 训练时间：约12.7秒（ProtoSSM）+ 1.6秒（ResidualSSM）

### 管道二 — Distilled SED（CNN 模型）

基于 EfficientNet-B0 的声音事件检测模型，通过 Perch v2 **师生蒸馏**训练，以 ONNX 格式在 CPU 上高效推理。

**模型架构：**
- 骨干网络：`tf_efficientnet_b0.ns_jft_in1k`，输入梅尔频谱图（256 mels，n_fft=2048，SR=32kHz）
- **GeM 频率池化**：可学习广义均值池化（初始 p=3.0），聚焦鸟类相关频段
- **注意力瓶颈**：512维全连接 → 1D 卷积 → 逐帧注意力权重 → 片段级预测
- **Perch 蒸馏头**：GAP + 线性映射，将 EfficientNet 特征投影到 Perch 的1536维空间，实现知识迁移
- **10折集成**：`sed_fold0.onnx` 至 `sed_fold9.onnx`，推理时取平均

**数据增强策略：**
- 信号级：随机增益抖动（±6 dB）、噪声注入（SNR 10–30 dB）
- SpecAugment：频率与时间掩码（GPU 上执行）
- Focal-Focal MixUp：两条焦点录音的 Beta 分布混合
- Focal-Soundscape MixUp：将焦点叫声注入标注声景背景
- 稀有物种动态上采样：每折每类至少保证20个样本

### 优化策略详解

**1. 模型集成与动态加权**
结合 ProtoSSM 序列模型与 SED 检测模型的预测，通过动态权重（xSED）平衡分类与检测优势。已映射物种（有 Perch logit）权重为 0.60/0.40；未映射物种 Proto 分量权重为 0.35。

**2. 时序平滑与上下文增强**
利用 `gaussian_filter1d` 对相邻时间窗口的预测结果进行加权平滑，消除孤立帧的误报，增强持续鸣叫事件的连续性。

**3. 元数据与先验知识注入**
从文件名（`BC2026_Train/Test_{id}_{site}_{date}_{time}.ogg`）解析站点和时刻信息，注入物种分布先验，对特定地点/时间的物种出没概率进行校准。

**4. 自适应阈值优化**
针对不同物种采用动态阈值替代全局固定阈值，44个稀有物种基于 OOF 预测校准（平均阈值0.464，范围[0.20, 0.50]）。

**5. 同型音镜像（Sonotype Mirroring）**
对10个关联物种对共享预测信号，处理物种间共鸣或误识别的情况。

**6. 分类层级平滑**
在属/科层级对预测结果进行后处理平滑，提升训练数据稀少物种的鲁棒性。

### 所需基础知识

- **音频深度学习**：Mel-Spectrogram 与 MFCC 的基本原理；频谱参数调优以提升 GPU 处理效率
- **被动声学监测（PAM）**：声音时间序列分析；多通道音频的动态上下文处理
- **集成学习**：K折交叉验证与多折集成提交策略
- **Kaggle 离线竞赛规范**：将所有依赖（Python Wheels）打包进 Dataset，确保代码在断网隔离环境下稳定运行

### 文件说明

| 文件 | 说明 |
|------|------|
| `birdclef-2026.ipynb` | 主推理 / 提交 Notebook |
| `bc2026-distilled-sed.ipynb` | SED 模型训练 Notebook（从头重训） |
| `utils_all.py` | 推理 Notebook 所调用的工具函数 |

### 运行方法

1. 将 `birdclef-2026.ipynb` 和 `utils_all.py` 上传到 Kaggle Notebook。
2. 挂载所需数据集：
   - SED ONNX 折叠模型（`sed_fold0.onnx` … `sed_fold9.onnx`）
   - Perch v2 ONNX 模型（`perch_v2_no_dft.onnx`）
   - Perch 嵌入缓存（`full_perch_meta.parquet`、`full_perch_arrays.npz`）
3. 加速器设置为 **CPU**（比赛规则要求）。
4. 运行全部单元格，90分钟内生成 `submission.csv`。

如需从头训练 SED 模型，使用 `bc2026-distilled-sed.ipynb`（建议使用 GPU）。

### 参考资料

- Baseline：[BirdCLEF-2026-v9 by raunakdey07](https://www.kaggle.com/code/raunakdey07/birdclef-2026-v9)
- SED 训练 Baseline：[bc2026-distilled-sed by tuckerarrants](https://www.kaggle.com/code/tuckerarrants/bc2026-distilled-sed)
- Google Perch v2：[bird-vocalization-classifier](https://www.kaggle.com/models/google/bird-vocalization-classifier)

---

## License

This project is released under the [MIT License](LICENSE).
