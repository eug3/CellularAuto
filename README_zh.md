**English version: [README.md](README.md) | 以下是简体中文版。**

# 生长型神经元胞自动机（PyTorch 实现）

本仓库是对 **Mordvintsev、Randazzo、Niklasson、Levin（2020）**《
[Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/)》一文的
忠实 PyTorch 复现。

每个细胞携带一个 16 通道的状态向量（RGBA + 12 个隐藏通道）。一个极小的可学习
"基因组"（约 8 300 个参数）—— 两层 1×1 卷积网络 —— 描述了每个细胞如何根据其
3×3 邻域更新自身。通过 BPTT（沿时间反向传播）端到端训练，这条更新规则学会了
**从单个种子像素长出指定图案**、**将其稳定下来**（持久吸引子），并在加入损坏
训练后能够**在被擦除后再生**。

本项目提供一个命令行入口（`main.py`）和一个**完全用 tkinter 写的 GUI**，
用于训练和实时观察生长过程。无需网页、无需 Notebook，只依赖 Python 标准库
+ PyTorch/Numpy/Pillow。

---

## 特性
- **可对任意图片训练**（PNG / JPG / WEBP / BMP / GIF）：emoji、Logo、涂鸦都可。
- **tkinter 实时动画查看器**：播放 / 暂停 / 单步 / 重置，以及"感知场旋转"滑块
  （对应论文 Experiment 4，不必重训就能得到旋转后的图案）。
- **交互式再生**：在画布上按住左键拖动擦掉一部分图案，观察其自我修复（需选择
  *regen* 训练选项，对应论文 Experiment 3）。
- **样本池训练**（Experiment 2，论文中最经典的"会自己长出来"结果）。
- **带损坏数据增强的训练**（Experiment 3，真正的再生能力）。
- **设备**：渲染过程在 Apple Silicon MPS 或 CPU 上运行；训练默认用 CPU
  （该模型很小、CPU 上数值更稳定 —— 详见下文 *设备选择*）。

---

## 安装

```bash
cd CellularAuto
python3.12 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> 需要 **Python 3.12**（或任何有 torch 预编译 wheel 的 3.x 版本）；目前 Python
> 3.13 / 3.14 还没有 PyTorch wheel。依赖只有 `torch`、`numpy`、`Pillow`，
> tkinter 是 CPython 自带。
>
> macOS 上若 `import tkinter` 报错"无 _tkinter"，请 `brew install python-tk@3.12`。

仓库自带几张示例目标图（位于 `targets/`）：红心、笑脸、蜥蜴、圆环。

---

## 快速上手

### 1. 启动 GUI
```bash
python main.py
```
然后：
1. 点击 **Open image…**，例如选 `targets/heart.png`。
2. （可选）勾选 **train with damage (regen)** 以训练出能再生的模型。
3. 点击 **Start training**。训练在后台线程进行，进度条会显示当前 epoch。
   用默认的 4000 epochs 在 CPU 上约需 10–15 分钟，结果图案相当完整。
4. 训练完成后，按 **Play** 即可观看从中心种子长成完整图案的动画。

### 2. 命令行训练（不开 GUI）
```bash
python main.py --image targets/heart.png --train --epochs 4000
# 权重 -> models/heart.pth   损失曲线 -> models/heart_loss.json
```
加 `--regen` 即可进行损坏增强训练（Experiment 3）。

### 3. 用已训练好的模型批量导出生长动画帧（不开 GUI）
```bash
python main.py --image targets/heart.png --play \
               --model models/heart.pth --steps 96 --out out_frames
```

---

## 它是怎么工作的

```
状态 [B, 16, 72, 72]
   │ perceive：用 [identity, Sobel-x, Sobel-y] 三个核做 depthwise 3×3 卷积
   ▼
感知向量 [B, 48, 72, 72]
   │ UpdateCNN：1×1 卷积 (48→128) + ReLU → 1×1 卷积 (128→16)
   │            （末层权重初始化为 0 → 一开始什么都不变，do-nothing）
   ▼
更新量 Δ [B, 16, 72, 72]   × 每格的随机掩码  (fire_rate = 0.5)
   │ + 活细胞掩码   （仅当更新"前"且更新"后"的 3×3 最大池化 α > 0.1 才存活）
   ▼
新状态
```

**持久化训练**（Experiment 2）：
- 维护一个 1024 大小的样本池，初始全是单种子细胞。
- 每步取出一批 8 个样本，按"前向滚动前"的 loss 从高到低排序，把 **loss 最高
  的那一条替换为全新种子**（"重新播种"技巧，避免模型走捷径把所有图都变成零）。
- 滚动一个 `[64, 96)` 区间内的随机步数。
- 损失 = RGBA 4 个通道与（按 α 预乘的）目标之间的 MSE。
- 对**每个参数张量独立做**梯度 L2 归一化：`g ← g / (‖g‖ + 1e-8)`，防止后期
  loss 突跳（论文 footnote 4 提到的训练不稳定）。
- 优化器 Adam，`lr = 2e-3`，到第 2000 步后衰减为 `2e-4`。

**再生训练**（Experiment 3）：每步将 batch 中 loss 最低的 3 个样本随机擦掉
一个半径 `r ∈ [0.1, 0.4]` 的圆盘（归一化坐标）。学到的动力学随即在目标周围
形成一个更宽的吸引盆，从而能从各种损坏中恢复。

**旋转**（Experiment 4）：在每次卷积前先把 Sobel-x / Sobel-y 感知核绕原点旋转
一个角度 θ，**同一组权重**就能让图案按任意角度生长 —— 无需重训。GUI 中对应
"Rotate"滑块。

---

## 设备选择

|              | 训练                       | 渲染（GUI / --play）            |
|--------------|---------------------------|---------------------------------|
| Apple Silicon| **CPU**（默认，稳定）     | MPS（很快）或 CPU               |
| NVIDIA       | CUDA                      | CUDA                            |
| 其它         | CPU                       | CPU                             |

**为什么 Apple Silicon 上训练也用 CPU？** 当前 PyTorch（2.x）对沿长 CA rollout
（`[64, 96)` 步）反向传播在 MPS 后端上存在数值不稳定问题：训出来的模型会塌缩成
"杀手规则"，让种子在数步内 α 归零、整体消失（论文 footnote 4 所说的训练不稳定
在 MPS 上被放大）。而 CPU 上稳定，模型又很小（约 8 K 参数），CPU 训练每 4000 步
只需几分钟就够了。

可用 `--device` 标志覆盖（`auto` / `mps` / `cuda` / `cpu`）；该参数控制的是
**渲染**设备。若想强行让训练也尝试 MPS，请在 `main.py` / `gui.py` 调用 `train()`
处把 `force_cpu=True` 改成 `False`（不建议，会大概率得到不成长或死亡的模型）。

---

## 文件结构
```
CellularAuto/
├── ca_model.py     # CAModel: perceive + UpdateCNN + 活体掩码 + 随机更新
├── image_utils.py  # 目标图加载、种子、显示、损坏掩码、设备选择
├── train.py        # SamplePool、train_step、训练主循环、保存/加载
├── gui.py          # tkinter 应用: 查看器 + 训练线程 + 擦除 + 旋转
├── main.py         # 命令行入口（GUI / --train / --play）
├── requirements.txt
├── targets/        # 自带示例目标图
├── models/         # 训练好的 .pth 权重存放在此
└── .temp/          # 调试快照与临时文件（已加入 .gitignore）
```

---

## 已自带 demo 模型
仓库 `models/` 里附带两个已训练好的 demo 模型，直接加载即可看动画：

| 模型 | epochs | 最终 loss | 96 步生长细胞数（目标） |
|---|---|---|---|
| `heart_demo.pth` | 600 | 0.0025 | 5 → 107 → 264 → 508 → **487**（目标 392）|
| `lizard_demo.pth`| 400 | 0.0039 | 5 → 80  → 189 → 364 → **373**（目标 341）|

两者质心都精准落在网格中心 (36, 36)，呈现典型的"过冲 → 稳定"持久吸引子行为。
打开 GUI 后从 **Saved models** 下拉里选一个、点 **Load selected model**、再点
**Play** 即可。

---

## 提示与常见问题
- **最好选哪些图**：色彩有限、轮廓清晰、四周留白的图标型图——emoji、Logo、
  卡通形象。照片几乎不可能干净收敛。
- **loss 降不下来 / 图案不生长**：增加 `--epochs`。论文推荐 8000 步；≥2000 步
  基本就能看到不错的结果。
- **图案长出来后会爆炸式增长**：这是经典 CA 不稳定。请确认用的是默认的样本池
  持久化训练（默认就是），并加长训练；或暂不使用 `--regen`。
- **拖动擦除后没有自我修复**：当前加载的模型未用损坏数据训练过，请勾选 *regen*
  重新训练（或命令行 `--regen`）。
- **电脑装的是 Python 3.14，`pip install torch` 失败**：请用 3.12：
  `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`。
- **导入 `tkinter` 失败**：`brew install python-tk@3.12` 后重试。

---

## 参考文献
- Mordvintsev, Randazzo, Niklasson, Levin. *Growing Neural Cellular Automata*.
  Distill, 2020. DOI [10.23915/distill.00023](https://doi.org/10.23915/distill.00023)。
- 论文官方 TensorFlow 参考实现：
  <https://github.com/google-research/self-organising-systems/blob/master/notebooks/growing_ca.ipynb>
