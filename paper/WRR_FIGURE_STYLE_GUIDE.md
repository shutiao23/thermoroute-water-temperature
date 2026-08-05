# WRR 画图文档手册（Water Resources Research 图件风格手册）

**文档角色：** 基于 8 张 WRR 已发表论文图件逐图拆解后提炼的画法/配色/用途手册，
供本项目（ThermoRoute）WRR 投稿图件绘制时复用。样本来源：

- Read, J. S., et al. (2019). *Process-Guided Deep Learning Predictions of Lake Water
  Temperature.* WRR. https://doi.org/10.1029/2019WR024922
- Topp, S. N., Barclay, J., Diaz, J., Sun, A. Y., Jia, X., Lu, D., Sadler, J. M., &
  Appling, A. P. (2023). *Stream Temperature Prediction in a Shifting Environment:
  Explaining the Influence of Deep Learning Architecture.* WRR 59, e2022WR033880.
  https://doi.org/10.1029/2022WR033880

**权威链（冲突时自上而下）：**

| 层 | 文件 | 管什么 |
|---|---|---|
| 1 | `paper/FIGURE_REDRAW_SPEC.md` | 语义配色 token、尺寸/字号、value ID 绑定、gate、禁用语义、导出与验收 |
| 2 | 本手册 | 视觉风格、图型选择、可复现模板参数 |
| 3 | `docs/WRR_FIGURE_STYLE_TEARDOWN.md` | 样本原图逐图拆解、核验状态、勘误表 |

本手册**不定义**硬指标；数值一律以 spec §3.2/§3.4 与出图脚本常量为准。
样本描述与原图冲突时，以 `docs/WRR_FIGURE_STYLE_TEARDOWN.md` §5 勘误表为准。

**投稿硬指标基线：** 本项目 AGU 模板已承诺值 = 全宽 140 mm（`agujournal2019.cls`
`\textwidth 5.5in` = 139.7 mm）/ 单栏 85 mm，正文/轴文字 ≥8 pt、最小 7.5 pt，
矢量 SVG/PDF 优先，栅格 ≥300 dpi、线画 ≥600 dpi。

AGU 官方 Graphics Requirements 页面抓取受限（403），以下为检索核实的公开指引值，
**提交前须以编辑部最新版本复核**：单栏 50–85 mm、双栏 105–170 mm、最大高度 228 mm、
图内最小字号 8 pt（终稿尺寸）。

> **2026-08-05 修订**：单栏由 88 mm 改为 85 mm（spec §3.2）。88 mm 超出 AGU 单栏上限，
> 会在生产环节被缩至 96.6%，使 7.5 pt 刻度实际印为 7.24 pt，击穿 spec 自身的字号下限。
> spec §3.2 的 180 mm 导出变体虽超双栏上限 170 mm，但已限定为"仅在出版社要求时"，措辞无需修订。

---

## 1. 样本图总览与风格共性

8 张样本图覆盖 6 大图型：

| # | 来源 | 图型 | 一句话用途 |
|---|------|------|-----------|
| 1 | 2019 Fig1 | 泳道式概念框架图 | 对比 PB 与 PGDL 两类模型的结构、数据流与概念耦合 |
| 2 | 2019 Fig2 | 点-误差棒-虚线连接曲线 | 训练数据量敏感性实验（小样本优势） |
| 3 | 2019 Fig3 | 多面板点-误差棒（marker 内嵌字母） | 不同训练/测试划分下的稳健性 |
| 4 | 2019 Fig4 | 小提琴 + rug + 中位线 | 68 个湖的跨站点性能分布对比 |
| 5 | 2019 Fig5 | 同色系空心/实心点-误差棒 + 竖直分栏 | 预训练数据量与多样性的消融 |
| 6 | 2022 Fig1 | 2×3 矩阵式架构概念图 | 两模型的时序/空间感知机制差异 |
| 7 | 2022 Fig5 | ggplot2 分面水平条形图 | 泛化（地理/域偏移）性能变化 |
| 8 | 2022 归因图 | 分面时间序列 + ±1 SD 带 + 共享图例 | 可解释性：各输入变量的归因时序 |

两篇论文风格不同但都符合 WRR/AGU 规范，可视为两套可互换的"皮肤"：

- **皮肤 A（2019 论文，偏 `theme_classic`）**：纯白背景、只留左/下轴线、无网格、
  图例无边框、空心 marker、细虚线连接线。适合点-范围类实验图与概念图。
- **皮肤 B（2022 论文，ggplot2 `theme_bw` 系分面风）**：**白色面板背景 + 浅灰网格线**、
  灰色分面标题条（黑细边）、黑色面板边框、实心色块、右侧共享图例。适合多分面分类对比与归因图。
  （早期版本误记为"浅灰底 + 白网格 / `theme_gray`"，已按原图更正，见 `docs/WRR_FIGURE_STYLE_TEARDOWN.md` §5 E1。）

**跨图共性（真正要学的部分）：**

1. **颜色 = 角色**：同一模型/角色在全文所有图保持同一颜色（PB 绿、DL 橙、PGDL 紫、
   基线灰），读者跨图零成本对应。
2. **双/多重编码**：颜色之外再用 marker 形状（菱/方/圆）、填充（空心/实心）、
   内嵌字母（M/S）区分序列——色盲与黑白打印下仍可辨。
3. **误差轴反向**：RMSE 类"越小越好"的 y 轴一律反向（小值在上方），
   使"图上位置越高 = 越好"与直觉一致。
4. **不确定性必画且必说明**：误差棒（range 或 SE）、violin+rug、不确定性带三选一，
   caption 必须写明带/棒的确切统计含义（range / SE / ±1 SD / 分位，四者不可混称）、
   以及 marker 代表 mean 还是 median。样本 2022 Fig 8 的带是 **±1 SD**，不是置信区间。
5. **轴线做减法**：去掉顶/右 spine（皮肤 A）或弱化网格（皮肤 B）；同一论文内只用一套皮肤。
6. **分面而非堆叠**：实验维度（划分方式、季节、模型、偏移类型）用面板/分面拆开，
   每面板只回答一个问题；共享坐标轴与图例，避免重复。
7. **概念图用"泳道 + 列头 + 彩色箭头"**：结构分区用彩色边框泳道，流程列用灰色斜体
   表头，数据实体用黑框白底，概念耦合用强调色（品红）虚线框/箭头，反馈用点线。
8. **caption 长而完整**：`Figure N.` 加粗起笔，先给主题/结论，再逐一解释符号、颜色、
   误差、缩写与边界（与 spec §3.3 的五段式 caption 语法一致）。

---

## 2. 逐图拆解

### 图 1（2019 Fig1）——泳道式概念框架图

- **用途**：方法动机图。一张图同时讲清 (a) 两类模型各自的
  Inputs→Parameters→Computations→Predictions→Feedback 流水线，(b) 新概念如何嵌入
  （energy balance 虚线框）、(c) 两模型间的数据/概念迁移（品红箭头）。
- **布局**：上下两条泳道（上绿 = PB，下紫 = PGDL）；顶部 5 个灰色*斜体衬线*列头；
  左侧 90° 旋转的彩色斜体泳道名；泳道内部再分格（ Inputs | Parameters |
  Computations | Predictions | Feedback ）。
- **元素语义**：
  - 黑框白底矩形 = 数据实体（drivers、observations、P、y）；
  - 泳道色细实线箭头 = 模型内部数据流；
  - 品红（magenta）虚线框 = 被共享/迁移的物理概念；品红虚线箭头 = 概念耦合，
    品红实线箭头 = PB 输出作为 PGDL 预训练数据；
  - 点线（dotted）= 反馈回路（calibrate / pre-train / train），并用花括号把
    "预测 − 观测" 括起来接到反馈列；
  - 灰底块 + 旋转灰字（Calibrate）= 可选/本实验省略的步骤。
- **可复用模式**：`泳道(模型) × 列(流程阶段)` 矩阵；强调色只给"本文贡献的耦合"，
  其余全部降为黑/灰/泳道色，读者视线自然落在贡献上。

### 图 2（2019 Fig2）——数据量敏感性曲线

- **用途**：核心实验图。x = 训练剖面数（2…980，按类别等距排布的数量梯度），
  y = Test RMSE（反向轴），3 个模型 3 条序列。
- **画法要点**：
  - marker 空心（白底 + 彩色边），形状区分模型（菱 = PGDL、方 = DL、圆 = PB）；
  - 序列连接线用**同色细虚线**——弱化"线"、强调"点与误差"；
  - 误差棒 = 多次重复的 range，细实线、无帽或短帽；
  - 图例置于面板内左上空白处，无边框、字号略大于刻度；
  - 只留左/下 spine；背景近纯白。
- **可复用模式**：`空心 marker + 虚线连接 + range 误差棒 + 反向 y` 是 WRR 模型对比
  实验的"标准句式"，尤其适合 x 为样本量/站点数梯度时。

### 图 3（2019 Fig3）——多面板稳健性对比

- **用途**：同一组模型在 3 种训练/测试划分（相似期、冷年训暖年测、非夏训夏季测）下的表现。
- **画法要点**：
  - 3 个面板横排，面板标题 `a) Train & test similar` 式，左上、黑体无衬线；
  - x = 3 个模型类别；每类内并排两个 marker，**marker 内嵌字母** M/S 区分两个湖——
    省掉第二图例且不占 x 宽度；
  - y 轴反向、三面板共享，仅最左面板显示刻度标签；
  - 误差棒 = 5 次重复 range，marker = 均值。
- **可复用模式**：`面板 = 实验设定，x = 方法，marker 内嵌字母 = 数据集` 的三维对比压缩法。

### 图 4（2019 Fig4）——跨站点分布：violin + rug + 中位线

- **用途**：68 个湖的 RMSE 分布对比（4 种模型配方，含未率定灰基线）。
- **画法要点**：
  - 实心填色 violin（无边框），颜色沿用模型色，未率定基线用灰；
  - violin 内部黑色短横线 = 每个湖的个体值（rug/strip），长粗黑线 = 中位数；
  - y 反向（1 在上、5 在下）；x 类别标签自动换行两行。
- **可复用模式**：`violin(分布形状) + rug(原始点) + median(粗线)` 三件套，
  比箱线图信息量大、比散点云整洁，适合"多站点整体性能"图。
  注意：本项目 spec §3.2 倾向原始点/ECDF/箱摘要而非平滑 violin——若用 violin，
  需同时画出 rug 原始点以保留证据粒度，并在 caption 说明带宽不作密度声明。

### 图 5（2019 Fig5）——同族变体消融 + 分栏注释

- **用途**：预训练数据量/多样性对 PGDL 的影响（消融）。
- **画法要点**：
  - 同一紫色的两种填充：**空心菱 = extended、灰实心菱 = limited**——
    "同色不同填充 = 同模型族变体"的编码法；
  - x 标签带下标（sim₂、sim₁₀…seas₅₀₀）；
  - 一条细黑**竖直分线**把 x 分成"来自 Figure 2 的实验 / 来自 Figure 3 的实验"两组，
    分线两侧底部小字注释来源——把跨图实验串成一张总览；
  - y 反向，误差棒 = 5 次重复 range。
- **可复用模式**：消融图用"同色+填充变化"表达控制变量，用竖直分线+小字注释挂接其他图。

### 图 6（2022 Fig1）——2×3 矩阵式架构概念图

- **用途**：两模型（RGCN / GWN）× 三视角（时序感知 / 空间感知 / 整体结构）的差异总览。
- **画法要点**：
  - 行 = 模型（左侧 90° 旋转粗体行名），列 = 视角（顶部粗体列题），细灰线分格；
  - LSTM 单元 = **teal 粗圆角矩形**（全图唯一的饱和强调色）；
  - 激活函数 σ/tanh = 灰色小方块；内部数据流 = 细灰箭头；
  - 图结构 = 紫色圆点节点 + 细黑边，底衬**浅灰河网线稿**提供地理语境；
  - "放大"语义用细灰弯箭头从小河网图指向放大图。
- **可复用模式**：架构对比用 `行(模型) × 列(能力维度)` 矩阵；强调色只给核心模块，
  地理底图降到 10% 灰度当背景纹理。

### 图 7（2022 Fig5）——分面水平条形图（ggplot2 皮肤 B）

- **用途**：相对基线的 RMSE 变化（负 = 变差），2 种偏移 × 2 种指标分面。
- **画法要点**：
  - ggplot2 分面：灰色标题条（Overall / Warmest 10%），列上方外标题
    （Geographic Shift / Domain Shift），**白色面板 + 浅灰网格 + 黑面板边框**；
  - 水平分组条形，viridis 双色：GWN 深紫 `#440154`、RGCN 青绿 `#21918C`；
  - 误差须 = 标准误（SE），黑色；x = 0 处黑色竖直参考线；
  - 右侧共享图例（标题 "Model" + 色块）。
- **可复用模式**：`相对基线的变化量` 用水平条形 + 零线最直观；分面承载
  "场景 × 指标" 两个维度；条形方向让类别标签（Coastal/Appalachians…）水平可读。

### 图 8（2022 归因图）——分面归因时间序列

- **用途**：可解释性。两模型在 4 个季节上、各气象输入对预测的归因随序列日的变化。
- **画法要点**：
  - 顶行：累积归因曲线——**38 条半透明黑/灰线叠加**，深浅来自重叠，**无单独的总体线**；
    两模型序列长度不同（60 vs 180 天），x 轴范围不共享；
  - 主区 4×2 分面：行 = 季节（右侧灰条 DJF/MAM/JJA/SON），列 = 模型（顶部灰条）；
  - 每个输入变量一色：深蓝（气温）、浅紫（降水）、粉（潜在蒸散）、橙（短波辐射），
    **实线 = 均值 + 半透明同色带 = ±1 标准差**（原 caption 明写 mean and standard deviation，
    非置信区间——引用时不得改述为"置信带"）；
  - 每格左侧内嵌同色水平小条 = **各变量跨全部时间步的相对总贡献**（inset），非离散滞后项；
  - 底部共享图例：线段 + 半透明色块双元素 swatch。
- **可复用模式**：归因/敏感性时间序列 = `分面(季节×模型) + 变量配色 + 线+带 + 共享图例`；
  个体异质性用细灰线堆叠、总体用黑/深色线压顶。

---

## 3. 设计系统提炼

### 3.1 配色

**样本 A（2019 论文）≈ ColorBrewer Dark2**（色盲安全、中等饱和、学术感强）：

| 角色 | Hex | 样本中的用途 |
|------|-----|-------------|
| 绿 `green` | `#1B9E77` | 过程模型 PB |
| 橙 `orange` | `#D95F02` | 纯深度学习 DL |
| 紫 `purple` | `#7570B3` | 本文方法 PGDL |
| 灰 `gray` | `#666666` / 浅灰 `#999999` | 未率定/基线/有限预训练填充 |
| 品红 `magenta` | `#E7298A` | 概念耦合/贡献强调（仅概念图） |

**样本 B（2022 论文）≈ viridis 截取 + 低饱和多变量色**：

| 角色 | Hex（目测近似取色） | 用途 |
|------|--------------------|------|
| 深紫 | `#440154` | 模型 GWN |
| 青绿 | `#21918C` | 模型 RGCN |
| teal（强调） | `#007C7C` | 架构图核心模块（LSTM 圆角框） |
| 深蓝 | `#2B2BA3` | 归因：气温 |
| 浅紫 | `#C9A2DC` | 归因：降水 |
| 粉 | `#E593B2` | 归因：潜在蒸散 |
| 橙 | `#F5B26B` | 归因：短波辐射 |

**配色规则：**

1. 颜色绑定"角色/模型"，全文（主图 + SI）不变；新角色从同一色族扩展，不引入彩虹色。
2. 基线/参照永远灰；强调色（品红/teal）只给"本文贡献"，一张图最多一个强调色。
3. 任何序列必须有色外第二编码（形状/填充/线型/内嵌字母），红绿不单独表义。
4. 同模型族变体用"同色不同填充"（空心/实心/灰芯），不同模型才换色。

### 3.2 字体与文字

- 图内文字一律无衬线（Helvetica/Arial/DejaVu Sans）；概念图的列头/泳道名可用
  *斜体衬线*制造"注释层"与"数据层"的区分（样本 1 的做法）。
- 字号：轴标题/图例 8–9 pt，刻度 7.5–8 pt，面板标签 `(a)/(b)` 或 `a)` 9–10 pt 加粗，
  全图同一偏移位置。数学符号（ŷ、Δ、下标）用斜体/真下标，不用纯文本拼写。
- 面板标题简短名词短语（"Train: coldest years / Test: warmest years" 式两行亦可）。
- 轴标题带单位：`Test RMSE (°C)`。

### 3.3 轴、网格与布局

- 皮肤 A：`spines: 仅左+下`，无网格，白底；点-范围图、概念图用。
- 皮肤 B：ggplot2 分面灰条 + **白底 + 浅灰网格** + 黑边框；多分面分类图、归因图用。
  **一篇论文只选一套皮肤，且只用一个绘图工具**——样本自身 matplotlib 图（2022 Fig 3/9）
  与 ggplot2 图（Fig 4–8）的刻度、图例、字体渲染肉眼可辨不同，属其不足，不要照抄。
- "越小越好"的误差轴反向（`invert_yaxis`），并在 caption 或轴标题暗示方向语义。
- 数量梯度 x（2, 10, 50, …, 980）按**类别等距**排布即可，不必真 log 轴；
  若用 log 轴必须在面板与 caption 标注（spec §3.2）。
- 分面维度分配：`行 = 次要维度（季节/指标），列 = 主要对比维度（模型/偏移）`；
  共享 y/x 轴时仅边缘面板画刻度标签。
- 图例：皮肤 A 放面板内空白处、无边框；皮肤 B 放右侧/底部共享；能直接标注就不放图例。
- 参考线：0 线、名义覆盖率线、上限线必须画且与数据序列线型区分（细黑实线/长虚线），
  并在 caption 点名。

### 3.4 与本项目语义配色的协调（重要）

本项目 `FIGURE_REDRAW_SPEC.md §3.1` 已冻结 Okabe-Ito 系语义 token
（`TR_BLUE #0072B2`、`DAMPED_ORANGE #E69F00`、`LSTM_GREEN #009E73`、
`LGBM_PURPLE #CC79A7`、`WARNING_VERMILION #D55E00`、灰系基线等）。协调规则：

- **角色颜色以 spec token 为准**，不改用 Dark2/viridis（避免与已承诺的 TeX/SVG 产物冲突）。
- 从样本**借结构与表达**：反向误差轴、空心 marker+虚线连接、marker 内嵌字母、
  violin+rug+median、同色填充变体、竖直分线注释、泳道概念图、分面归因图等。
- 样本的"灰 = 基线、强调色 = 贡献"原则与 spec 一致（`PERSIST_GRAY/CLIM_GRAY` 基线、
  `ALLOWED_TEAL` 允许路径、`WARNING_VERMILION` 限制），直接沿用。
- 多变量归因/敏感性图若需 4 色以上，从 Okabe-Ito 全集扩展
  （`#56B4E9` 天蓝、`#F0E442` 黄，慎用）或 spec 已列 token，不新造彩虹。

---

## 4. 图型选择指南（场景 → 图型）

| 你要表达的场景 | 推荐图型（样本编号） | 关键参数 |
|----------------|---------------------|----------|
| 研究动机/方法框架/信息边界 | 泳道概念图（图1）或矩阵概念图（图6） | 泳道色=范式，黑框=数据，强调色虚线=贡献耦合，点线=反馈 |
| 样本量/数据量敏感性 | 点-误差棒-虚线连接（图2） | 空心 marker、range 棒、反向 y、x 类别等距 |
| 多种划分/场景下的模型对比 | 多面板点-误差棒（图3）或分面条形（图7） | 面板=设定，marker 内嵌字母=数据集 |
| 多站点/多湖整体分布 | violin+rug+median（图4） | 实心 violin、黑短横 rug、粗黑中位线 |
| 消融/控制变量 | 同色空心/实心 + 竖直分线（图5） | 填充编码变体，分线+小字挂接他图 |
| 相对基线的变化（泛化/偏移） | 分面水平条形 + 零线（图7） | 误差须=SE，0 参考线，负方向语义写明 |
| 归因/敏感性时间序列 | 分面线+带 + 细灰个体线（图8） | 行=季节/列=模型，底部共享图例 |
| 区间/可靠性 | （spec Fig3 已定）coverage-width 散点、reliability 图 | 名义线/对角线必画，bin 分母可见 |
| 配对效应/正式比较 | （spec Fig2 已定）forest + 全模型分布小倍数 | 0 线与 +0.05 上限线，无星号 |

ThermoRoute 主图映射建议：Fig1(a) 持久性动机可用图4式分布小倍数或图2式点图；
Fig1(b) 有界校正 = 锚线 + `A±δ` 阴影带（图8 的"线+带"句式）；Fig1(c) 证据脊柱 =
图1 的左→右流程链；Fig2 forest 保持 spec 设计；Fig4 敏感性点矩阵可用图3/图5句式。

---

## 5. 可复现模板参数

### 5.1 matplotlib（皮肤 A）

```python
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 8.5,
    "lines.linewidth": 1.2, "lines.markersize": 5.5,
    "errorbar.capsize": 1.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,   # 真文本嵌入
    "savefig.dpi": 600, "savefig.bbox": "tight",
})

WRR_DARK2 = {"ref": "#666666", "base": "#1B9E77", "dl": "#D95F02",
             "ours": "#7570B3", "accent": "#E7298A"}

# 反向误差轴 + 空心 marker + 虚线连接 + range 棒
ax.invert_yaxis()
ax.plot(x, y, ls="--", lw=0.8, c=c)                      # 弱连接线
ax.errorbar(x, y, yerr=..., fmt="o", mfc="white", mec=c,  # 空心 marker
            mew=1.2, ms=6, ecolor=c, elinewidth=0.9, capsize=1.5)
```

### 5.2 ggplot2（皮肤 B）

```r
theme_wrr <- theme_classic(base_size = 8) +
  theme(axis.line = element_line(linewidth = 0.4),
        legend.frame = element_blank(),
        plot.tag = element_text(face = "bold", size = 10))

# 分面皮肤 B —— 样本实际为 theme_bw 系：白面板 + 浅灰网格（不是 grey95 底 + 白网格）
facet_skin <- theme_bw(base_size = 8) +
  theme(strip.background = element_rect(fill = "grey85", color = "black", linewidth = 0.4),
        panel.background = element_rect(fill = "white", color = NA),
        panel.grid.major = element_line(color = "grey92", linewidth = 0.3),
        panel.grid.minor = element_blank(),
        panel.border     = element_rect(fill = NA, color = "black", linewidth = 0.4))

scale_color_manual(values = c(PB = "#1B9E77", DL = "#D95F02", PGDL = "#7570B3")) +
scale_shape_manual(values = c(PB = 16, DL = 15, PGDL = 18))   # 双编码

# violin + rug + median
ggplot(df, aes(model, rmse, fill = model)) +
  geom_violin(scale = "width", show.legend = FALSE) +
  geom_point(shape = 95, size = 2.2, show.legend = FALSE) +      # 个体短横
  stat_summary(fun = median, geom = "crossbar",
               aes(width = 0.35), show.legend = FALSE) +
  scale_y_reverse()
```

### 5.3 概念图

- 工具：Illustrator/Inkscape 或 R `grid`/Python `matplotlib.patches` 均可；
  关键是**网格对齐**（泳道等高、列等宽、箭头正交或 45°）。
- 线宽层级：数据流 0.8–1.0 pt、概念耦合 1.0–1.2 pt、边框 1.2–1.6 pt（teal 圆角框可 2 pt）。
- 底衬地理线稿灰度 ≤ `#DDDDDD`，节点紫/teal，边细黑 0.5–0.6 pt。

---

## 6. Caption 语法（与 spec §3.3 对齐）

顺序固定：① 展示了什么（含图型名，如 "Kernel density plots for…"）；
② 队列/时期/信息集；③ 聚合口径、分母与不确定性构造（"vertical lines show the range
of five iterations; markers are the mean" 式必写）；④ 一句 takeaway；⑤ _claim 边界_。
样本 caption 的可直接套用的句式：

- "The vertical lines and markers represent the range and the mean of five iterations, respectively."
- "Bars and whiskers represent mean and standard error of … across all combinations of replicates."
- "Negative == Worse Performance"（轴语义直接写进轴标题亦可）。

---

## 7. 提交前自检清单

- [ ] 宽度 = 单栏 85 mm 或全宽 140 mm；高度 ≤228 mm；
      最终尺寸下最小文字 ≥7.5 pt、正文 ≥8 pt
- [ ] 矢量导出（SVG/PDF，`pdf.fonttype=42` 嵌字、`svg.fonttype="none"` 保留文本）；
      不可避免的栅格 ≥300 dpi、线画 ≥600 dpi（当前 PNG 以 300 dpi 导出，仅作预览）
- [ ] 同一 colormap 在主图 + SI 中方向一致（样本 2022 Fig 2 正向 / Fig 4 反向，属其不足）
- [ ] 若含地图，caption 写明投影与比例尺（本项目加严项，样本三张地图均未标）
- [ ] 同一角色全图同色；每个序列有色外第二编码；灰 = 基线、强调色 ≤1 个/图
- [ ] 误差轴方向语义正确（RMSE 反向）；参考线（0/名义/上限）已画且 caption 点名
- [ ] 误差棒/带/ violin 的统计含义在 caption 写明（range/SE/median/带宽声明）
- [ ] 一篇论文单一皮肤（A 或 B）；分面共享轴与图例；面板标签位置一致
- [ ] 无 3D、渐变、阴影、饼图、红绿单独表义、未标注的 log/截断轴
- [ ] 色盲模拟 + 灰度打印检查通过；标签无裁切、图例完整、caption 引用一致
- [ ] 所有经验坐标绑定 value ID（spec §2.2），caption 数值与主表共享同一 ID/舍入
