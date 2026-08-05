# WRR (AGU) 图件风格：样本实证分析与本仓库适用规则

**本文档的角色（先读这一段）**

本文是 **样本证据层**：对已发表 WRR 图件逐图拆解，记录"作者实际是怎么画的"，
并裁决其中哪些做法本仓库可借、哪些禁用。它 **不定义** 本仓库的硬性规范。

权威链（冲突时自上而下）：

| 层 | 文件 | 管什么 |
|---|---|---|
| 1 | `paper/FIGURE_REDRAW_SPEC.md` | 语义配色 token、尺寸/字号、证据绑定 value ID、gate、禁用语义、导出与验收 |
| 2 | `paper/WRR_FIGURE_STYLE_GUIDE.md` | 设计系统提炼、图型选择、可复现模板参数 |
| 3 | 本文（`docs/`） | 样本原图拆解、可借表达句式、勘误与核验状态 |

> **治理提示**：`docs/` 与 `paper/` 下曾存在两份同名 `WRR_FIGURE_STYLE_GUIDE.md`，
> 内容分歧且全仓库无人引用。现已按上表划分职责：**硬指标只在第 1 层出现一次**，
> 本文一律以引用方式指向 spec，不再复述数值。第 5 节列出本轮已回写第 2 层的勘误。
>
> **2026-08-05 更名**：本文由 `docs/WRR_FIGURE_STYLE_GUIDE.md` 更名为
> `docs/WRR_FIGURE_STYLE_TEARDOWN.md`。两份文件同名会被误读为重复副本，实际职责不同：
> `paper/WRR_FIGURE_STYLE_GUIDE.md` 是第 2 层设计手册，本文是第 3 层样本拆解证据。
> 文件名现已反映该区别。

---

## 1. 期刊硬性规范（带来源与核验状态）

以下为 AGU 公开指引值。AGU/Wiley 官方页面对本环境返回 **HTTP 403**，数值经检索结果核实，
**投稿前须以编辑部最新版本复核**。

| 项目 | AGU 公开指引 | 本仓库采用值（权威 = spec §3.2） | 状态 |
|---|---|---|---|
| 单栏宽 | 50–85 mm | 85 mm | ✅ 一致（2026-08-05 由 88 mm 修订，理由见下） |
| 双栏/整页宽 | 105–170 mm | 140 mm（= 模板 `\textwidth 5.5in` = 139.7 mm） | ✅ 一致 |
| 最大高度 | ≤ 228 mm | 实际出图 122 / 156 / 182 mm | ✅ 一致 |
| 图内最小字号 | 终稿尺寸下 8 pt | 正文/轴 ≥8 pt，绝对下限 7.5 pt | ✅ 基本一致 |
| 格式 | EPS / TIF / JPG / PDF | SVG + PDF（嵌字），PNG 仅预览 | ✅ |
| 分辨率 | 栅格 ≥300 dpi；线画常见要求 600–1200 dpi | 栅格 ≥300、线画 ≥600（spec §3.4） | ✅ |

**已裁决（2026-08-05）：单栏 88 mm → 85 mm。**
理由：AGU 单栏上限为 85 mm，按 88 mm 出图会在生产环节被缩至 `85/88 = 96.6%`，
致使 7.5 pt 刻度实际印为 7.24 pt、8 pt 轴文字印为 7.73 pt，**击穿 spec 自身的字号下限**。
按 85 mm 出图不触发缩放，字号承诺才成立。当时无任何脚本使用 88 mm，修订零成本。

**仍开放：180 mm 导出变体**——超出 AGU 双栏上限 170 mm，但 spec §3.2 已将其限定为
"publisher-requested variant, not the governing placed-size assumption"，
即仅在编辑部明确要求时才产出。措辞已足够，无需修订；仅需在真的收到此类要求时复核。

**尺寸/字号的唯一事实来源是仓库代码，不是本文**：
- [render_fig01_preopening_concept.py:27-28](paper/agu_submission/figures/render_fig01_preopening_concept.py#L27-L28) — `WIDTH_MM = 140`
- [render_pre_supporting_figures.py:38](paper/si/figures/render_pre_supporting_figures.py#L38) — `FULL_WIDTH = 140 * MM`

---

## 2. 样本 → 本仓库的裁决表（照抄前必读）

样本符合 WRR 规范，但 **不等于符合本仓库已冻结的 spec**。逐条裁决：

| 样本做法 | 本仓库 | 理由 |
|---|---|---|
| 身份色用 viridis 采样（GWN 深紫 / RGCN 绿） | ❌ **禁用** | spec §3.1 已冻结 Okabe-Ito token；改色会与已产出的 TeX/SVG 冲突 |
| Train/Test 用 tab10 `#1f77b4`/`#ff7f0e` | ❌ **禁用** | 非色盲最优，且不在 token 表内 |
| 近黑面板底色（Fig 6 下） | ❌ **禁用** | spec §3.2：`Background is white`；且灰度打印必然失效 |
| 平滑 violin | ⚠ **限制** | spec §3.2 倾向原始点/ECDF/箱摘要；若用须叠 rug 并在 caption 声明带宽不作密度主张 |
| 河网图无坐标轴/无经纬网 | ⚠ **加严** | 可省网格，但 **必须** 在 caption 写明投影与比例尺；样本三张地图均无，属其不足 |
| 同一 colormap 方向全文不一致 | ❌ **禁用** | 样本 Fig 2 用 magma 正向（高=浅）、Fig 4 用 magma 反向（高=深），读者需重新学习一次 |
| matplotlib 与 ggplot2 混用 | ❌ **禁用** | 样本 Fig 3/9 与 Fig 4–8 的刻度、图例、字体渲染肉眼可辨不同；本仓库单工具单皮肤 |
| 颜色 = 角色，跨图不变 | ✅ **采纳** | 与 spec §3.1 同构 |
| 色外第二编码（形状/填充/线型/内嵌字母） | ✅ **采纳** | spec §3.1 明确要求 |
| 误差轴反向（RMSE 小值在上） | ✅ **采纳** | 位置高 = 更好，与直觉一致 |
| 灰 = 基线/上下文，强调色 ≤1 个/图 | ✅ **采纳** | 对应 `PERSIST_GRAY` / `CLIM_GRAY` / `ALLOWED_TEAL` |
| 分面承载实验维度，共享轴与图例 | ✅ **采纳** | — |
| 方向语义写进图内（"Blue = GWN outperformed RGCN"） | ✅ **采纳** | 自解释性；但本仓库 POST 值绑定前不得出现方向性形容词（spec §3.3） |

**取色声明**：本文所有 hex 均为 **目测近似**，不可直接用于出图。
样本 RGCN 绿在两份手册中分别被记为 `#21918c` 与 `#21a58a`（后者不在 viridis 曲线上），
实际约在 viridis 0.6–0.7 段（`#35b779` 附近）。需要精确值时从原文 PDF 取色，不要引用本文。

---

## 3. 跨图视觉语言（已更正）

1. **身份色跨图固定**：GWN = viridis 起点深紫 `#440154`；RGCN = viridis 中段绿。
   柱状图、直方图、图例、facet 全部沿用，读者零学习成本。→ 机制采纳，色值不采纳（§2）。
2. **连续量 colormap**：viridis（敏感性）、magma（RMSE、观测数，log 刻度）、
   plasma 离散版（生态区分类）。**方向必须全文统一**。
3. **差异量 diverging**：Fig 4 为蓝–白–**棕**（PuOr/BrBG 系，非 RdBu），
   正负方向用图内文字注释说明。
4. **两种面板底色**（更正原"三种"）：
   - 纯白 + 黑框 + 外向刻度：matplotlib 统计图、地图、示意图；
   - 白底 + **浅灰网格** + 黑色面板边框 + 灰色分面条（ggplot2 `theme_bw` 系）：分面统计图。
   - ~~浅灰底 + 白网格（`theme_gray`）~~ —— 原文误判，样本中不存在，见 §5。
   - 近黑面板仅出现于 Fig 6 下，本仓库禁用。
5. **河网地图约定**：无底图、无经纬网、无坐标轴；河段 = 按值着色的细线段；
   无数据/上下文河段 = 浅灰；水库 = 圆点单独图例；
   colorbar 垂直（深色底配白刻度）或水平（白底黑框、刻度朝下、单位置于其下方）。
   → 本仓库加严：caption 必须补投影与比例尺。
6. **概念图可衬线，数据图必须无衬线**：2019 Fig 1 图内主体为 Times 系衬线；
   2022 Fig 1 数学符号（*h*ₜ、*c*ₜ、*x*ₜ）斜体衬线，其余标签无衬线。

---

## 4. 逐图拆解

**核验状态**：✅ = 本轮已对照原图核验；◻ = 未获得样本，沿用原稿描述，使用前需复核。

### 2022WR033880（Topp et al., 2023, *WRR* 59, e2022WR033880）

> Topp, S. N., Barclay, J., Diaz, J., Sun, A. Y., Jia, X., Lu, D., Sadler, J. M., & Appling, A. P. (2023).
> *Stream Temperature Prediction in a Shifting Environment: Explaining the Influence of Deep Learning
> Architecture.* Water Resources Research. https://doi.org/10.1029/2022WR033880

#### ✅ Fig 1 — 模型架构矩阵示意图（2 行 × 3 列）
- **布局**：行 = 模型（左侧 90° 旋转粗体：RGCN / GWN），列 = 视角（顶部粗体：
  Temporal Awareness / Spatial Awareness / Model Overview），细灰线分格。
  前两列格下有无衬线副标题（LSTM / GCN / DTC / DGC），**Model Overview 列无副标题**。
- **配色**：teal 粗圆角矩形 = LSTM 单元（全图唯一饱和强调色，同时用作 DTC 输入节点**填充**
  与 Gated DTC 虚线框）；灰色小方块 = σ/tanh 激活；淡紫圆点 = 图节点，紫色曲箭头 = GCN；
  浅紫方块 = DGC；细灰箭头 = 内部数据流。
- **易漏的三处**：① 空间列底衬 **≤10% 灰的河网线稿** 提供地理语境；
  ② 细灰弯箭头 = "放大"语义（从小河网指向放大图）；
  ③ 右上角 "Transferred variables" 为纯文字注释，无框。
- **可复用**：`行(模型) × 列(能力维度)` 矩阵；强调色只给核心模块；地理底图降为背景纹理。

#### ✅ Fig 2 — 流域/生态区地图三联
- **画法**：`geom_sf` 风格多边形填色 + 深色细边界；右联为河段线着色。
- **配色**：plasma 离散（黄→橙→品红→紫→深蓝）= 生态区；三分类 hold-out（深蓝/品红/橙）；
  观测数 = magma **正向**（1 = 黑 → 10000 = 浅），log colorbar，白色刻度线，灰 = 无数据。
- **图例**：细黑边色块 + 右侧标签，多行标题，嵌入面板间空白，不压数据。
- **不足**：三张地图均无比例尺、无指北针、无投影声明。本仓库复用时须补（§2）。

#### ◻ Fig 3 — 分布曲线 + 箱线图（matplotlib 默认风）
- 左：两行共享 x 轴密度线，蓝/橙 = Test/Train，均值竖线，右上带框 legend，
  面板内 `Δ°C=1.27` 注释；y 轴为旋转 90° 的情景名。
- 右：填充箱线（同蓝橙），黑中位线与须帽，菱形离群点。
- 风格：黑色完整边框、外向刻度、DejaVu Sans、白底。

#### ✅ Fig 4 — 河网 RMSE 地图 + 月度分组柱状
- **上**：GWN / RGCN 河网 **magma 反向**着色（1 = 浅黄 → 4 = 近黑，高误差更醒目），
  浅灰 = 上下文河段，纯蓝圆点 = 水库（独立图例）；
  右联 "RGCN minus GWN" 用 **蓝–白–棕** diverging（刻度 >2 / 1.25 / 0.5 / −0.25），
  上方直接写 "Blue = GWN outperformed RGCN" 消歧。
- **下**：RMSE / Bias 两行 × 12 月分组柱，身份紫/绿；
  **白底 + 浅灰水平网格**，无面板边框，轴文字灰色；图例居右（色块 + "GWN/RGCN"）。
- **排版**：上空间、下时间，两层叙事；共享 y 轴标题 "°Celsius" 竖排居左。

#### ◻ Fig 5 — facet 水平条形（ggplot2 签名样式）
- `facet_grid` 2×2：列外大标题 Geographic / Domain Shift；行灰条黑边（Overall / Warmest 10%）。
- 水平配对条 + 黑色误差棒（mean ± SE），零线加粗黑竖线；
  共享 x 标题居下放大、共享 y 标题竖排居左；图例居右 "Model"。
- 条形方向让类别标签（Coastal / Appalachians…）水平可读。

#### ◻ Fig 7 — 原稿完全遗漏
原稿称覆盖 "Figure 1–9" 但从未拆解 Fig 7。获取原图后按本节格式补齐；
在补齐前，不得宣称本文档已完整覆盖该论文图集。

#### ✅ Fig 6 — 直方图 + 黑底河网 facet
- **上**：紫/绿半透明重叠直方图（alpha ≈ 0.6–0.7，重叠区呈条纹感），
  **白底 + 浅灰网格**，图例居右、标题 "Model"；x 轴标题含单位 `Sensitivity to Spatial Noise (Δ °C)`。
- **下**：灰 facet 条（GWN / RGCN，黑细边）+ **近黑面板** + viridis 河网线；
  colorbar 位于面板外白底上，**只标 High / Low**（无数值），刻度为白色短线；
  水库 = 浅蓝/白圆点，独立图例。
- **裁决**：近黑面板本仓库禁用（§2）；"colorbar 只标 High/Low" 亦不可照抄——
  spec §2.2 要求经验坐标绑定 value ID，定性 colorbar 无法满足。

#### ✅ Fig 8 — 归因 spaghetti + 4×2 facet 网格
- **上**：累积归因，**38 条半透明黑/灰线叠加**（深浅来自重叠，**无单独的总体黑线**），双 facet
  （GWN / RGCN，顶部灰条）；注意两模型序列长度不同（60 vs 180 天），x 轴范围不共享。
- **下**：`facet_grid` 行 = 季节（**右侧**灰条 DJF/MAM/JJA/SON）、列 = 模型（顶部灰条）；
  实线 = 均值 + 半透明同色带 = **±1 标准差**（非置信区间）；
  变量四色：深蓝 = Air Temp、紫 = Precip、粉红 = PET、橙 = Shortwave；
  每格左侧内嵌水平小条 = **各变量跨全部时间步的相对总贡献**（原 caption: "Insets represent
  the relative total contribution of each feature across timesteps"）；
- **图例**：底部 2×2 网格，每个 swatch = 实线 + 同色半透明带（双元素），与图中编码一一对应。
- **可复用**：`分面(季节×模型) + 变量配色 + 线+带 + 共享图例`；个体异质性用细线堆叠。

#### ◻ Fig 9 — 双联河网差值图（matplotlib 风）
- viridis 河网着色；水平 colorbar 黑框、刻度朝下，单位（°C、m/yr）置于 colorbar 正下方；
  面板标题无衬线居上；白底无轴。

### ✅ 2019WR024922 Fig 1 — 泳道式概念图（对照）

> Read, J. S., et al. (2019). *Process-Guided Deep Learning Predictions of Lake Water Temperature.*
> Water Resources Research. https://doi.org/10.1029/2019WR024922

- **布局**：上下两条泳道（上浅绿 = Process-based，下浅紫 = PGDL），各再纵分五格；
  顶部 5 个 **灰色斜体衬线** 列头（Inputs / Parameters / Computations / Predictions / Feedback）；
  左侧 90° 旋转的同色斜体泳道名。
- **元素语义**：黑边白底矩形 = 数据实体（drivers、observations、P、y）；
  泳道色细实线箭头 = 模型内部数据流；**品红虚线框** = 被共享的物理概念（energy balance），
  **品红实线箭头** = PB 输出作为 PGDL 预训练数据；**点线** = 反馈回路（calibrate / pre-train /
  train），花括号把"预测 − 观测"括起来接到反馈列；灰底块 + 旋转灰字（Calibrate）= 本实验省略的步骤。
- **字体**：图内主体为衬线（Times 系），与正文一致——**概念图可衬线，数据图必须无衬线**。
- **可复用**：`泳道(范式) × 列(流程阶段)` 矩阵；强调色只给"本文贡献的耦合"，
  其余降为黑/灰/泳道色，视线自然落在贡献上。ThermoRoute Fig 1(c) 证据脊柱可直接套此句式。

---

## 5. 勘误表（本轮更正，已同步回 `paper/WRR_FIGURE_STYLE_GUIDE.md`）

| 编号 | 原说法 | 更正 | 证据 |
|---|---|---|---|
| E1 | 统计图底色 = 浅灰 + 白网格（`theme_gray`）；模板代码写 `panel.background=grey95, panel.grid=white` | 白面板 + 浅灰网格 + 黑边框 + 灰分面条（`theme_bw` 系） | 2022 Fig 4 下 / Fig 6 上 / Fig 8 |
| E2 | Fig 4 右联为蓝–白–**红**（RdBu/CoolWarm） | 蓝–白–**棕**（PuOr/BrBG 系） | 2022 Fig 4 右 colorbar |
| E3 | Fig 4 河网 "magma 着色" | magma **反向**（高值深） | 2022 Fig 4 colorbar |
| E4 | Fig 8 带 = "置信带" | **±1 标准差** | 2022 Fig 8 原 caption |
| E5 | Fig 8 内嵌小条 = "离散滞后项" | **各变量跨时间步的相对总贡献** | 2022 Fig 8 原 caption |
| E6 | Fig 8 上 = "细灰线 + 黑色总体线" | 38 条半透明线叠加，**无单独总体线** | 2022 Fig 8 |
| E7 | 尺寸 178 mm / 高 ≤247 mm / 字号 6–8 pt | AGU：双栏 ≤170 mm、高 ≤228 mm、最小 8 pt；本仓库取 140 mm / ≥8 pt | AGU 指引 + spec §3.2 + 出图代码 |
| E8 | 复刻清单要求身份色改用 viridis | 以 spec §3.1 的 Okabe-Ito token 为准 | spec §3.1 |
| E9 | 允许近黑面板底色 | 禁用 | spec §3.2 `Background is white` |
| E10 | 覆盖 "Figure 1–9" | 实际缺 Fig 7；Fig 3/5/9 未核验 | 本文档 §4 |
| E11 | 目测 hex 未标注（`#21a58a`、`#2a9d8f`） | 全部标为近似，精确值须从 PDF 取色 | §2 取色声明 |

---

## 6. 复现配方

**不要新写 rcParams。** 仓库已有两份通过验收的参考实现，直接沿用其样式块：

| 用途 | 参考实现 | 关键参数 |
|---|---|---|
| 主图 | [render_fig01_preopening_concept.py:280-293](paper/agu_submission/figures/render_fig01_preopening_concept.py#L280-L293) | `DejaVu Sans` / `font.size 7.5` / `axes.linewidth 0.65` / `pdf.fonttype 42` / `svg.fonttype "none"` |
| SI 图 | [render_pre_supporting_figures.py:91-107](paper/si/figures/render_pre_supporting_figures.py#L91-L107) | `font.size 8.0` / `legend.fontsize 7.5` / 同上字体嵌入设置 |

字体嵌入两项必须保留：`pdf.fonttype = 42`（PDF 内嵌 TrueType，非曲线化）、
`svg.fonttype = "none"`（SVG 保留可搜索文本）。缺任一项 spec §3.4 的字体嵌入检查即失败。

**导出**：SVG + PDF 为交付物；PNG 当前以 `dpi=300` 导出，**低于 spec §3.4 对线画的 600 dpi 要求，
仅可用作预览**，不得作为投稿件。

**若确需 ggplot2 皮肤**（本仓库当前不使用，仅作为样本还原参考）：

```r
# 样本 2022WR033880 的实际皮肤 ≈ theme_bw，不是 theme_gray
theme_wrr_facet <- theme_bw(base_size = 8) +
  theme(panel.background = element_rect(fill = "white", color = NA),
        panel.grid.major = element_line(color = "grey92", linewidth = 0.3),
        panel.grid.minor = element_blank(),
        panel.border     = element_rect(fill = NA, color = "black", linewidth = 0.4),
        strip.background = element_rect(fill = "grey85", color = "black", linewidth = 0.4),
        strip.text       = element_text(size = 8),
        legend.key       = element_blank())
```

**概念图**：矢量工具手画或 `matplotlib.patches`；关键是网格对齐（泳道等高、列等宽、箭头正交或 45°）。
线宽层级：数据流 0.8–1.0 pt、概念耦合 1.0–1.2 pt、边框 1.2–1.6 pt、强调圆角框可 2 pt。
底衬地理线稿灰度 ≤ `#DDDDDD`。

---

## 7. 出图自检清单（可核对，非口号）

每项都能给出"通过/不通过"的客观判据。

- [ ] **尺寸**：整页 140 mm；单栏 85 mm；高度 ≤228 mm。判据：读脚本常量。
- [ ] **字号**：终稿尺寸下正文/轴 ≥8 pt，刻度/图例 ≥7.5 pt。判据：SVG 中搜 `font-size` 最小值。
- [ ] **配色**：每个角色的 hex 出现在 spec §3.1 token 表内。判据：`grep` 图内所有 hex 比对 token 表。
- [ ] **第二编码**：每个序列除颜色外另有形状/填充/线型/标注。判据：图例逐项检查。
- [ ] **灰度与色盲**：灰度打印后所有序列仍可区分；deuteranopia 模拟通过。判据：转灰度 + 模拟工具截图存档。
- [ ] **底色**：面板为白色，无渐变/阴影/3D/圆角卡片。判据：目视 + SVG 无 `linearGradient`。
- [ ] **参考线**：0 线、名义覆盖率线、上限线已画，线型区别于数据序列，且 caption 点名。
- [ ] **不确定性**：误差棒/带的统计含义（range / SE / ±1 SD / 分位）在 caption 写明；
      marker 是 mean 还是 median 写明。判据：caption 逐句核。
- [ ] **地图**：若含地图，caption 含投影与比例尺（本仓库加严项，样本未做）。
- [ ] **colormap 方向**：同一 colormap 在全文（主图 + SI）方向一致。判据：列出全部 colorbar 逐一比对。
- [ ] **caption 五段式**：展示什么 → 队列/时期/信息集 → 聚合口径与不确定性构造 →
      一句 takeaway → claim 边界（spec §3.3）。
- [ ] **禁用语义**：无红绿单独表义；warning red 不表示效应方向；POST 值绑定前无方向性形容词。
- [ ] **导出**：SVG + PDF，字体已嵌入（`fonttype 42` / `svg.fonttype none`）；PNG 仅预览。
- [ ] **证据绑定**：所有经验坐标绑定 value ID（spec §2.2），caption 数值与主表共享同一 ID 与舍入。
- [ ] **复现**：同一证据二次渲染字节一致，或满足已声明的归一化/容差规则（spec §3.4）。

---

## 附：来源

- AGU Publications, Text & Graphics Requirements —
  https://www.agu.org/publications/authors/journals/text-graphics-requirements （本环境 403，数值经检索核实）
- Wiley/AGU Graphics — https://onlinelibrary.wiley.com/page/journal/21699402/homepage/graphics.htm （403）
- Topp et al. (2023), https://doi.org/10.1029/2022WR033880
- Read et al. (2019), https://doi.org/10.1029/2019WR024922
