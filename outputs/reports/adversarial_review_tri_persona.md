# 《ThermoRoute》三视角对抗式审稿合并报告

**日期**: 2026-07-02

## 方法说明

本合并报告采用三个角色化的专业审查视角（环境工程、计算机科学、工业研究）执行统一的对抗式审查协议；这些是模拟审查视角，不代表相关院校、企业或任何真实教授/研究员的意见或背书。协议流程为:**4 轮侦察**(通读论文、代码仓库、产物文件与 git 时间线)+ **6 视角对抗审查**(超参选择协议 / 统计口径与推断 / 数字一致性 / 代码正确性与可复现工程 / 物理机理正确性 / 基准公平性)+ **逐条对抗核查**(代码逐行、产物逐位重算、git 提交时间线比对,凡属专业推断处均标注"(推断)")+ **完整性补查**(对已收敛/被证伪的原始指控做反向核实,保留公允边界)。

**历史语义勘误：** 本报告早期版本曾把 b1、s2、p3 误称为“级联”。它们只是三个普通监测站点；站点排列不证明水力连通、上下游关系或传播时滞。下文统一称为“早期三站监测案例”。

**时点说明：** 本报告审查的是 2026-07-02 当时的仓库状态，保留用于追踪历史缺陷；它不是当前 Route A 的完成度或投稿结论，后续修复须以当前代码、冻结协议和可重放证据为准。

三视角合计产出 **原始 58 条** 意见,跨视角去重后收敛为 **34 条独立缺陷**,再经逐条对抗核查确认 **50 条**(部分独立缺陷在不同 horizon / 产物 / 位置各自成立而拆分计数),其中 **critical 8 条 / major 27 条 / minor 15 条**。三个模拟视角均给出 **Reject(允许完成全链重跑与证据链重建后重新投稿)** 的推荐意见。以下保留历史正文（仅作身份与三站语义勘误）,末尾给出综合意见。

---

## 报告A（环境工程教授式审查视角）

# 对《ThermoRoute》的审稿意见(Reviewer 2)

**模拟审稿视角**: 河流热动力学、水文数据协议与物理引导机器学习方向的环境工程教授式审查。以下意见基于对论文、代码仓库与全部产物的逐条对抗式核查。

---

## ① 总体评价

这篇论文声称用物理引导架构在 120 个 USGS 站点上击败经典物理模型与强梯度提升基线, 但我核查后发现: 其关键超参 δ 是在盲测年份上扫描选定的(论文却写 "selected on validation"), 其所谓 "canonical air2stream" 是一个参数语义错位、附带文献中不存在的低温钳制项的自造变体, 其头条 win-rate 的分母把 6 个盲测期零观测的站点悄悄计为"失败", 而 Discussion 对最重要基线对比的结论方向与摘要正好相反——作者似乎并不完全清楚自己的核心结果是什么。物理机理部分同样令人失望: 图 4(a) 的硬编码标题断言着论文正文已经撤回的机理, 而 κ 的流量依赖在架构上根本不是一个可逐站检验的量。这不是一篇需要润色的论文, 而是一条需要重建的证据链。**推荐: Reject**(允许在完整重跑协议链后重新投稿)。

---

## ② 分级意见

### Critical

**C1. 残差界 δ=1.5 由盲测集选定, 论文声称 "selected on validation" 与产物时间线直接矛盾——盲测协议实质性失效**

- **证据**: `paper/ThermoRoute_paper.md:221` 声称 δ "is selected on validation"; 但盘上唯一的扫描产物 `outputs/tables/usgs_retune.csv`(mtime 6/27)列格式与旧版测试集脚本(`git show a644e25:scripts/11_retune.py`, `idx = wd.idx("test")`)完全吻合, 而现行 val 版 `scripts/11_retune.py:74` 提交于 6/30 后**从未重跑**(outputs 中无任何 val 版产物)。该测试扫描中 δ=1.5 仅在 h7 最优(1.487 vs 1.533/1.503), h1/h3 最优值分别是 0.4 与 1.0——δ 恰恰是按盲测 h7 表现挑的, 而 h7 正是对 LightGBM 优势最脆弱的 horizon。δ=1.5 已硬编码进 `scripts/09_usgs_experiment.py:50`、`13_rigor.py:44`、`13b_ablation_worker.py:45`, 污染全部 120 站主结果、4 折 LGO 与全部消融。
- **危害**: 论文自认 δ 是 "what lets the residual add skill at 3–7 days" 的关键超参。它被测试集选定, 意味着 2019–2020 盲测集不再是 one-shot, 所有 headline 数字的显著性声明失去协议基础; "selected on validation" 是对方法学的失实陈述。
- **修改要求**: 用现行 val-only 脚本在 120 站面板上重跑 δ 扫描并归档新产物(含 split/h_val 列); 以 val 选出的 δ 重跑 09/13/13b 全链并替换论文全部数字。若 val 也选出 1.5, 必须如实披露"最初由测试集扫描选定、后经 val 复核"的时间线。现有 "selected on validation" 表述不得保留。

**C2. 所谓 "canonical Toffolon–Piccolroaz air2stream-a8" 实为自造变体, "击败经典物理基线" 的标题级声明按原文不能成立**

- **证据**: `src/thermoroute/air2stream.py:56-65` 的实现缺失规范式的流量加权结构 θ(a5 + a6·cos − a8·T)——季节项未乘 θ、无 −a8·θ·T 弛豫项, a5–a8 参数语义全部错位; a7/a8 被挪用为 published 模型中不存在的指示函数低温钳制(`if T_next < a8*th: T_next -= a7*(T_next − a8*th)`), 且 θ^{+a4}、a4∈[0,3](L95/L99)无法表达规范式 θ^{−a4} 的调制方向; 模块自身 docstring(L20-22)的公式已自证偏离原式。标定为固定单起点局部 least_squares(L101), 非官方 PSO/全局优化, 且 L90-91 的 NaN 压缩使校准递推跨越缺测日。结果: h1 中位 RMSE 0.797 与裸持久性 0.797 同值、劣于 damped 0.774, 且在 94% 站点劣于 damped。论文 L48-49 与 L276 两处标榜 "canonical", 全文无任何差异披露。
- **危害**: 摘要级声明 "beats the canonical Toffolon–Piccolroaz air2stream-a8 at every lead" 的对照对象不是文献中的 air2stream, 而是一个被削弱且失真的替身。一个物理模型在 1 天步长上与平凡持续性打平(中位数层面), 是基线未正确标定的强信号。
- **需要澄清的边界**(公允起见): 逐站层面 a2s 预测并非退化为持续性(59% 站点优于 raw persistence), h1 采用气候学气温强迫也部分解释其近持续性表现; 但这不改变 "canonical" 标签失实这一核心问题。
- **修改要求**: 按 Toffolon & Piccolroaz (2015) 原式重写 `_step`(恢复 θ·(…−a8T) 结构与参数语义、删除自造低温钳制)或直接调用官方代码, 改用官方推荐的全局优化标定, 按真实日期连续段递推; 或者, 论文全文改称 "air2stream-like simplified variant" 并逐项披露与原式差异, 撤下 "canonical" 与 "beats canonical air2stream at every lead" 措辞。

**C3. 头条 win-rate 83/89/88% 的分母是 120, 把 6 个盲测期零观测的站计为"失败"; 论文自身的显著性表(n=114)给出 88/94/93%, 两套数字同文并存且 N 标注互相矛盾**

- **证据**: `scripts/09_usgs_experiment.py:322` `win = float((d.rmse_thermo < d.rmse_damped).mean())` 未 dropna, NaN<NaN=False 使 6 个无盲测数据的站被计为负例; 同一张表的 median/skill 却用 skipna 的 114 站。从 `usgs_scores_v2.csv` 复算: win/120 = 0.833/0.892/0.883, win/114 = 0.877/0.939/0.930, 后者与 `claim1_significance.md` 逐位一致。论文摘要 L47-48 写 "83/89/88% of the n=120 stations", L288 写 "Wilcoxon paired tests (n=120)"(实际配对检验只有 114 对——我复算 n=114 时 p 值与论文逐位一致), 而 L157-159 又自称 114 是 "the headline-N for win-rates"; Figure 2 标题(L310)更在同一句里写 "on the 114 stations ... wins 83/89/88%"——在 114 口径下该句字面为假。
- **危害**: 摘要 headline 统计量的 N 标注错误, 同一指标在同一论文里有两套数字, 图注与图内点数不符。虽然错误方向保守, 但任何交叉核对的审稿人都会立即发现, 并据此质疑其余所有数字。
- **修改要求**: win-rate 与 median/skill 统一为 114 站分母(修正 09:322, dropna 后计算), 摘要/Table A/§4.2/Fig2 改为 88/94/93%(或写 "100/107/106 of 114 evaluable stations"), L288 改为 n=114, 显式披露 6 站为何不可评估(见 M3)。

**C4. Discussion 断言 LightGBM 在 3–7 天优于 ThermoRoute, 与摘要、§4.2、claim1 产物的方向完全相反**

- **证据**: L451-452 "slightly better than ThermoRoute at 3–7 days"; 但摘要 L51-54 与 `claim1_significance.md`: LightGBM 仅在 h1 显著更优(−0.018, p=3.3e-05), TR 在 h3/h7 领先(+0.014/+0.006, win 0.75/0.67); §4.2 L293-306 亦写 "the ordering flips" 支持后者。
- **危害**: 对最重要强基线的对比, 同一篇论文给出两个方向相反的结论。无论哪个是笔误, 都表明作者未对自己的核心结果做过一致性通读; 我作为审稿人由此对全文数字的可信度打上问号。
- **修改要求**: 先确定对等口径下的真实方向(注意仓库中另有集成口径问题被其他审稿人指出), 再统一摘要、§4.2、Discussion 与 README 四处表述, 并注明口径。

### Major

**M1. 摘要把 LightGBM 的 h1 RMSE 0.620 冒充为 ThermoRoute 的数字(TR 真值 0.629), 在 vs air2stream 的头条对比中朝有利方向错报**

- **证据**: 摘要 L50 "(0.620 / 1.282 / 1.655 vs 0.797 / 1.464 / 1.809)"; 但 Table A(L281)与 `usgs_experiment_v2.md` 均为: LightGBM 列 **0.620**, ThermoRoute 列 0.629; 从 `usgs_scores_v2.csv` 复算 TR h1 中位数(n=114)= 0.629。h1 恰是论文自认 LightGBM 显著更优的 horizon——观感上等同借用对手的数字。
- **危害**: 定性结论不变(0.629 仍 < 0.797), 但摘要是读者最先核对的位置, 有利方向的错报会被直接判为不可信。
- **修改要求**: L50 改为 0.629, 全文 grep "0.620" 确认无其他误植; 若全篇改用逐 seed 均值口径则同步为 0.644。

**M2. 站点级 Wilcoxon 把空间相关站点当独立样本; 支撑 Claim 1 的产物标题硬编码 "(n=40)"; 摘要引用的 "Wilson 95% CI" 在仓库中无任何生成代码; 9 组以上检验无多重比较校正**

- **证据**: `scripts/12_claim_stats.py:73` 对 114 站直接 `wilcoxon(a, b)`; 论文 L150-151 自述面板横跨 35 州、13 个 HUC2 流域——同流域站点共享气象强迫与河网, 独立性假设不成立, 有效样本量远小于 114, p≤4×10⁻¹⁸ 量级被系统性夸大(推断: 相对 persistence 的优势即使做空间聚类修正大概率仍显著, 但相对 LightGBM 的边际优势未必)。`claim1_significance.md:1` 标题为 "(n=40)"(12_claim_stats.py:60 硬编码), 内容实为 114 站结果。全仓库无 Wilson CI 计算代码, 无 Bonferroni/Holm/FDR 任何提及, Limitations 一节对空间相关性只字未提。
- **危害**: 头条显著性的三重可信度缺陷: 统计假设失当、来源产物自我误标、摘要数字不可追溯。
- **修改要求**: (1) 全文统一 n=114 并修正硬编码标题; (2) 增加 HUC2 聚类 bootstrap 或以流域为单元的符号检验, 至少在附录展示 p 值对空间聚类的敏感性; (3) 全部站点级检验做 Holm/BH 校正; (4) 把 Wilson CI 写入 12_claim_stats.py 并输出成表。

**M3. 站点纳入审计与实际所用面板脱节: 6 个被记为通过 "盲测覆盖率 ≥0.80" 闸门的站, 在实际驱动 v2 的面板中盲测期水温覆盖率为 0——这正是 120→114 的真实成因**

- **证据**: `usgs_acquisition.md:9,12` 声称闸门 "ensure that every accepted station can contribute observations to the 2019–2020 blind-test evaluation", 表中 n06=1.0、n19=0.967、n32=1.0、n77=0.982、n106=1.0、n114=0.871; 但实际使用的 `panel_usgs_100.parquet` 中这 6 站 2019+ WTEMP 观测率为 0.0(逐站复核), 它们在 `usgs_predictions_v2.parquet` 中 test 行数=0。深层原因: 同一 site_id 在 `stations_meta.csv` 与 `stations_meta_120v2.csv` 中对应**不同的 USGS 站号**——审计报告描述的是 120v2 面板, 而实验跑的是站号映射错位的 panel_usgs_100。论文 L144-147 关于闸门 "prevents a station from sneaking into the panel ... only to drop out at evaluation" 的方法学声明, 对实际面板为假。
- **危害**: 这是水文数据协议层面的审计链断裂: 论文自称"可审计"的纳入过程审计的不是实际用的数据, 一条被自身数据证伪的方法学声明留在正文中。
- **修改要求**: 核对并修复 site_id→USGS 站号映射, 重新生成与实际面板严格对应的纳入审计; 论文明确主样本为 114 站, 列出被排除的 6 站及原因; 修正 L144-147 的闸门声明。

**M4. 图 4(a) 硬编码标题 "κ rises with flow (shorter memory)" 与其自身数据、(b) 面板和论文正文的撤回结论正相反**

- **证据**: `scripts/10_usgs_analysis.py:173` 硬编码该标题; 同脚本产物 `usgs_analysis.md`: mean κ_low=0.110 > κ_high=0.103, κ_high/κ_low>1 仅 4% 站点(median ratio 0.94); 论文 L399 图注与 §4.5 明确 "we retract the flow-dependent thermal-memory claim"。且 (a) 面板是全站 pooled 分箱(L166-170), 会把站间偏置混入, 呈现 Simpson 悖论式伪趋势——实际曲线在高流量端也是回落的。
- **危害**: 论文正式图件的图内断言与图注、正文、数据四方矛盾, 读者在同一页会看到一个已被撤回的机理被图题重新声明。
- **修改要求**: 改为中性标题(如 "κ vs. standardised log-flow, pooled"), 改用 within-station 去偏曲线或逐站斜率分布; 确保图内文字与 §4.5 的撤回一致。

**M5. "no learned model beats per-station damped persistence" 的黑体断言被自家 Table 2 反驳: LightGBM 在 h=1 站均 0.255 < damped 0.261, 且 3 站中 2 站胜出**

- **证据**: 论文 L40(摘要)、L100-101(黑体)、L249-250(§4.1)、L268(Fig1 图注)四处绝对化断言; `paper_tables.md` Table 2: h1 damped 0.261 vs LightGBM 0.255; `scores_all.csv` 逐站: b1 0.2958<0.3288, s2 0.2815<0.2928。无 DM 检验支撑弱化解读。
- **危害**: 该断言是引出 120 站大样本研究的动机句, 出现在四个位置, 被自己提交的表格逐位证伪。作为"诚实负结果"卖点的句子本身不诚实, 尤为刺眼。
- **修改要求**: 改为有限定的表述(如 "在 3–7 天无学习模型显著优于 damped; h1 处 LightGBM 有 0.006 °C 边际优势"), 或补 LightGBM vs damped 的 DM 检验证明 h1 差异不显著后再作弱断言。

### Minor

**m1. 图 3 标题 "89–97% within ±0.05" 来源不明, 与正文 93/91/86%(Wilson 下界低至 78%)矛盾** — L373 vs L344-345 与 `claim3_calibration.csv`(0.9298/0.9123/0.8596); 疑为把 h1 的 Wilson 区间 87–97% 误写成全 horizon 范围, 使校准显得比实际更均匀。改为 "86–93%" 或逐 horizon 引用。

**m2. Track-H 方法陈述暗示模型消费 t+1…t+h 未来气象强迫、部署需替换为 NWP(L178-182), 与代码事实不符** — 我逐一核实: ThermoRoute/LightGBM/air2stream 全部只用 ≤t 信息(`datasets.py:142`、`features.py:84-105`、`air2stream.py:152` 未来步用气候气温), 不存在未来气象泄露, 但披露不准确(方向为低估实用价值), "ranking forcing-invariant" 因无未来强迫而近乎恒真。请在 §3.1 澄清所比较模型不消费任何未来气象。

**m3. κ 的流量斜率 k_flow 是全站共享的单一标量(`thermoroute.py:65`), "3 站 κ 升 2×" 与 "仅 4% 站点" 均不是该架构能检验的逐站机理量** — 实际权重: 120 站模型 k_flow=−0.013≈0, 3 站模型 +0.093——两处"逐站"叙事都是单一全局参数经 sigmoid 折射的必然产物。论文已在 §3.2 公式中披露参数化且结论方向(撤回)与正确诊断一致, 故仅需修正 §4.5/Fig 4b 的 "per-station replication" 式措辞: 直接报告拟合斜率, 而非伪逐站统计。

**m4. `mechanism_summary.md` 仍把已撤回的 κ–流量热记忆机理(κ low 0.039→high 0.098, 1/κ≈15–22 d)作为干净正结论呈现, 无任何失效标注** — 与论文 §4.5 撤回直接冲突, 易被当作现行证据引用。顶部标注为 3 站单 seed 历史结果、已在大样本撤回, 或归档到 legacy 目录。

**m5. §4.5 把 h1 "router 95% 权重在气温" 解读为 "短程气水热交换主导"(L391-394), 物理方向误导** — router 只作用于被 tanh 钳制在 ±1.5 °C 的残差; §4.6 消融(noPrior h1 0.629→1.149)与 κ≈0.1(h1 时约 89% 是持续性)都表明 h1 预报主体是持续性型先验。改述为 "先验之外的残差在 h1 主要由气温调制", 明确 h1 主体是持续性先验。

**m6. 头条显著性产物 `claim1_significance.md` 标题为 "(n=40)", 内容实为 120 站面板(每 horizon n=114)结果** — 12_claim_stats.py:60 从 40 站时代遗留的硬编码; 与 M2 一并修复, 动态写入实际站数并重新生成。

**m7. 表格基线路径无锚点保护(潜在缺陷, 头条无恙)** — 我审计确认: v2 头条的 persistence/damped 走窗口路径, 全部 227,136 条 test 行锚点与目标 100% 实测, φ 用原始面板估计, 头条对插补锚点敏感度为零。但 `features.py:105` 的表格路径直接取传入面板值且 `require_observed_target` 不检查锚点——任何人将 `run_persistence` 用于缺口面板就会产生稻草人基线。请加对称的锚点过滤, 并在论文补一句评估样本定义: "issue day and all three target days must have observed WTEMP"。

**m8. §4.1 "The only value ThermoRoute adds here is probabilistic ... the point baselines cannot provide"(L261-263)措辞回避了 LightGBM** — 仓库自己的 LightGBM 基线同样输出分位数与超温概率(`baselines.py:154-203`), 且没有有效证据表明 TR 的概率产品在早期三站监测案例上优于它(该案例的决策价值表已因 #17/#18 缺陷被撤回, 不能反向引用)。该句应改写为承认 LightGBM 亦提供概率预警、TR 概率产品在早期三站监测案例上无已证明的优势。

---

## ③ 什么会让我改变主意

如果修订稿能做到以下几点, 我愿意把推荐改为 Major Revision 乃至更高:

1. **重建盲测协议**(针对 C1): 提交 val-only 的 δ 扫描新产物, 以 val 选定的 δ 完整重跑 09/13/13b, 用新数字替换全文; 若结论(TR 在 3–7 天相对 LightGBM 的边际优势)在干净协议下依然成立, 这反而是论文最有力的稳健性证据。
2. **一个诚实的物理基线**(针对 C2): 官方或逐式核对过的 air2stream-a8 + 全局优化标定 + 未插补校准。即便修复后 TR 的胜幅缩小, 一个真实的对照远比一个 "canonical" 标签值钱; 若坚持现实现, 全文改称 variant 并披露差异我也接受。
3. **一套自洽的数字**(针对 C3/C4/M1): 全文单一 N 口径(114)、单一产物来源、摘要/正文/图注/README 四方一致; 附一张 headline 数字 → 生成脚本 → 产物文件的对照表。
4. **统计诚实**(针对 M2): HUC2 聚类稳健性检验 + 多重比较校正后, 相对 persistence/damped 的核心声明若依然显著(我推断大概率如此), 论文的骨架是站得住的; 相对 LightGBM 的声明则按校正后结果如实降格。
5. **机理部分降格为诊断**(针对 M4/m3/m4/m5): 撤回的就彻底撤干净——图题、中间报告、解读措辞全部与 §4.5 对齐。一篇敢于报告 "κ 机理不成立" 的论文我会尊重; 一篇图题和正文互相打架的论文我不会。

这项工作的数据规模、协议意识(sample registry、observed-target 闸门)和负结果披露的意愿都高于该领域平均水平——正因如此, 证据链上这些本可避免的裂缝才格外可惜。把链条焊牢再来, 我会认真重读。

---

## 报告B（计算机科学教授式审查视角）

# 对抗式审稿报告 —— ThermoRoute: Physics-guided River Water Temperature Forecasting

**模拟审稿视角**: 机器学习方法 / 数据泄露 / 统计推断 / 数字一致性方向的计算机科学教授式审查。
**审稿日期**: 2026-07-02
**声明**: 以下每条意见均经逐条对抗核查(代码逐行、产物逐位重算、git 时间线比对)确认;凡属专业推断处均标注"(推断)"。

---

## ① 总体评价

这是一篇把"可复现""盲测""诚实负结果"写进卖点的论文,而恰恰在这三处全部失守:决定残差技能的关键超参 δ=1.5 实际由已废弃的**测试集扫描**选定,论文却白纸黑字写 "selected on validation";对最强基线 LightGBM 的头条显著性优势(h3/h7, p≤10⁻³)完全依赖把 5 个 seed 的预测**集成**后对阵单种子基线,换回论文自称的 "seed mean" 口径后 h7 优势直接崩塌(p=0.54);§4.4 引用的 Brier-skill 与当前产物相差 2.5 倍且优劣方向颠倒,Discussion 对核心对比的结论与摘要正相反,早期三站监测案例的主表数字用 HEAD 代码根本复现不出来。单独看,每一条都够 major;合在一起,它们描绘的是一条从超参选择、统计口径到论文写作全程失控的真值链。作者显然做了大量扎实的工程工作,vs persistence 的大幅优势也是真实的,但以论文当前的声明体系,我无法信任其中任何一个具体数字。**推荐: Reject**(修复真值链并按对等口径重写核心声明后可重新投稿)。

---

## ② 分级意见

### Critical

**C1. δ_scale=1.5 由已废弃的测试集扫描选定,论文声称 "selected on validation" 属实质性盲测协议违反**
- 证据: `paper/ThermoRoute_paper.md:219-221` 写 "the latter is selected on validation";但盘上唯一扫描产物 `outputs/tables/usgs_retune.csv`(mtime 6/27)列名为 `delta_scale,h1,h3,h7`,与旧版脚本(`git show a644e25:scripts/11_retune.py`, L66 `idx = wd.idx("test")`)格式吻合,与现行 val 版 `scripts/11_retune.py:74/100-101`(写 `split` 与 `h{h}_val` 列)不符;val 版 6/30 提交后**从未重跑**(outputs/ 下无任何 val 产物)。δ=1.5 硬编码进 `scripts/09_usgs_experiment.py:50`、`13_rigor.py:44`、`13b_ablation_worker.py:45`,污染全部 120 站主结果、4 折 LGO 与消融。且测试扫描中 1.5 仅在 h7 最优(1.487 vs 0.4 的 1.533)——δ 恰按盲测 h7 表现挑选,而 h7 正是对 LightGBM 优势最脆弱的 horizon。
- 危害: 盲测集不再 one-shot,2019–2020 年的信息已进入模型配置;所有大样本数字在方法学上不可用,且论文方法陈述与产物时间线直接矛盾。
- 修改要求: 用现行 val-only 脚本在 120 站面板重跑 δ 扫描并归档带 `h{h}_val` 列的产物;以 val 选出的 δ 重跑 09/13/13b 全链替换全部数字。若 val 也选 1.5,须在论文中如实披露"最初由测试集选定、后经 val 复核"的时间线。二者必居其一,"selected on validation" 的现有表述不得保留。

**C2. "5-seed mean" 实为 5 成员预测集成对阵单种子 LightGBM;对等口径下 h7 显著优势崩塌(p=0.54)**
- 证据: `scripts/12_claim_stats.py:44-46` 对 TR 先按 (site, issue_date) 对 5 seed 的 y_pred 取均值再算 RMSE(预测集成),基线不集成(parquet 中 LightGBM 仅 seed=0);`09_usgs_experiment.py:294-296` 同法,注释却写 "seed-mean"。复算逐 seed median RMSE: h1 均值 0.644 / h3 1.292 / h7 1.664,论文报的 0.629/1.282/1.655 是**集成值**且优于任何单 seed。显著性: 集成口径 h3 +0.014 (p=1.3e-08, wr 0.75)、h7 +0.006 (p=8.6e-04, wr 0.67);逐 seed 均值口径 h3 +0.005 (p=0.032, wr 0.60)、h7 +0.001 (**p=0.54**, wr 0.54),h1 劣势由 −0.018 扩大至 −0.040。库内 `src/thermoroute/results.py:61-91` 有正确的逐 seed 实现,被脚本绕过。论文 L241-242 "we report the seed mean" 属失实标注,全文无任何集成披露。
- 危害: 摘要与 §4.2 的标题级卖点"TR 在 3/7 天显著优于强梯度提升"建立在不对等比较之上——方差缩减的收益只给了自家模型。vs persistence/damped 的优势不受影响,但"物理引导优于 LightGBM"的叙事被直接动摇。
- 修改要求: (1) 全部 "seed mean" 改为 "seed ensemble" 并额外报告逐 seed 均值±std;(2) 给 LightGBM 同等集成待遇或改用逐 seed 口径重算 claim1;(3) 按对等口径重写摘要与 §4.2——现有证据只支持"集成 TR 在 3–7 天边际领先",不支持单模型层面的显著优势。

**C3. §4.4 高温预警 Brier-skill/AUPRC 是已废弃 40 站试点的旧数,与当前产物差 2.5 倍且 TR-vs-LightGBM 优劣颠倒**
- 证据: `paper/ThermoRoute_paper.md:352-354` 报 Brier-skill +0.30/+0.25/+0.24、AUPRC 0.57/0.51/0.49,"comparable to LightGBM (+0.33/+0.30/+0.28)";当前 `outputs/tables/usgs_calibration.csv` 为 TR 0.7446/0.6038/0.5143、AUPRC 0.9230/0.8154/0.7416,且 TR 三个 horizon **全部优于** LightGBM。论文数字仅与 `git show 595730f` 的 40 站试点表吻合;同段 PICP 却来自新表——同一段落混用两代产物,读者无从知晓。
- 危害: 核心校准指标错 2.5 倍、方向写反,§4.4 整段不可信。
- 修改要求: 用同一次(样本一致的)运行统一重写预警段,确保 PICP/Brier/AUPRC/REV 出自同一来源;若保留旧数必须显式标注 40 站试点。

**C4. 头条 win-rate 83/89/88% 把 6 个盲测期零观测的站计为"失败"(分母 120),与自家 claim1 表(88/94/93%, n=114)矛盾,N 标注错误**
- 证据: `scripts/09_usgs_experiment.py:322` `(d.rmse_thermo < d.rmse_damped).mean()` 未 dropna,NaN<NaN=False,6 个无盲测数据的站被计为负;同表 median/skill 用 `.median()` 自动跳 NaN(n=114)。复算: win/120 = 0.833/0.892/0.883,win/114 = 0.877/0.939/0.930,后者与 `claim1_significance.md` 逐位一致。摘要 L48 写 "of the n=120 stations",L288 写 "Wilcoxon paired tests (n=120)"(实际配对检验 n=114,复算 p 值逐位吻合),而 L157-159 又自称 114 是 "headline-N";Fig2 标题(L310)在 114 站的图上印 120 分母的数字,按字面为假。这 6 站(n06/n19/n32/n77/n106/n114)在 `panel_usgs_100.parquet` 中 2019–2020 WTEMP 观测率为 0。
- 危害: 摘要 headline 统计量分母错误、N 标注错误、同一仓库两张表互相冲突;虽方向保守,审稿人交叉核对必然发现。
- 修改要求: win-rate 与 median/skill 统一 114 站分母(或明确写 "100/107/106 of 114"),修正 09:322 与 L288 的 n 标注,统一 usgs_experiment_v2.md 与 claim1 数字。

**C5. Discussion 断言 "LightGBM 在 3–7 天略优于 ThermoRoute",与摘要、§4.2、claim1 产物方向完全相反**
- 证据: L451-452 "slightly better than ThermoRoute at 3–7 days";而摘要 L51-54 与 claim1 表: LightGBM 仅 h1 显著更优(−0.018, p=3.3e-05),TR 在 h3/h7 领先(+0.014/+0.006)。
- 危害: 同一篇论文对最重要的强基线对比给出两个相反结论。讽刺的是,按逐 seed 对等口径(见 C2),Discussion 的版本反而更接近事实(推断)——但无论哪个是笔误,都足以让审稿人怀疑作者是否清楚自己的核心结果。
- 修改要求: 先按 C2 确定对等口径下的真实方向,再统一摘要、§4.2、Discussion、README 四处表述并注明口径。

**C6. 早期三站监测案例（历史 Track A）的全部论文数字产自 git 历史之外、bug 修复之前的代码,HEAD 复现失败**
- 证据: `outputs/logs/run_experiments.log`(mtime 6/26 04:00)早于首批代码提交(6/28)两天;`review_response.md:47-48` 自述其后修复了 CQR 测试年标签泄漏与 off-by-one。用 HEAD 完整重跑: TR joint 盲测 RMSE = 0.289/0.535/0.783,论文 L250 = 0.343/0.557/0.808(h1 差 16%,远超 seed 噪声,而 DampedPersistence 重算与论文精确一致,证明管线未变);逐站 PICP 重跑 0.800–0.913,论文 L262 "0.65–0.91" 下界失效;"动态 κ 损害精度 (0.287 vs 0.343)" 的 fixedKappa 对比在重跑下 h1 反转。
- 危害: §4.1 主表、fixedKappa 定性结论、PICP 下界三处实质失效,产生论文数字的代码状态不可考。
- 修改要求: 用 HEAD 完整重跑 Track A(01→08),以再生产物重写 §4.1 全部数字;无法再生的 6/26 数字一律从论文与 README 移除。

### Major

**M1. 摘要把 LightGBM 的 h1 RMSE 0.620 冒充为 ThermoRoute 的数字(真值 0.629)**
- 证据: L49-50 vs air2stream 对比三元组 "(0.620/1.282/1.655)";Table A(L281)同行 0.620 在 LightGBM 列,TR 列为 0.629(CSV 复算 n=114 中位=0.629)。
- 危害: 摘要头条数字向有利方向错报,且 h1 恰是论文承认 LightGBM 显著更优的 horizon——观感等同于借用对手的数字。定性结论(0.629 仍 < 0.797)不受影响,故列 Major 而非 Critical。
- 修改要求: 改为 0.629 并全篇统一口径;全文 grep "0.620" 确认无其他误植。

**M2. §4.4 的 PICP/REV 表出自被取代的 _120 预测(脚本 10 不识别 v2 文件),同一句混排两代产物**
- 证据: `scripts/10_usgs_analysis.py:46-47` 只认 `usgs_predictions_120.parquet`,无 v2 分支(12/13 均为 v2 优先,不对称);`usgs_calibration/rev.csv`(6/30 21:15)实读的是 6/28 的 _120 文件(逐位复现证实)。论文 L343 的 PICP "0.911"(h3)在 v2 口径下应为 0.909;同一句的 "93/91/86%" 却来自 v2 的 claim3 表。两批样本不同(LightGBM 行数 309,363 vs 346,164),论文未披露,与 "sample-consistent v2" 的提交声明矛盾。需说明经核查的收敛边界: κ 分析**并未**混批(mechanism() 不读预测文件,κ=4% 是 v2 一致产物),REV 正向结论在 v2 重算下稳健(批间差 0.0005–0.002,小于 TR−LGB 差距)——问题是来源混杂与精确不可复现,而非结论翻转。
- 修改要求: 给脚本 10 增加 v2 输入分支,用 v2 重算 calibration/REV 表并统一 §4.4 数字来源;manifest 记录各表输入文件。

**M3. 调参预算不对等: TR 的关键 δ 经扫描选择(且历史上碰了测试年),LightGBM/Ridge 零搜索,而声称的优势仅 +0.006~+0.014**
- 证据: `src/thermoroute/baselines.py:159-162` LightGBM 全部超参固定(唯一自适应是 val 早停),L85 `Ridge(alpha=10.0)` 固定;TR 的 δ 经 `11_retune.py` 三值扫描(见 C1)。论文 §3.5 无任何双方调参预算披露。测试扫描显示 δ 选择对 h7 的影响量(0.046)大于对 LightGBM 的全部声称优势。
- 危害: 在 +0.006 量级的微小优势上,给 LightGBM 最小的 val 网格即可能抹平或反转结论(推断,但量级支持)。不满足顶会基线公平性最低要求。
- 修改要求: 对 LightGBM 做同等自由度的 val 小网格(num_leaves×lr 至少 6 组),Ridge alpha 亦在 val 上选;重算 Table A 与 claim1;baselines 小节明确报告双方调参预算。

**M4. 站点级 Wilcoxon 把空间相关站点当独立样本,9+4 组检验无多重比较校正,"Wilson 95% CI" 无任何生成代码,claim1 产物标题硬编码 "(n=40)"**
- 证据: `12_claim_stats.py:73` 对 114 站直接符号秩检验;论文 L150-151 自述面板跨 35 州 13 个 HUC2——同流域站点误差高度空间相关,独立性假设不成立,有效样本量远小于 114,p≤4×10⁻¹⁸ 被系统性夸大(推断,方向确定)。全仓库无 Wilson 区间计算代码,无 Bonferroni/Holm/FDR 任何提及;`claim1_significance.md:1` 标题硬编码 "(n=40)" 而内容是 120 站 v2 结果(`12_claim_stats.py:60`)。
- 修改要求: 全文 n 统一为 114;补 HUC2 聚类 bootstrap 或以流域为单元的检验展示 p 值敏感性;全部检验做 Holm/BH 校正;把 Wilson CI 写入脚本使摘要数字可追溯;修正硬编码标题。

**M5. §4.4 将早期三站监测案例的 PICP 范围 0.65–0.91 冒充 40 站 pilot 的 n=36 逐站范围(实测 0.70–0.99)**
- 证据: L347-349 "(n=36 PICP range 0.65–0.91)";该数字在 L262 与 git 历史中均标注为早期三站监测案例;用 40 站 `usgs_predictions.parquet` 实算逐站 PICP 为 0.70–0.99(既欠覆盖也过覆盖)。经核查,挪用发生在提交 fb6c1bf(而非 6444fc0);"小样本逐站覆盖离散更大"的定性结论仍成立,但 0.65–0.91 无任何 40 站产物支撑。
- 修改要求: 以实算 0.70–0.99 替换或删除该定量对比;全文检索 "0.65–0.91" 确保早期三站监测案例与 pilot 两处各有产物支撑。

**M6. "no learned model beats damped persistence" 被自家 Table 2 反驳**
- 证据: 摘要 L40、引言 L99-100(黑体)、§4.1 L249-250、Fig1 图注四处绝对化断言;而 `paper_tables.md` Table 2: LightGBM h1 站均 0.255 < damped 0.261,逐站 3 站中 2 站更优(b1 0.2958<0.3288, s2 0.2815<0.2928)。
- 危害: 引出大样本研究的动机句被自家提交表格直接矛盾。
- 修改要求: 改为限定表述("3–7 天无学习模型显著优于 damped;h1 处 LightGBM 有 0.006 °C 边际优势"),或给出 DM 检验支持弱化断言。

**M7. README 三处标题级声明与产物矛盾(废弃数字/方向错误)**
- 证据: (a) `README.md:29-31` 把决策价值列为当前负结果,而论文 L358-364 与 `usgs_rev.csv`(TR 0.9034/0.8548/0.8239 vs persistence 0.8434/0.7062/0.5886)是 120 站正向结果,负结果仅对已退役的 40 站试点成立;(b) `README.md:20` "beats air2stream-a8 at h7 (1.652 vs 1.695)" 引自废弃 `usgs_experiment_120.md`,且连旧表也是三个 lead 全胜(0.628<0.811, 1.281<1.370),v2 真值为每 lead 全胜;(c) `README.md:21-22` LGO "+0.13–0.23 skill" 与 `claim2_kfold_lgo.csv`(vs persist 均值 +0.181/+0.170/+0.240,逐折 0.153–0.252)任何口径都对不上,上界 0.23 还低于真实 +0.24。
- 危害: README 是审稿人核对仓库的第一入口,三处与论文对同一标题级声明给出不同甚至相反结论,会被解读为结果在版本间不稳定或选择性升级。均为文档陈旧(部分方向为低报),故 Major 而非 Critical。
- 修改要求: 以 v2 产物为唯一事实来源整体重写 README 结果段,并标注各数字的来源表格文件。

**M8. 论文残留六处 40 站旧文本,Data availability 指向错误面板,复现路径断裂**
- 证据: L16/L102/L118(贡献 3)/L465(结论)/L487/L497 均描述 "40 站/17 州";Data availability(L499-500)只列 40 站的 `panel_usgs.parquet`;真正驱动 120 站结果的 `panel_usgs_100.parquet`(名 100 实为 120 站、657480 行)全文未提;新采集的 `panel_usgs_120v2.parquet` 从未被任何实验消费;`run_all.sh:10` 示例指向 40 站旧面板。
- 危害: 贡献、结论、数据可用性三个定义研究范围的法定位置描述的是被取代的数据集;按论文指引复现只能拿到 40 站数据。
- 修改要求: 六处统一为 120 站/35 州;Data availability 改列实际使用的面板(建议重命名为 panel_usgs_120.parquet 并同步脚本路径);补充 v2 实验的确切命令行。

### Minor

**m1. 消融对比 5-seed 集成 full vs 3-seed 集成消融体,集成规模不等**(`13_rigor.py:172-180`)。经配平复算: noRouter/noMoE/noPrior 结论稳健(3-seed full 下 noRouter 仍 p=1.4e-13),唯 fixedKappa 的 h1 delta 由 +0.006 缩至 +0.001——恰好论文本已将其定性为 negligible。要求: full 限制为 seeds 0-2 重算消融表,补报逐 seed 均值±std(全文目前无任何 seed 方差数字)。

**m2. 图3标题 "89–97% within ±0.05" 与正文 93/91/86%(Wilson 下界低至 78%)矛盾**(L373 vs L344-345),疑为把 h1 的 Wilson 区间误写成全 horizon 范围。改为与 claim3 一致的 86–93% 或逐 horizon 引用。

**m3. LGO 迁移 skill 用折内合并 RMSE、主表用逐站中位数,两口径被直接对比**(`13_rigor.py:120` vs `12_claim_stats.py:54`)。经同口径重算差异 ≤0.002,"transfer 接近 in-sample" 的结论仍成立——但 Table B 须注明聚合方式,或注明两表不可直接相减。

**m4. README vs damped 技能写 +0.18(真值 +0.174→+0.17,与论文摘要打架),显著性写 p≤10⁻⁶(产物为 10⁻¹⁸~10⁻²⁰ 量级)**(`README.md:18`)。改为 +0.17/+0.08/+0.04 与 p≤4×10⁻¹⁸。

**m5. review_response #11 的验证声明 "CI now contains the headline RMSE" 在 h=1 被自家产物证伪**(`paper_tables.md:88` 0.343 ∉ [0.306, 0.337];根因是 `07_make_tables.py:176` 用 seed-ensemble 预测算 CI、Table 2 用逐 seed RMSE 均值,估计对象不一致)。经核查该 CI 未进入论文正文,h3/h7 成立,故仅要求统一估计对象并修正处置表描述。

**m6. §4.1 "The only value ThermoRoute adds here is probabilistic" 回避了 LightGBM 在早期三站监测案例上同样输出分位数与超温概率**(L261-263;`baselines.py:154-203`)。已撤回的 decision_value 表不能反向坐实 LightGBM 更优,但该句应改写为承认 LightGBM 亦提供概率预警、TR 概率产品在该案例上无已证明优势。

**m7. 关于"persistence 基线被插补锚点降级"的疑虑,经量化审计对 v2 头条不成立**(全部 227,136 条 test 行锚点与目标 100% 实测;剔除低覆盖站 median skill 变化 ≤0.005)——这一点应予作者肯定。残余要求: `features.py:105` 的表格基线路径无锚点保护,应补与 `datasets.py:138-140` 对称的过滤;并在 §3.5 明写评估样本定义("发布日与全部目标日水温实测"),使 114 站子集可独立核验。

---

## ③ 什么会让我改变主意

我不是不可说服的。若修订稿满足以下全部条件,我愿意把推荐改为 Major Revision 乃至更高:

1. **δ 的选择链条洗净**: 提交带 `split`/`h{h}_val` 列的新扫描产物,证明 δ 由 val 独立选定;若 val 仍选 1.5,给出完整时间线披露。此后 09/13/13b 全链重跑的数字替换全文。
2. **对等口径下重述核心声明**: 给 LightGBM 同等集成(或全部改逐 seed 口径)+ 同等 val 调参预算后,如果 TR 在 3–7 天的优势仍显著(哪怕缩小),我会接受"物理引导带来中程增益"的弱化版声明;如果不显著,论文改以 vs persistence/damped 的大幅稳健优势 + 迁移性 + 校准为主线——那依然是一篇可发表的论文,只是不是现在这篇。
3. **单一真值链**: 一次 HEAD 完整重跑覆盖早期三站监测案例与 120 站全部表格,manifest 恢复 e531553 式全量 sha256 登记,每个 headline 数字可追溯到唯一输入文件。README、摘要、正文、Discussion 四处对同一声明给出同一数字、同一方向。
4. **统计报告达标**: n=114 全文统一、HUC2 聚类稳健性检验、多重比较校正、Wilson CI 有代码产物。若聚类校正后 vs persistence 的 p 值仍在 10⁻⁶ 以下(推断: 大概率如此),核心结论反而会更硬。

做到这四条,这篇论文的骨架——一个诚实报告负结果、带物理先验的概率预报系统——是有价值的。做不到,再漂亮的叙事也只是建立在一条断裂真值链上的修辞。

---

## 报告C（工业高级研究员式审查视角）

# 对抗式审稿报告 — ThermoRoute: Physics-Guided River Water-Temperature Forecasting

**模拟审稿视角**: 代码正确性 / 可复现工程 / 基准规范方向的工业高级研究员式审查。
**声明**: 以下每条意见均经过对代码、git 历史、产物文件与日志的逐条核查; 凡属基于素材的专业推断均已标注"(推断)"。

---

## ① 总体评价

这是一份把"可复现工程"写进卖点、却在工程上处处失守的投稿: 论文自称的可复现体系——manifest、CI、单元测试、one-command reproduction——在冻结提交上分别处于被清空、红灯、常败与跑不通的状态, 而支撑摘要的三个标题级数字(δ 超参、"seed mean"、win-rate 分母)各自埋着一个协议级缺陷。最致命的组合是: 关键超参 δ=1.5 实际由已废弃的**测试集扫描**选定却被写成 "selected on validation", 而在对等的逐种子口径下, "TR 在 3/7 天显著优于 LightGBM" 的核心声称直接崩塌(h7 p=0.54); 与此同时, 早期三站监测案例的原始预测文件已被覆盖销毁、产生论文数字的代码状态不在 git 历史中、HEAD 重跑无法复现表格。我承认作者做了大量诚实的负结果披露, 数据管线中防泄漏的意图是真实的, 但意图不能替代执行。**推荐: Reject**(鼓励在完成全链重跑与证据链重建后重新投稿)。

---

## ② 分级意见

### Critical

**C1. 残差界 δ=1.5 由已废弃的测试集扫描选定, 论文却声称 "selected on validation"——盲测协议实质性失效**
- 证据: 盘上唯一扫描产物 `outputs/tables/usgs_retune.csv`(mtime 6/27)列名为 `delta_scale,h1,h3,h7`, 与旧版脚本(`git show a644e25:scripts/11_retune.py`, 明确 `idx = wd.idx("test")`)格式吻合; 现行 val 版 `scripts/11_retune.py:74`(6/30 提交)会写 `split` 与 `h{h}_val` 列, 但对应产物在 `outputs/` 中不存在——val 扫描从未重跑。δ=1.5 硬编码于 `scripts/09_usgs_experiment.py:50`、`scripts/13_rigor.py:44`、`13b_ablation_worker.py:45`; 且该测试扫描中 1.5 仅在 h7 最优(1.487 vs 0.4 的 1.533), h1/h3 最优分别是 0.4 与 1.0。论文 `paper/ThermoRoute_paper.md:221` 写 "the latter is selected on validation"。
- 危害: 论文自认 δ 是 "what lets the residual add skill at 3–7 days" 的关键超参, 它恰按盲测 h7 表现挑选, 而 h7 正是对 LightGBM 优势最脆弱的 horizon; 全部 120 站 v2 主结果、4 折 LGO、全部消融继承了这个被污染的值。盲测集不再是 one-shot, 方法学陈述与产物时间线直接矛盾。
- 修改要求: 用现行 val-only 脚本在 120 站面板重跑 δ 扫描并归档新 CSV; 以 val 选出的 δ 重跑 09/13/13b 全链并替换全部大样本数字; 若 val 也选出 1.5, 必须在论文中如实披露"最初由测试集选定、后经 val 复核"的时间线。"selected on validation" 的现有表述不得保留。

**C2. "5-seed mean" 实为 5 成员预测集成对阵单种子基线; 对等口径下 h7 显著性崩塌 (p=0.54)**
- 证据: `scripts/12_claim_stats.py:44-46` 在 ensemble=True 时先对 5 个 seed 的 y_pred 按 (site, issue_date) 取均值再算 RMSE, 基线不集成(parquet 中 LightGBM 仅 seed=0); `scripts/09_usgs_experiment.py:294-296` 同样做法, 注释却写 "seed-mean"。逐 seed 复算: TR h1 各 seed median RMSE 0.641–0.646(均值 0.644), 论文报的 0.629 是集成值, 优于任何单 seed。显著性对比: 集成口径 h3 +0.014 (p=1.3e-08)、h7 +0.006 (p=8.6e-04); 逐 seed 均值口径 h3 +0.005 (p=0.032)、**h7 +0.001 (p=0.54, 胜率 0.54)**, h1 劣势从 −0.018 扩大到 −0.040。库内 `src/thermoroute/results.py:69-91` 明明正确实现了逐种子打分, 却被 09/12/13 绕过。
- 危害: 方差缩减的收益单方面给了本文模型, 且被系统性误标为 "seed mean"(论文 L241-242); 摘要 "significantly leads at 3 and 7 days, winning 75%/67%" 的标题级声称在对等口径下不成立。vs persistence/damped 的优势两种口径下均稳健, 但 "物理引导优于强 GBM" 的 3–7 天叙事被动摇。
- 修改要求: 全部 "seed mean" 改为 "seed ensemble" 并补报逐 seed 均值±std; 给 LightGBM 同等集成待遇或改用逐种子口径重算 claim1; 按对等口径重写摘要与 §4.2——现有证据只支持"集成 TR 在 3–7 天边际领先"。

**C3. 所谓 "canonical air2stream-a8" 实为自造变体, "击败经典物理基线" 的摘要声明不能按原文成立**
- 证据: `src/thermoroute/air2stream.py` 缺失 Toffolon–Piccolroaz (2015) 的 θ·(a5+a6cos−a8T) 流量加权结构, a7/a8 被挪用为文献中不存在的指示函数低温钳制(L62-65), θ^{+a4} 且 a4∈[0,3] 无法表达规范式 θ^{−a4} 的调制方向, 模块自身 docstring(L20-22)即与原式不符; 标定为固定 p0 的单起点 least_squares(L101, 非官方 PSO/全局搜索), 且 L90-91 按 NaN 掩码压缩数组后跨缺测日递推。结果: h1 中位 RMSE 0.797 与 raw persistence 0.797 同值、在 94% 站点劣于 damped(0.774)。论文 L48-49 与 L276 两处标榜 "canonical", 全文无任何差异披露。
- 危害: 摘要级对照对象被错误标签——这不是与文献 air2stream 的比较, 而是与一个被结构失真、标定弱化的变体的比较。需注明经核查收敛的两点: 原指控中 "θ^a4 乘错项"(a1–a3 应在括号外)对原式的引述有误, 该具体机制不成立; "退化为持续性" 仅为中位数层面的巧合等值, 并非逐站退化。但"非 canonical + 标定弱化 + 冒称 canonical"三项均坐实。
- 修改要求: 按原式重写 `_step`(或直接调用官方 air2stream 代码), 改用全局/多起点标定, 按真实日期连续段递推; 否则论文必须改称 "air2stream-like simplified variant" 并逐项披露差异, 撤下 "canonical" 与 "beats … at every lead" 的措辞。

**C4. 头条 win-rate 83/89/88% 的分母是 120(把 6 个盲测期零观测的站计为"输"), 与自家 claim1 表(88/94/93%, n=114)矛盾, N 标注错误贯穿摘要/表A/图2**
- 证据: `scripts/09_usgs_experiment.py:322` `win = float((d.rmse_thermo < d.rmse_damped).mean())`——6 行 NaN 站因 NaN<NaN=False 计为失败; 同表 median/skill 用 `.median()` 自动跳 NaN(n=114)。复算: win/120 = 0.833/0.892/0.883, win/114 = 0.877/0.939/0.930, 后者与 `claim1_significance.md` 逐位一致。摘要 L48 写 "of the n=120 stations", L288 写 "Wilcoxon paired tests (n=120)"(实际配对检验只有 114 对, 复算 p 值逐位吻合 claim1), 而 L157-159 又自称 114 是 "the headline-N"; Fig2 caption(L310)在 "114 stations" 的图上安了 120 分母的数字, 字面为假。这 6 站(n06/n19/n32/n77/n106/n114)在 `panel_usgs_100.parquet` 中 2019–20 WTEMP 观测数为 0, 从未进入比较。
- 危害: 同一张表内两种分母混用, 摘要 headline 统计量的 N 是错的, 与仓库自身显著性产物冲突。方向虽保守, 但审稿人交叉核对 Fig2 或 claim1 表即刻发现。
- 修改要求: win-rate 与 median/skill 统一 114 站分母(09:322 dropna 后再算), 摘要/表A/图2/§4.2 改为 88/94/93%(或 "100/107/106 of 114"), L288 改 n=114。

**C5. 早期三站监测案例（历史 Track A）的全部论文数字产自 git 历史之外、CQR bug 修复之前的代码, 当前 HEAD 复现失败**
- 证据: `outputs/logs/run_experiments.log`(mtime 6/26 04:00)早于首批代码提交(eddf0fd 等, 6/28)约两天; `review_response.md:47-48` 自述 CQR 测试年标签泄漏(#20)与 off-by-one(#21)已 "Fixed (bug)", 但修复后早期三站监测案例的产物从未再生。HEAD 重跑: TR joint 盲测 RMSE 0.289/0.535/0.783 vs 论文(L250)0.343/0.557/0.808(h1 差约 16%, 远超 seed 噪声); 逐站 PICP 0.800–0.913 vs 论文 "0.65–0.91"; fixedKappa 对比在 h1 反转(0.300 vs 0.289)。确定性基线重算与论文精确吻合, 排除了数据/评估管线差异。
- 危害: §4.1 主表、"动态 κ 损害精度" 的定性结论、"PICP 0.65–0.91" 下界三处实质失效, 且产生这些数字的代码状态已不可考。
- 修改要求: 用当前 HEAD 完整重跑 Track A(01→08), 以再生成产物重写 §4.1 全部数字; 无法再生的 6/26 数字一律视为不可信并移除。

**C6. 论文依赖的原始 predictions.parquet 已被覆盖销毁, HEAD 的 sha256 登记被清空——产物-哈希-提交三方对账不可能完成**
- 证据: e531553 版 `outputs/manifest.json` 登记 `predictions.parquet` sha256_16=70d03c4c…、16,758,229 字节; 盘上现文件哈希/字节均不符(且 `.gitignore:15` 排除 `outputs/predictions/`, 仓库无副本, 原件灭失)。HEAD 工作区 manifest 全文仅剩两行占位符(该删除发生于提交 2179c15)。重跑件与旧 `scores_all.csv` 在相同键上数值直接对不上(TR-noPrior b1/h1: 重算 1.138 vs 表中 1.747), 且缺 3 个消融模型。论文 L506-515 却声称 one-command reproduction、"regenerable by the staged scripts"。
- 危害: 支撑 §4.1 的证据链在 HEAD 上断裂, 可复现性声明被自己的仓库证伪。
- 修改要求: 恢复 e531553 格式的全量 manifest 并在重跑后重新登记; 将论文引用过的预测产物纳入受控存档(Zenodo/DVC), 禁止脚本默认覆盖。

### Major

**M1. §4.4 的 PICP/REV 表出自被取代的 _120 预测: `scripts/10_usgs_analysis.py:46-47` 不识别 v2 文件**
- 证据: 10 号脚本输入回退链只认 `usgs_predictions_120.parquet`, 与 12/13 号脚本的 v2 优先逻辑不对称; `usgs_calibration.csv`(6/30 21:15)经逐位复算确为 6/28 旧批次产物, 论文 L343 的 h3 PICP 0.911 在 v2 口径下应为 0.909, 且同一句混排 _120(PICP)与 v2(93/91/86%)两代来源, 两批 LightGBM 行数差 3.7 万。经核查收敛: κ 分析(mechanism)不读预测文件、用 v2 模型+同一面板, κ=4% 是 v2 一致产物; REV 结论对批次稳健(批间差 0.0005–0.002 < TR−LGB 差距)。
- 危害: "sample-consistent v2" 的声称在 §4.4 处不成立, 数字精确不可复现。
- 修改要求: 给 10 号脚本加 v2 输入分支, 重算 calibration/REV 表并统一 §4.4 数字来源; manifest 记录每表输入。

**M2. air2stream 基线被双重压制: 插补标签上 teacher-forcing 校准 + 预报拒用当日观测气温**
- 证据: `scripts/09_usgs_experiment.py:179` 以插补面板喂 `run_air2stream`(训练期 WTEMP 观测率均值 0.78、最低 0.34——即平均 21%、最差 66% 的校准标签是 DOY 中位数伪观测), 它是唯一在插补标签上训练的模型(TR/LightGBM 均 require_observed_target=True, damped 的 φ 用原始面板 NaN 成对估计); `air2stream.py:152-155` 预报第一步用 Ta_clim[t+1] 而非 Track-H 下合法可用的观测 Ta_t, 驱动与季节相位整体错位一天——而对照模型都消耗 lag-0 气温(论文自述 h1 路由 ~95% 权重在气温)。结果 a2s h1 被压到与裸持久性持平(0.797)。论文 §3.5 对此协议只字未提。
- 危害: 摘要对"经典物理基线"的胜幅建立在未披露的弱化协议上。
- 修改要求: 在原始面板上 NaN 成对校准, 预报第一步用观测 Ta_t; 重跑两个 a2s 变体并更新对照表, 或在论文中披露协议差异。

**M3. TR-noPrior 消融被 ±1.5°C tanh 限幅混淆, "先验贡献" 被系统性夸大**
- 证据: `thermoroute.py:257-258` use_prior=False 时锚点退化为气候值, L274-275 残差仍被钳制在 ±1.5, `train.py:45` 的 L1 缰绳继续把输出拉回锚点。实测: 36.7% 的 h1 测试日 |y−clim|>1.5°C(结构上不可能命中), 16.1% 的 noPrior 预测压在限幅上, delta min/max 恰为 ∓1.5。论文 L417 却把 h1 RMSE 1.149 全部归因为"先验的信息贡献"。
- 危害: 消融表最强的一条贡献声明混合了"移除先验"与"钳到坏锚点"两个效应。
- 修改要求: noPrior 变体解除/放大 delta_scale 并将 lambda_residual 置零(或改持久性锚)重跑; 论文说明现变体的约束语义。

**M4. 站点级检验的统计报告不可追溯: 空间相关站点当独立样本、9+4 组检验无多重比较校正、"Wilson 95% CI" 无任何生成代码、产物标题硬编码 "n=40"**
- 证据: `12_claim_stats.py:73` 对 114 站直接 Wilcoxon(面板横跨 13 个 HUC2, 同流域误差高度相关, 独立性不成立); 全仓 grep 无 Wilson/Bonferroni/Holm/FDR 任何实现; `claim1_significance.md` 标题硬编码 "(n=40)"(12_claim_stats.py:60)而内容是 120 站 v2 结果。
- 危害: p≤4×10⁻¹⁸ 的量级建立在错误独立性假设上(有效样本量远小于 114, 推断); 摘要引用的 Wilson CI 是不可追溯数字; 头条显著性产物自我标注为废弃试点。
- 修改要求: 统一 n=114; 补 HUC2 聚类 bootstrap 的空间稳健性检验; Holm/BH 校正; Wilson CI 写入代码并输出表格; 修正硬编码标题。

**M5. 论文残留六处 40 站旧文本, 复现指向错误面板: 贡献 3、结论、Data availability 描述的都是被取代的数据集**
- 证据: L16/102/118/465/487/497 均写 40 站/17 州; Data availability(L499-500)只列 40 站的 `panel_usgs.parquet`/`panel_usgs_wind.parquet`; 真正驱动 120 站结果的 `panel_usgs_100.parquet`(名 100 实 120)全文未提; 新采集的 `panel_usgs_120v2.parquet` 从未被任何实验消费; `run_all.sh:10` 示例指向 40 站面板。
- 危害: 按论文指引复现只能拿到 40 站试点数据——这是复现路径错误, 不是笔误。
- 修改要求: 六处统一为 120 站/35 州; Data availability 改列实际使用面板(建议重命名为 panel_usgs_120.parquet 并同步脚本); 标注或删除未使用的 120v2 面板。

**M6. CI 有史以来只跑过一次且在论文冻结提交上失败: 浮动依赖下 3 个泄漏测试崩溃**
- 证据: gh run list 仅一条记录(failure, 7/1, 即 HEAD); 失败点 `src/thermoroute/data.py:152` `col[miss] = fill` 在 numpy 2.5/pandas 3.0 下对只读数组赋值崩溃; ci.yml 装的是浮动下界 requirements.txt 而非锁文件——锁文件头注的 "forward compatibility" 声明被直接证伪。
- 危害: 对外仓库在冻结提交处 CI 为红, 与可复现叙事相悖。
- 修改要求: 修复 data.py:152(copy=True); CI 增加锁定环境 job; 变绿后再冻结论文。

**M7. run_all.sh 与真值链不是同一条路径: 重跑不会刷新论文表格, 干净克隆下还会把 120 站预测配 40 站面板**
- 证据: 09 默认输出 `usgs_predictions.parquet`(run_all 未传 --out_predictions), 而 12/13 优先读 _v2、10 只认 _120——重跑数小时后下游仍读旧文件; 现有 v2 结果实际来自 run_all 之外的手工命令(`p0_3_rerun.log`, 与 v2 文件 mtime 精确吻合); 干净克隆(预测被 gitignore)下 10 号脚本会用 40 站 `panel_usgs_wind.parquet` 配 120 站预测做 κ/校准分析(10:49)。
- 危害: "one-command reproduction" 与实际产生发表数字的命令是两条不同路径; 当前状态下 10/12/13 三个下游脚本读的甚至不是同一份预测文件。
- 修改要求: run_all 显式传 v2 文件名; 10/12/13 输入统一为 manifest 指定的单一真值文件; 删除按 _120 存在性猜面板的逻辑。

**M8. 测试套件对真值文件零覆盖且常红; 防泄漏关键断言空转**
- 证据: `tests/test_sample_consistency.py:29-33` 候选文件恰好漏掉论文依赖的 v2(只查两个已废弃且实测 FAILED 的旧文件——LightGBM 少 5080 个 test 键); 对 v2 手工执行同一断言通过(7 模型各 227,136 键全对齐)。`tests/test_leakage.py:74-76` 注释承诺核对"目标必须等于 issue+h 处的 WTEMP", 循环体内却只有 `assert h > 0`。README:110 声称 "13 pass"。
- 危害: 名为样本一致性注册表的测试恰好不保护真值; 深度模型路径的"目标严格在未来"没有任何实际验证强度。
- 修改要求: 候选列表加 v2(或 glob), 旧文件归档; 补全目标值实检; 更正 README 测试计数。

**M9. manifest 从全量 sha256 登记退化为两行占位符, 三代产物并存无 canonical 声明**
- 证据: e531553 的 manifest 本是 135 行全量清单(承诺当时已兑现), 提交 2179c15 将其删成两行 stub 且无再生脚本; `outputs/tables/` 下 40 站/120 首跑/v2 三代 scores 全部被 git 追踪, predictions 实为三代并存。经核查收敛: 责任提交是 2179c15 而非 e531553; §4.4 未发现由此实际错引(其 PICP/REV 与 v2 期表格吻合)。
- 危害: 仓库没有机器可读的"当前真值是哪个文件"的声明, 版本漂移风险真实存在(10 号脚本已在读旧代)。
- 修改要求: 恢复并脚本化全量 manifest(headline 数字→脚本→输入/输出+sha256+时间戳); 废弃版本移入 _archive/。

**M10. 产物卫生: 支撑 Claim 2/4 的 checkpoint 不入库、无指纹、静默跨批复用; 而 25MB 废弃 wip 与第三方论文扫描页却公开发布**
- 证据: `rigor_ckpt/` 12 个 parquet 被 gitignore 排除且 manifest 无 sha256; `13_rigor.py:105-107/135` 仅按 `f.exists()` 复用, 无 commit/面板/δ 校验——worker 日志证实已实际发生一次跨批拼合(串行 noPrior seed1 单跑 49,175s 被并行批静默复用); git 已提交并推送 `_archive_wip/usgs_predictions_wip.parquet`(25MB)与 `tmp/pdfs/topp2023_page-*.png`(第三方论文扫描, 版权存疑); 另有 5 个 OLD/prefix 旧 checkpoint 目录散落。
- 危害: 重建需数十小时的关键中间产物是证据链上唯一无指纹的一环, 任何人改名旧目录即可拼出混血结果而无告警。
- 修改要求: checkpoint 指纹入 manifest; 复用前校验指纹; 从 git 历史移除 wip 与扫描页; 归档旧目录。

**M11. §4.4 将早期三站监测案例的 PICP 范围 0.65–0.91 冒充 40 站 pilot 的逐站范围(实测 0.70–0.99)**
- 证据: L348 写 "n=36 PICP range 0.65–0.91"; 该数字在 git 历史(e2baf1a)中原本标注早期三站监测案例; 用 40 站预测文件实算逐站 PICP 为 0.729–0.989(pooled)。
- 危害: 用无产物支撑的数字论证"逐站紧致覆盖是大样本属性", 且方向误导(pilot 实为宽幅双向偏差而非单纯欠覆盖)。
- 修改要求: 用实算范围替换或删除该定量对比; 全文检索 "0.65–0.91" 确保早期三站监测案例与 pilot 各有产物支撑。

**M12. "no learned model beats damped persistence" 被自家 Table 2 反驳**
- 证据: 摘要 L40、引言 L100、§4.1 L249 三处黑体绝对化断言; 而 `paper_tables.md` Table 2 h1: LightGBM 0.255 < damped 0.261, 逐站 3 站中 2 站更优(scores_all.csv)。
- 危害: 引出大样本研究的动机句被自己提交的表格证伪, 且无 DM 检验支撑弱化解读。
- 修改要求: 改为限定表述("3–7 天无学习模型显著优于 damped; h1 处 LightGBM 有 0.006°C 边际优势")或补显著性检验。

**M13. 摘要把 LightGBM 的 h1 RMSE 0.620 冒充为 ThermoRoute 的数字(真值 0.629)**
- 证据: L50 的三元组首位 0.620 属 Table A(L281)的 LightGBM 列, TR 列为 0.629(CSV 复算 n=114 中位=0.629); 错误方向有利于本文模型, 且恰在论文自认 LightGBM 显著更优的 horizon。定性结论(仍胜 air2stream 的 0.797)不受影响, 但摘要头条借用对手数字, 观感极差。
- 修改要求: 改为 0.629 并与全篇口径统一(若改逐 seed 口径则为 0.644)。

### Minor

**m1. run_all.sh 在 [2/11] 步即因 pytest 失败中止**——本地残留的两个未跟踪旧 parquet 触发 2 failed(`set -euo pipefail` 即退出); 干净克隆下这两测试会 skip, 故非提交级缺陷, 但请清理本地产物并把 v2 纳入测试候选后, 在干净环境完整跑一遍留存日志。

**m2. DM 检验 h=1 时 HAC 带宽为 0, 退化为朴素配对 t 检验**(`significance.py:84-89`)——h1 损失差实测 lag-1 自相关达 0.30–0.40, p 值数值偏小; 经 Newey-West 重算加星模式零翻转, 结论不变, 但请改用数据相关带宽(如 ⌈n^{1/3}⌉)以自洽于该模块 docstring 自己的声明。

**m3. LGO 迁移 skill 用折内合并 RMSE, 与主表逐站中位数口径不同却被直接对比**(`13_rigor.py:120` vs `12_claim_stats.py:54`)——同口径重算差异 ≤0.002, "transfer 接近 in-sample" 结论保持, 但 Table B 须注明聚合方式、正文注明两表不可直接相减。

**m4. requirements-lock.txt 系手写而非环境导出**——pytest 锁 8.0.0 而产出环境实测 7.4.4(其余 11 包一致), 且无传递依赖闭包。请 pip freeze 重生成, 并在 CI 加锁定环境 job。

**m5. review_response #11 的验证声明 "CI now contains the headline RMSE" 在 h=1 被自家产物证伪**(0.343 ∉ [0.306, 0.337])——根因是 `07_make_tables.py:176` 的 bootstrap 对 seed-ensemble 预测算 RMSE, 与 Table 2 的逐 seed 口径是不同估计量。统一估计对象并修正处置表描述; 该矛盾限于内部产物, 论文正文未引用此 CI。

**m6. claim1_significance.md 标题硬编码 "(n=40)"**(`12_claim_stats.py:60`), 内容实为 120 站面板 n=114 的结果——改为动态站数并重新生成, 避免审计者误判论文引用了废弃试点。

**m7. features.py 表格基线路径无锚点保护**(`features.py:105` persistence 列直取传入面板值, `build_tabular` 的 require_observed_target 不检查发布日锚点)——经完整量化审计, v2 头条评估样本锚点与目标 100% 实测, 头条不受影响; 但一旦有人将 `run_persistence` 用于缺口面板即产生稻草人基线。请加对称的锚点过滤/断言, 并在 §3.5 补一句评估样本定义("issue day 与全部目标日均须实测")。

---

## ③ 什么会让我改变主意

我不是不可说服的。若修订稿满足以下全部条件, 我愿意把推荐升为 Major Revision 乃至更高:

1. **δ 的选择在 val 上重新完成并归档产物**, 以新 δ 重跑 09/13/13b 全链, 论文全部大样本数字来自这一次运行; 时间线在正文与 review_response 中如实披露。
2. **对等口径的 LightGBM 对比**: 要么双方同规模集成、要么逐种子口径, 摘要与 §4.2 按新结果如实改写——如果 h7 优势在对等口径下确实消失, 就承认它消失。
3. **Track A 在当前 HEAD 完整重跑**, §4.1 全部数字可由仓库一条命令再生, 并有全量 sha256 manifest + 绿色 CI(含锁定环境 job)作为证据; 原始预测产物受控存档。
4. **air2stream 按原式重写或如实降级为 "air2stream-like variant"**, 修复插补校准与气温错位后重跑对照。
5. win-rate/N/Wilcoxon 口径全文统一为 114, Wilson CI 与多重比较校正落到代码与表格, 空间聚类稳健性至少进附录。

这份工作的数据管线里有真实的防泄漏工程(datasets.py 的 observed 闸门、09 的 inner-join 对齐经我复核确实生效), v2 预测文件本身样本一致——地基是有的。问题在于作者把"可复现"当成了写进论文的形容词, 而不是每次改动后重新执行的动词。把上面五件事做完, 这篇论文值得再看一次; 做不完, 它经不起任何一个愿意打开仓库的审稿人。

---

## 综合意见

### ① 三位评审的共识与分歧

**高度共识(三人独立命中同一缺陷,构成 P0 骨架)**:

- **δ 超参盲测泄漏**(A-C1 / B-C1 / C-C1): 三人一致认定 δ=1.5 由已废弃的测试集扫描(`usgs_retune.csv` mtime 6/27,列格式匹配 `a644e25` 的 `idx=wd.idx("test")`)选定,而论文 `paper/ThermoRoute_paper.md:221` 写 "selected on validation",val 版脚本(6/30 提交)从未重跑。这是全体公认最严重、污染全部大样本数字的协议级失效。
- **win-rate 分母错误**(A-C3 / B-C4 / C-C4): 三人一致复算 `09_usgs_experiment.py:322` 未 dropna 导致 win/120=83/89/88% 与 claim1 的 win/114=88/94/93% 冲突,摘要 L48/L288 的 n=120 标注为错。
- **摘要 0.620 冒充 TR 数字**(A-M1 / B-M1 / C-M13): 三人一致 L50 首位 0.620 属 LightGBM 列,TR 真值 0.629。
- **"no learned model beats damped" 被自家 Table 2 反驳**(A-M5 / B-M6 / C-M12): LightGBM h1 站均 0.255 < damped 0.261。
- **站点级 Wilcoxon 统计缺陷**(A-M2 / B-M4 / C-M4): 空间相关当独立、无多重比较校正、Wilson CI 无代码、claim1 标题硬编码 "(n=40)"。

**两人共识**:

- **"seed mean" 实为预测集成**(B-C2 / C-C2): 计算机科学与工业研究两个视角独立复算,对等逐 seed 口径下 h7 优势崩塌(p=0.54),这是动摇"物理引导优于 LightGBM"核心叙事的关键发现。环境工程视角未独立展开此点,但在 C4 中提示"仓库中另有集成口径问题被其他审稿人指出",与之呼应。
- **air2stream 非 canonical**(A-C2 / C-C3): 环境工程与工业研究视角一致认定其为自造变体(缺 θ 加权结构、自造低温钳制),"beats canonical" 标签失实。工业研究视角补充 M2 指出该基线还被双重压制(插补标签校准 + 拒用当日观测气温)。
- **Discussion 方向自相矛盾**(A-C4 / B-C5): LightGBM 3–7 天优劣结论与摘要/§4.2 相反。
- **40 站旧文本 / Data availability 指向错误面板**(B-M8 / C-M5)。
- **§4.4 PICP/REV 混用两代产物**(B-M2 / C-M1)、**PICP 0.65–0.91 挪用**(A 未列 / B-M5 / C-M11)。

**分歧与视角独有**:

- **可复现工程链**(C 独有 C5/C6/M6-M10): 工业研究视角额外发现早期三站监测案例（历史 Track A）用 HEAD 复现失败、原始 predictions.parquet 被覆盖销毁、manifest 退化为占位符、CI 常红、checkpoint 跨批静默复用、第三方论文扫描页入库。计算机科学视角的 C6 独立命中该三站案例复现失败与代码状态不可考,与工业研究视角的 C5 高度一致;环境工程视角未涉及工程链(视角分工使然)。
- **物理机理图题矛盾**(A 独有 M4/m3/m5): 环境工程视角深入 κ-流量热记忆的图 4(a) 硬编码标题与撤回结论矛盾、k_flow 是全站单标量非逐站量、h1 物理解读方向错误——这是另两个视角未展开的领域纵深。
- **站点纳入审计脱节**(A 独有 M3): 环境工程视角发现 site_id→USGS 站号映射错位,是 120→114 的真实成因,属水文数据协议层独有发现。
- **调参预算不对等**(B 独有 M3): 计算机科学视角指出 TR 的 δ 经扫描而 LightGBM/Ridge 零搜索,在 +0.006 量级优势上不公平。
- **noPrior 消融被限幅混淆**(C 独有 M3): 工业研究视角指出 ±1.5°C tanh 钳制使"先验贡献"被夸大。

**公允收敛点(三人均标注的被证伪/收窄指控)**: persistence 基线未被插补锚点降级(v2 头条 100% 实测,A-m7/B-m7/C-m7 一致肯定); air2stream "退化为持续性" 仅中位数巧合非逐站(C-C3 澄清); REV 正向结论对批次稳健; κ 分析未混批。这些收敛提高了报告可信度。

### ② P0/P1/P2 优先级修复清单

**P0(协议级失效,不修则全部数字不可用 — 必须完成):**

1. **重跑 val-only 的 δ 扫描并全链重算**。证据: `paper/ThermoRoute_paper.md:221`、`scripts/11_retune.py:74`、`09_usgs_experiment.py:50` / `13_rigor.py:44` / `13b_ablation_worker.py:45` 硬编码 δ=1.5、`outputs/tables/usgs_retune.csv`(mtime 6/27)。以 val 选定 δ 重跑 09/13/13b,替换全文;若仍为 1.5 须披露时间线。删除 "selected on validation" 表述。
2. **对等口径重述 LightGBM 对比**。证据: `12_claim_stats.py:44-46`、`09_usgs_experiment.py:294-296`、`results.py:61-91`(正确实现被绕过)。改用逐 seed 口径或给 LightGBM 同等集成,补报均值±std;按对等结果重写摘要与 §4.2(h7 对等口径 p=0.54)。
3. **早期三站监测案例（历史 Track A）用 HEAD 完整重跑 + 恢复证据链**。证据: `outputs/logs/run_experiments.log`(6/26)早于代码提交、`review_response.md:47-48`、manifest 占位符(提交 2179c15)、`predictions.parquet` sha256 灭失。重跑 01→08 重写 §4.1,恢复全量 sha256 manifest,预测产物受控存档。

**P1(标题级数字错误与统计不诚实,直接影响可信度 — 应完成):**

4. **win-rate / N 全文统一为 114**。证据: `09_usgs_experiment.py:322`(未 dropna)、摘要 L48、L288、Fig2 L310、`claim1_significance.md`。改 88/94/93% 或 "100/107/106 of 114",L288 改 n=114。
5. **修正摘要 0.620→0.629 并全文 grep 一致性**。证据: 论文 L50 vs Table A(L281)LightGBM 列。
6. **统计报告达标**: HUC2 聚类稳健性检验 + Holm/BH 多重比较校正 + Wilson CI 写入 `12_claim_stats.py` + 修正硬编码 "(n=40)"(`12_claim_stats.py:60`)。证据: `12_claim_stats.py:73`、论文 L150-151。
7. **air2stream 按原式重写或如实降级为 "variant"**。证据: `src/thermoroute/air2stream.py:56-65`(缺 θ 加权 / 自造低温钳制)、`:101`(单起点标定)、`:152`(预报用气候气温)、`09:179`(插补标签校准)。撤下 "canonical" 与 "beats at every lead"。
8. **修正 Discussion 方向矛盾**。证据: 论文 L451-452 vs 摘要 L51-54 vs §4.2 L293-306;先按 #2 定口径再统一四处表述。

**P2(文档、工程卫生与机理降格 — 建议完成):**

- **40 站旧文本与 Data availability 修正**(6 处 + L499-500),补 v2 命令行;证据: 论文 L16/102/118/465/487/497、`run_all.sh:10`。
- **§4.4 数字来源统一**: 给 `10_usgs_analysis.py:46-47` 加 v2 分支,重算 calibration/REV;PICP "0.65–0.91" 改实测(早期三站监测案例 0.800–0.913 / pilot 0.70–0.99)。
- **机理部分彻底降格**: 图 4(a) 硬编码标题(`10_usgs_analysis.py:173`)改中性、`mechanism_summary.md` 标注撤回、§4.5 h1 物理解读修正、k_flow 单标量披露。
- **README 结果段以 v2 为唯一事实来源重写**(`README.md:18-31`)。
- **工程卫生**: 修 `data.py:152`(copy=True)让 CI 变绿并加锁定环境 job;补 `tests/test_sample_consistency.py` 与 `test_leakage.py:74-76` 的真值覆盖;run_all 与真值链统一;checkpoint 指纹入 manifest;移除 wip 与第三方扫描页。
- **noPrior 消融解除限幅重跑**(`thermoroute.py:257-258`)、**调参预算对等披露**、**features.py:105 锚点保护**、**DM 检验带宽修正**、**消融集成规模配平**。

### ③ 总体投稿建议

**三位评审一致推荐 Reject(允许在完成全链重跑与证据链重建后重新投稿)。** 这一致性本身是强信号: 三个互相独立、视角互补(环境工程 / 机器学习统计 / 可复现工程)的审查各自沿不同路径抵达同一结论,且在 δ 泄漏、win-rate 分母、0.620 误植、Table 2 自我反驳、Wilcoxon 统计缺陷五点上完全重合。

核心判断是: 这不是一篇需要润色的论文,而是一条需要重建的证据链。作者的工程投入(120 站数据规模、防泄漏闸门、负结果披露意愿、sample registry)高于领域平均,数据管线地基真实存在——三人均予肯定。但论文把"可复现""盲测""诚实"当成写进正文的形容词,而非每次改动后重新执行的动词,导致从超参选择、统计口径到论文写作出现全程失控的真值链断裂。

关键在于: 若 P0 三项(干净 δ 协议 + 对等 LightGBM 口径 + HEAD 全链重跑)完成后,TR 相对 persistence/damped 的大幅稳健优势 + 迁移性 + 校准依然成立(三人推断大概率如此),则这仍是一篇有价值的、诚实报告负结果的物理引导概率预报论文——只是需要把相对 LightGBM 的声明按对等结果如实降格(可能从"显著优于"降为"集成边际领先"或"持平")。三位评审均明确表示,做到各自列出的条件后愿将推荐升至 Major Revision 乃至更高。**建议作者以 P0 为不可绕过的前置门槛,完成后连同 headline 数字→脚本→产物文件的可追溯对照表一并重新投稿。**
