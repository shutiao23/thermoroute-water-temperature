# ThermoRoute Route-A 剩余工作与完成计划

更新日期：2026-07-27

当前状态：**PAUSED（暂停）**

工作分支：`feat/route-a-completion`

最近的源代码边界提交：`b9f6e01`（`fix(route-a): finalize preopening source boundary`）

## 1. 这份文档是什么

这是一个可以持续更新的项目执行计划和进度表，目的是把“还要做什么、先后顺序、完成标准、风险和大致时间”说清楚。它不是实验结果，也不改变已经冻结的科学协议。

权威边界仍由以下文件定义：

- `README.md`：项目范围、当前预开封状态和证据链总说明；
- `protocols/route_a_confirmatory_protocol.md`：冻结的科学设计、统计规则和停止规则；
- `outputs/README.md`：什么条件下生成文件才算正式证据；
- `scripts/run_all.sh`：开发阶段 19 个计算步骤的实际顺序；
- `paper/agu_submission/README.md`：论文投稿前必须完成的作者与出版工作。

本计划不能用来绕过上述协议，也不能把未完成缓存写成正式结果。

### 1.1 B-02 永久 descriptive 措辞（定稿，2026-07-30；R2-1 监督修补）

> **Permanent descriptive / 永久 fixed-cohort descriptive（仅锁五比较 confirmatory
> claim eligibility）。**  
> 本小节是执行计划中的**永久 descriptive 段落**，不改变协议字节，也不进入
> source hash。权威粘贴稿见
> [`docs/B02_PERMANENT_DESCRIPTIVE_CLAIM_WORDING.md`](docs/B02_PERMANENT_DESCRIPTIVE_CLAIM_WORDING.md)。
> **不抹掉** exploratory / secondary；禁止把 p/CI/Holm 写成 confirmatory 决策证据；
> 须遵守该文档的**禁止动词清单**。

**事实（outcome-free，开封前后不变）：**

- 冻结 cohort 至多 **15** 个 HUC2；逆 Herfindahl **effective_cluster ≈ 9.54**；
- 推断门控要求 **n_clusters ≥ 30**、**effective_fraction ≥ 0.75**、
  **largest_share < 0.25** → 合取必然失败；
- 门控**仅**锁定正式五比较的 confirmatory claim eligibility；唯一合格裁定永久为
  **`DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED`**（fixed-cohort descriptive）；
- exploratory / secondary **仍保留**，可按既有标签报告；
- **opening 之后禁止**把点估计、bootstrap CI、sign-flip/Holm 写成决策证据，或
  改写成 superiority / non-inferiority / equivalence / parity，或 national /
  U.S.-river 总体推广；禁止动词见 B-02 文档清单。

**可粘贴英文（摘要/结论）：**

Because the frozen Route-A cohort contains at most 15 HUC2 groups (inverse-Herfindahl effective cluster count ≈ 9.54) and therefore fails the outcome-free inference gate requiring n_clusters ≥ 30, effective_cluster_fraction ≥ 0.75, and largest_cluster_share < 0.25, the formal five confirmatory comparisons are permanently restricted to fixed-cohort descriptive effects under DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED for confirmatory claim eligibility; exploratory and secondary analyses remain reportable as such and are not erased by the gate. Whole-HUC2 bootstrap intervals, exact sign-flip p-values, and Holm adjustments are assumption-conditional sensitivities only and must not be written as confirmatory decision evidence; after opening they must not be rewritten as superiority, non-inferiority, equivalence, parity, or national / U.S.-river generalization.

**可粘贴中文（摘要/结论）：**

由于冻结的 Route-A cohort 至多只有 15 个 HUC2 组（逆 Herfindahl 有效簇数 ≈ 9.54），必然无法通过 outcome-free 推断门控（n_clusters ≥ 30、effective_cluster_fraction ≥ 0.75、largest_cluster_share < 0.25），因此正式五比较在 confirmatory claim eligibility 上永久限定为固定 cohort 的描述性效应，裁定为 DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED；exploratory / secondary 分析仍可按既有标签报告，门控不抹掉它们。全 HUC2 bootstrap 区间、精确 sign-flip p 值与 Holm 校正仅可作为假设条件敏感度，禁止写成 confirmatory 决策证据；开封后禁止改写为 superiority、non-inferiority、equivalence、parity，或全美/全国河流总体推广结论。

## 2. 不可违反的当前指令

1. **未经用户再次明确授权，不启动、恢复或并行运行任何实验。**
2. 当前只允许做轻量级的 Git、文档和只读状态检查。
3. 不修改 APFS、操作系统内核或用户的系统配置。实验只应使用普通文件；恢复前只做磁盘余量、文件完整性和运行环境检查。
4. 不删除现有断点、`.tmp` 事务文件、锁文件或成员缓存。恢复程序应先按代码中的事务规则验证，再决定能否回收或继续。
5. 不在原始工作树 `feat/yiqu-upgrade` 上继续 Route-A 工作，不修改或提交用户原有的结果文件。
6. 当前工作树中的 `outputs/tables/lightgbm_joint_validation_selection.csv` 不随计划文档一起处理或提交。
7. 可以按冻结流程生成 authorization 并完成只读 preflight；但在创建 opening intent、请求 2021--2023 目标标签或执行 opening 前，必须再次停下来让用户审核并明确批准。
8. `b1`、`s2`、`p3` 是三个普通监测站。任何代码、图表或论文文字都不得把它们写成水库、级联水库或已知水力连通系统。

## 3. 当前已经完成到哪里

### 3.1 已完成并已提交的工程工作

- 120 站 canonical development panel、稳定站号 registry 和 HUC 元数据证据链；
- 最终预标签协议、概率指标勘误、推断范围修订和模型矩阵修订；
- development predictor bridge，当前状态为 `PASS_EXACT_PRODUCT_BRIDGE`；
- 模型序列化、运行身份、输入闭包、原子发布、线程限制、断点恢复和失败关闭逻辑；
- Stage-09、09b、16、24、25、27、28、29、30 及 opening/release 相关代码和测试；
- clustered inference、temporal coverage、outcome QC、claim registry、双 profile release verifier；
- “三个普通监测站而非水库级联”的语义修正与发布边界保护；
- 已跟踪的代码、协议和测试改动已经提交到 `feat/route-a-completion`。

### 3.2 当前实验缓存快照

当前 Stage-09 run ID 为 `74aba73df5559f626c91`。以下内容只是可恢复缓存，**在 completion receipt 生成并验证前都不是正式结果**：

- ThermoRoute 正式 5 个 seed 的 checkpoint、member bundle 和 development prediction 已存在；
- 一个 random held-station warm-start diagnostic checkpoint/prediction 已存在；
- LightGBM 已完成 87 次拟合：
  - 4 个候选配置 × 3 个 horizon = 12 次 validation selection 拟合；
  - 5 个 seed × 3 个 horizon × 5 个 head = 75 次冻结配置拟合；
- Stage-09 的 7 个 mandatory control arm × 5 个 seed，共应有 35 个成员：
  - 已生成 20 个最终 member archive；
  - 仍有 15 个成员未完成，其中 `arm04/seed0` 已到 epoch 15，`arm04/seed1` 已到 epoch 11；
  - 另外 13 个成员尚未形成最终 archive；
- Stage-09 completion receipt 仍不存在，因此 Stage-09 整体状态只能写成 **INCOMPLETE / PAUSED**；
- Stage-09b、Stage-16、Stage-25 的正式 completion receipt 均未生成；
- model-suite freeze、development replay、输入证据 freeze、chronology、authorization 和 opening 均未完成。

## 4. “项目完成”的定义

当前 Route-A 只有同时达到下面四层，才可以称为完成：

1. **开发计算完成**：Stage-09、09b、16、25 都有完整且可复验的 completion receipt；
2. **预开封证据完成**：模型套件、独立 replay、后续输入、推断 gate 和 Git chronology 全部冻结且验证通过；
3. **一次性开封完成**：用户批准后，2021--2023 目标数据只获取一次，QC、可信评分、coverage audit 和 result rendering 全部通过；
4. **论文与发布完成**：论文中的数字全部由 receipt 生成，claim validator、clean-room release、引用、作者信息、许可和版式检查全部完成。

任何一层缺失，都不能把旧数字、缓存文件或熟悉的文件名当成正式结论。

## 5. 详细执行计划

### Phase 0：恢复前安全检查

状态：**等待用户授权，不执行**

恢复实验前需要：

1. 确认没有训练进程在运行；
2. 检查剩余磁盘空间、目标目录可写性和已有缓存的可读性；
3. 对两个 resumable checkpoint 只做格式、hash、sidecar 和事务状态验证；
4. 保留原有缓存副本或可恢复备份，绝不手工拼接模型文件；
5. 选择运行平台：
   - 若继续使用当前 Mac，Stage-09 先以 `--control-workers 1`、Stage-09b 以 `--precompute-workers 1` 的低负载方式验证一个恢复成员；
   - 若迁移，优先使用稳定的 Linux 环境或 Windows + WSL2；
   - 原生 Windows 需要改写 POSIX shell、文件锁、权限、`fsync`/同主机 fresh-process 等平台逻辑；
   - 不同操作系统、CPU 或数值库属于新的 runtime identity，不能把旧缓存和新结果直接混成一条 canonical evidence chain。
6. 一个成员恢复成功后，再根据实测内存、磁盘写入和耗时决定是否继续；不自动提高并发。

2026-07-27 的只读检查显示，当前 `outputs/` 约 876 MB，磁盘仍约有 405 GiB 可用。因此暂时没有证据表明死机是“文件太大把磁盘写满”导致的；恢复时应优先排查并发、内存压力、普通文件 I/O/事务恢复和主机稳定性。

完成标准：形成一份只读恢复检查记录；没有修改正式输出，也没有请求 post-2020 标签。

### Phase 1：完成 Stage-09 并生成正式回执

状态：**部分完成，暂停中**

待做工作：

1. 由恢复代码验证并处理两个中断的 checkpoint 事务；
2. 完成剩余 15/35 个 control member；
3. 验证 35 个 control member 的 seed、forecast key、target bytes、预算和源代码身份完全一致；
4. 从已验证成员重新物化正式 control predictions/bundles；
5. 重新验证 5-seed ThermoRoute、5-seed LightGBM、baseline 和 diagnostic outputs；
6. 生成 canonical Stage-09 predictions、scores、report、三类 pointer 和 `route_a_stage09_completion.json`；
7. 对 completion receipt 和它绑定的全部文件做独立校验。

完成标准：Stage-09 completion receipt 存在、self-hash 正确、所有绑定文件可重放，且没有读取 2021--2023 outcome。

### Phase 2：完成开发阶段 19 步流水线

`scripts/run_all.sh` 定义的顺序如下。不能因为某个缓存文件存在就跳过该步骤的正式 validator。

| # | 工作 | 当前状态 | 完成证据 |
|---:|---|---|---|
| 1 | Stage-09：baseline、ThermoRoute、LightGBM、LGO、7×5 controls | 部分完成、暂停 | Stage-09 completion receipt |
| 2 | Stage-09b：PlainMLP、PlainCausalTCN、完整五 seed feature ladder，共 45 个成员 | 未完成 | Stage-09b v3 completion receipt |
| 3 | per-station LightGBM exploratory foil | 未完成 | 当前版本输出及校验记录 |
| 4 | development holdout 与 5-seed ablations | 未完成 | 当前版本诊断输出 |
| 5 | 4 个 leave-HUC2-region-out ThermoRoute fold | 未完成 | 4 个 fold checkpoint/prediction |
| 6 | region-transfer assemble 与 global LightGBM fold 对照 | 未完成 | 汇总表和描述性图 |
| 7 | Stage-16 global LSTM 5-seed in-sample | 未完成 | Stage-16 模型与 prediction closure |
| 8 | Stage-16 LSTM 4 个 transfer fold | 未完成 | 4 个 fold 输出 |
| 9 | LSTM/ThermoRoute/LightGBM 汇总报告 | 未完成 | 当前版本 report |
| 10 | algebraic bounded-deviation diagnostic | 未完成 | 描述性诊断输出 |
| 11 | REV fail-closed 状态 | 未完成 | `REV_NOT_EVALUATED...` 记录 |
| 12 | probabilistic、coverage、pinball、reliability、Brier 分析及 `--check` | 未完成 | 校验通过的表/图 |
| 13 | legacy transfer 与 regime stratification 描述性诊断 | 未完成 | 当前版本输出 |
| 14 | adaptive conformal diagnostics | 未完成 | 当前版本输出 |
| 15 | frozen ensemble 输入压力/OOD robustness | 未完成 | common-key robustness 输出 |
| 16 | calibration、latent diagnostics 和 claim statistics | 未完成 | 当前版本报告与统计表 |
| 17 | Stage-25 station-agnostic pooled external suite | 未完成 | Stage-25 completion receipt + `--check` |
| 18 | Stage-24 冻结完整 model suite | 未完成 | canonical model-suite registry |
| 19 | Stage-27 isolated full-model replay 与 development manifest | 未完成 | replay receipt + strict manifest pass |

补充要求：Stage-16 的正式 completion receipt 必须完整绑定 same-station LSTM 的选择、模型、预测和 parity audit；只跑出一个报告不算完成。

### Phase 3：按 Git 时间顺序冻结证据

以下提交顺序是证据设计的一部分，不能合并、倒置或通过重写历史伪造：

1. **D — ancillary development evidence**：提交必要但不属于权威模型闭包的开发诊断证据；
2. **M — minimal authoritative model freeze closure**：只提交 `chronology._collect_model_artifacts` 定义的精确模型闭包、四个 completion receipt、model-suite registry 和 replay receipt；禁止 `git add outputs/` 或 `git add -f outputs/` 这种整目录提交。M 创建时，candidate、target-period input 和 opening namespace 必须尚不存在；
3. **I — input evidence**：只有 M 已提交后，才提交 candidate discovery evidence、冻结的 30 站 registry/lock、历史 predictor 的 raw/normalized 文件及其 manifest；仍不得读取 outcome；
4. **G — inference gate**：从已冻结 cohort 和元数据生成并单独提交 claim-blocking inference gate；G 是 chronology receipt 的 creation-base；
5. **C — chronology**：使用 M、I、G 的完整 commit SHA 生成并提交 pre-label Git chronology receipt，证明 amendment seal、M、I、G 的祖先关系和文件字节未被事后修改。

M、I、G 必须是三个不同的提交，并保持 `M` 严格早于 `I`、`I` 严格早于 `G`。M 之后不得再修改 `src/`、`scripts/`、`tests/`、`protocols/`、`.github/` 或其他受保护的 source/control path；否则必须回到模型冻结之前重新建立合法证据链。

每个提交都必须：

- 使用精确 allowlist 暂存文件；
- 排除 lock、日志、resume 目录、临时文件和非权威缓存；
- 在 M 之前按精确路径受控清理 `src/`、`scripts/`、`tests/` 下的 ignored `__pycache__`/`.pyc`，后续验证使用 `PYTHONDONTWRITEBYTECODE` 或 repo 外临时 cache；不得把这项清理扩大到 checkpoint、resume 或用户文件；
- 检查 `git diff --cached`、`git diff --check` 和对应 validator；
- 保留当前分支历史，不 rebase、不 squash、不 amend 已进入证据链的提交；
- 不修改原始工作树里的用户文件。

### Phase 4：预开封输入、gate、preflight 和人工停止点

状态：**未开始**

待做工作：

1. 在 M 之后获取并冻结 external candidate metadata；
2. 按冻结规则确定 30 个外部站点，只把它当 metadata-only/exploratory cohort；
3. 获取并冻结所需历史气象/流量 predictor，记录 provider、请求、时间、hash、缺失和冲突；
4. 生成 inference gate。当前最多只有 15 个 HUC2 group，因此“至少 30 个 cluster”这一条件预期必然失败；这不是程序错误，而是结论范围限制；
5. 冻结 chronology 并验证 commit 祖先关系；
6. 运行完整测试、claim validation、manifest、fresh-process replay 和本机 PREOPEN release-mechanics 验证；
7. 在干净工作树上创建唯一的 opening authorization，随后确认 worktree 只存在协议允许的这一份未跟踪 authorization；
8. 运行 preflight，必须得到 `PREFLIGHT_VALID_LABELS_STILL_SEALED`，并确认 intent、outcome 和 `outputs/confirmatory` 仍不存在；
9. **停下并向用户提交审核包**：模型闭包摘要、输入清单、gate 结果、chronology、authorization、preflight 结果、风险和将要请求的数据。没有用户明确批准，不进入 Phase 5。

完成标准：所有预开封 gate 通过或按协议明确失败；没有目标标签进入 repo；用户能够在不可逆操作前作出明确决定。

### Phase 5：用户批准后执行一次性 opening

状态：**未授权、不得执行**

待做工作：

1. 创建唯一 opening intent/opening ID；
2. 请求并保存 2021--2023 outcome 原始响应、qualifier、冲突和 provenance；
3. 运行 outcome QC，禁止根据结果重新选站、改模型或改阈值；
4. 使用冻结模型进行 trusted scoring；
5. 在完全相同的 forecast keys 上比较 candidate 和 reference；
6. 运行固定五比较、Holm、cluster sensitivity、probability metrics 和 temporal coverage audit；
7. 对物理源文件做 replay，并把 coverage audit 绑定到 opening receipt；
8. 生成完整 opening receipt。若中断，只允许按相同 opening ID 和事务规则继续；不得另开一套“更好看”的结果。

完成标准：opening receipt 完整、自洽、可复验；所有负面、冲突或不可估结果都被保留。

### Phase 6：结果解释、论文和发布

状态：**未完成**

待做工作：

1. 只通过 receipt-aware renderer 生成论文中的五条 canonical result statement；
2. 完成 POST submission renderer，并验证它只消费经过绑定的证据；
3. 更新结果表、图、摘要、讨论和结论，但保留所有永久 limitation；
4. 禁止把固定 cohort 的描述性差异写成全美河流总体上的 superiority、non-inferiority、equivalence 或 parity；
5. 将 15 HUC2 cluster 下的 p-value、bootstrap CI 和 Holm 值明确写成 assumption-conditional sensitivity；
6. 对 claim registry、Markdown、TeX、PDF 和 clean-room release archive 做一致性检查；
7. 逐页检查最终 PDF 的表格、图、公式、引用、交叉引用和可读性；
8. 由作者提供并核实姓名、单位、ORCID、邮箱、funding、conflict of interest、repository URL、DOI 和许可信息；
9. 核对每条文内引用与 bibliography，并检查最新目标期刊要求；
10. 对每个拟发布文件做逐字节 rights review。当前公开分发仍被数据、Git history bundle 和 AGU class 等许可问题阻断。

完成标准：论文数字全部能追溯到 opening receipt；validator、release verification 和人工版式检查全部通过；作者与许可信息真实完整。

## 6. 对抗式审查后建议的后续研究

下面这些是提高论文外部可信度的重要工作，但不能悄悄塞进当前已冻结 Route-A 作为事后补充。若实施，应建立新的、预注册的研究阶段：

1. 扩大到至少 30 个独立且可报告的空间 cluster，再做真正有资格支持总体推断的确认性检验；
2. 使用有历史发行版本的气象预测，做真正 as-issued operational replay，而不仅是 retrospective historical-information evaluation；
3. 增加完全不使用目标站 WTEMP 历史的严格 ungauged evaluation；
4. 对不同模型使用预先声明且更公平的 tuning/compute budget；
5. 使用官方且可验证的 Air2stream 或其他强物理/过程基线，而不是 style-based unofficial implementation；
6. 增加跨年份、跨季节、极端热事件、缺失机制和 sensor qualifier 的预注册稳健性分析；
7. 在独立 Linux 主机上做 fresh-from-source reproduction，并报告跨硬件数值差异，而不是宣称跨硬件 bitwise reproducibility；
8. 使用公开时间戳、独立 custodian 或 write-once storage，提高预标签治理可信度；
9. 在未来数据上做真正前瞻验证，避免把一次 retrospective opening 当成部署证据；
10. 邀请统计、水文、环境工程和 ML reproducibility 专家做独立复核，并公开逐条响应。

## 7. 时间估计

以下是机器墙钟时间的区间估计，不是承诺。它依赖机器、是否复用通过校验的缓存、是否单 worker、磁盘稳定性、网络 API 和失败重跑次数。当前用户指令下计时为 **0**，因为实验不会自动恢复。

| 工作包 | 估计时间 |
|---|---:|
| Phase 0 恢复前安全检查 | 1--3 小时人工/轻量检查 |
| 完成 Stage-09 剩余 15 个 control member、物化与回执 | 约 4--12 机器小时 |
| Stage-09b 45 个成员及回执 | 约 12--30 机器小时 |
| per-station、ablation、4-fold transfer、诊断与 robustness | 约 10--24 机器小时 |
| Stage-16 LSTM、4-fold transfer 和正式回执 | 约 12--30 机器小时 |
| Stage-25、model freeze、isolated replay、manifest | 约 6--12 机器小时 |
| metadata/predictor 获取、gate、chronology、preflight | 约 6--24 小时，另受网络限制 |
| opening、QC、scoring、coverage、receipt | 约 6--18 小时，另受数据源限制 |
| 论文结果层、图表、引用、release 和人工 QA | 约 2--5 个工作日；rights review 另计 |

合计粗估：

- 剩余训练和开发计算链约 **50--100 机器小时**，高不确定性；
- input、gate、opening、论文和发布还需要约 **3--7 个工作日**，不包含无法预先估算的许可/rights review；
- 在稳定机器上连续运行、且缓存全部有效时，整个技术链约 **4--10 个自然日**；
- 在当前 Mac 上为了避免再次死机而坚持单 worker、分段运行和冷却检查，建议按 **7--15 个自然日**准备；
- 若发生缓存损坏、跨操作系统重建 runtime identity、API 限流或需要重新训练，时间会进一步增加。

第一次安全恢复出一个完整成员后，应使用真实速度重算后续 ETA，而不是继续沿用这个宽区间。

## 8. 下一步动作

当前唯一允许的下一步是：

1. 保持所有实验暂停；
2. 让用户审核本计划；
3. 用户明确允许后，先执行 Phase 0，只汇报安全检查结果；
4. 用户再次确认后，才以单 worker 恢复 Stage-09；
5. 到 opening 前必须再次停止，不能自动越过人工审核点。
