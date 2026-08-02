# ThermoRoute 权利 / FAIR — 最长隐藏风险路径（R2-2 闭合刀）

| 字段 | 值 |
| --- | --- |
| 角色 | R2-2 Worker B（对抗式闭合；非放行） |
| 配套 | `docs/RIGHTS_INVENTORY_DRAFT.md`（暴露面清单；本文件专攻**仍未闭合**的隐藏路径） |
| 日期 | 2026-07-30 |
| 范围 | 只写 `docs/`；**未**改 source-hash 路径；**未** commit；**未**触碰 Stage09 / `outputs/runs` |
| 审查状态 | **未完成逐字节审查**；本文件是风险路径地图，**不是**授权结论、再分发放行或 FAIR 发布完成凭证 |

> **对抗式声明**：下列每一条路径都曾或仍可能被误读为「已授权 / 可再分发 / PUBLIC 即许可」。本刀明确否决这些推断。任何「清单写完 = 权利闭合」的措辞一律不成立。

---

## 0. 本刀不做什么

- **不**声称已对每个 blob 完成 SHA-256 对照与条款核验。
- **不**签发 Zenodo PUBLIC、mirror、第三方传递包或 fork-safe CI 数据工件的放行。
- **不**把 `LICENSE`（MIT）延伸为数据再许可。
- **不**把 GitHub `visibility=PUBLIC` 写成 redistribution grant。
- **不**把 Route A 跑完或 descriptive benchmark 写成数据权利已就绪。

---

## 1. 最长隐藏风险路径（按误导力排序）

下列路径不是「最显眼的文件列表」，而是**最容易让读者以为问题已解决**的隐藏链。任一环未闭合，则整条链不得当作已授权。

### 路径 P1 — AGU class 进 archive（第三方条款缺口）

```
paper/agu_submission/agujournal2019.cls (+ template / trackchanges)
  → 出现在 origin/main 与多个 feature tip
  → 读者误读：与 MIT 代码一并「开源可再分发」
  → 实际：MIT 不覆盖 AGU/第三方 TeX 再分发条款
  → 闭合条件（尚未满足）：可引用 third-party notice **或**从任何 public code/data archive 排除
```

| 断言 | 状态 |
| --- | --- |
| class 文件在 public tip 存在 | 事实（见清单 §1.1） |
| 再分发已授权 | **否** |
| THIRD_PARTY_NOTICES 已写入并核验 | **否** |
| 可进入 PUBLIC archive / Zenodo PUBLIC | **否** |

**禁止措辞**：`AGU class is bundled under MIT` / `投稿包可公开镜像` / `class 已审查放行`。

### 路径 P2 — Legacy 三站 CSV（来源与再分发未文档化）

```
data/b1.csv | data/p3.csv | data/s2.csv
  → tip 直接暴露 + history 早期提交仍可达
  → 读者误读：仓库 PUBLIC + MIT = 数据可再分发
  → 实际：README 已写来源与再分发授权尚未文档化
  → 闭合条件（尚未满足）：来源、采集协议、再分发条款可引用记录
```

| 断言 | 状态 |
| --- | --- |
| 文件在 public tip | 事实 |
| 再分发授权已文档化 | **否** |
| 可因 MIT 视为 open data | **否** |

**禁止措辞**：`case-study data is MIT-licensed` / `三站数据已开放` / `legacy CSV 可随代码发布`。

### 路径 P3 — Panel 再分发（打包字节 ≠ provider 授权）

```
data_usgs/panel_usgs*.parquet (+ 衍生 CSV / stations_meta*)
  → tip 暴露冻结 panel（含 WTEMP/FLOW 与气候衍生列）
  → 读者误读：USGS/Daymet/gridMET「可下载」= 本仓库打包字节可再打包/再许可
  → 实际：provider 端可下载 ≠ 对本仓库已打包字节的再分发授权
  → feature 分支 raw_snapshots / predictor bridge 进一步扩大暴露面
  → 闭合条件（尚未满足）：按数据类记录 ToS URL、attribution、是否允许再打包；逐字节审查未做
```

| 断言 | 状态 |
| --- | --- |
| panel 在 remote tip | 事实 |
| 「可下载」已转化为本仓再分发权 | **否** |
| 本仓打包字节可再许可为 MIT | **否** |
| raw_snapshots 已逐类 ToS 核验 | **否** |

**禁止措辞**：`USGS panel is redistributable` / `open climate covariates` / `panel 已 FAIR 发布` / `衍生列继承 upstream open license（未经核验）`。

### 路径 P4 — History 可达对象（tip 删除 ≠ 对象消失）

```
clone / fetch 任意含敏感历史的 ref
  → tip 已无路径：_archive_wip/**、.zenodo.json、旧 outputs figures|tables|reports
  → git rev-list --objects / bundle / shallow 误判仍可能取出 blob
  → 读者误读：「main 已清理 = 不可再分发内容已消失」
  → 实际：history 可达对象仍是 provenance 风险面，不是可再分发内容
  → 闭合条件（尚未满足）：完整 SHA inventory + 合格 remediation 策略评估（本 Worker 不执行）
```

| 断言 | 状态 |
| --- | --- |
| tip 删除消除再分发风险 | **否** |
| history blob 可作为 public redistribution 内容 | **否** |
| 历史 `.zenodo.json` 的 open/MIT data 主张 | **无效；不得复用** |
| 逐对象 history digest 已完成并归档为放行依据 | **否**（命令草案见清单 §3；执行≠闭合） |

**禁止措辞**：`history cleaned / sanitized for release` / `deleted paths are safe` / `旧 Zenodo 元数据仍有效`。

### 路径 P5 — PUBLIC remote ≠ 授权（可见性快照陷阱）

```
gh repo view → visibility PUBLIC, isPrivate=false
  → forks、mirror、CI checkout、第三方 runner 可拉取 tip + history
  → 读者误读：PUBLIC = 已获 redistribution / FAIR / 第三方传递许可
  → 实际：公开可读仅为访问事实快照；--distribution PUBLIC fail-closed
  → LOCAL_EVIDENCE_ONLY 包明示禁止分发，亦不构成再分发许可
```

| 断言 | 状态 |
| --- | --- |
| 仓库曾/现为 PUBLIC（快照） | 事实（以当时 `gh` 为准） |
| PUBLIC = redistribution grant | **否** |
| PUBLIC = FAIR 发布完成 | **否** |
| `--distribution PUBLIC` 当前可成功 | **否**（fail-closed） |

**禁止措辞**：`public GitHub release implies open data` / `remote PUBLIC 即已授权` / `可见性已解决权利问题`。

---

## 2. 五条路径如何串联成「假放行」叙事（对抗式拆解）

常见错误叙事（**全部否决**）：

1. 「仓库是 PUBLIC」→ 所以数据可分享。  
2. 「有 MIT LICENSE」→ 所以 panel / CSV / AGU class 都开源。  
3. 「USGS/Daymet 本来就能下载」→ 所以本仓 parquet 可再打包。  
4. 「main 上已删 `_archive_wip` / `.zenodo.json`」→ 所以历史风险没了。  
5. 「权利清单草案写完了」→ 所以可以做 Zenodo PUBLIC / 改协议放行。

**正确读法**：P1–P5 任一未闭合 → 禁止把任何数据包、投稿附属 TeX、或历史 bundle 当作已授权再分发对象；清单与本文件仅记录缺口。

---

## 3. 与 FAIR 的关系（缺口，非完成态）

| FAIR 维 | 当前 | 不得写成 |
| --- | --- | --- |
| Findable | 无正式 public data release / 无合格 release assets | 「已可发现的正式数据 DOI」 |
| Accessible | governance stop；PUBLIC 门闩 fail-closed | 「已对公众开放数据包」 |
| Interoperable | legacy panel raw replay / provenance 不完整 | 「panel 已可互操作重放」 |
| Reusable | 缺分数据类 license、attribution、再打包条款 | 「可按 MIT 复用数据」 |

---

## 4. 仍缺的闭合项（显式未完成清单）

在声称「权利审查完成」之前，至少仍缺：

1. 对每个 public remote tip + 敏感 history 对象的 **已执行并归档** 的 SHA-256 inventory（草案命令 ≠ 已跑完的放行证据）。  
2. 按数据类填写的来源 / ToS / attribution / 是否允许再打包表（**空白即未授权**）。  
3. AGU class：可引用条款写入 notice **或** 正式排除决定（**两者皆缺**）。  
4. 历史 `.zenodo.json` 主张的作废记录进入任何未来 deposit 检查清单（不得复用旧主张）。  
5. Owner + 合格审查人对 remote 访问策略 / history remediation 的评估结论（本文件不代替）。  
6. 明确：**逐字节审查尚未完成**——在完成前，禁止「开源仓库 = 可公开分发数据包」。

---

## 5. 危险措辞黑名单（本刀删除/禁止）

以下措辞若出现在 Route B、投稿、release notes、或后续 docs，应视为缺陷并删除：

| 危险措辞 | 为何危险 |
| --- | --- |
| 已授权 / redistribution authorized | 暗示闭合完成 |
| 权利审查完成 / byte-level review complete | 事实不成立 |
| PUBLIC remote 即许可 / visibility grant | 混淆访问与授权 |
| 数据亦 MIT / data under MIT | 越权延伸 LICENSE |
| panel 可再分发 / ready for Zenodo PUBLIC | 无 ToS 闭合 |
| AGU class OK to ship | 缺 notice/排除 |
| history 已清理，可发布 | tip 删除 ≠ 对象消失 |
| 清单闭合 / FAIR done | 草案 ≠ 发布 |
| Route A 完成打通了数据发布 | descriptive benchmark ≠ 权利放行 |

---

## 6. 回指

- 暴露面分类与命令草案：`docs/RIGHTS_INVENTORY_DRAFT.md`
- 报告措辞（与权利无关，但同属开封后约束）：`docs/B02_PERMANENT_DESCRIPTIVE_CLAIM_WORDING.md`、`docs/REPORTING_POLICY_DESCRIPTIVE_ONLY.md`

---

*R2-2 闭合刀结束。状态：风险路径已标；授权未颁；逐字节审查未完成。*
