> Historical record (superseded by the conventional design).

# bb02498a8396ea7c6110 void 事件记录

更新日期：2026-08-02  
范围：Route-A Stage-09 形式证据 lineage（outcome-free）  
本文件不进入 `source_tree_hash`，也不构成 completion receipt。

## 1. 一句话结论

`bb02498a8396ea7c6110` **已作废（VOID）**，不得 resume、不得 promote、不得当作正式 Stage-09 证据。  
当前已完成的同源 formal run 是 `7cb2bfb18c1f9aa3dba7`（`source_sha256=19289553…`）；
它已取得 `PASS_FORMAL_STAGE09_COMPLETE`。其后 watcher 启动的 09b 在成员训练前 fail-closed；当前没有相关进程。

## 2. 原因（已拍板，非猜测）

用户在 R0-2 明确选择 **「改代码重跑」**，以修复：

- **M-01**：development CQR/Platt 校准 inclusion 与 confirmation 对齐（per-horizon independent）；
- **M-04**：composite loss 温度尺度协变（`temperature_loss_scale`）。

这两类改动落在 hashed 路径（`src/**`、`scripts/**`、`tests/**`、`protocols/**`）内，提交 `008e1f34` 后 live `source_tree_hash` 从

- `ee99225c55b2ceacdac6fdf596f0b452d5dbdb4d417edb81e234732b81521ab1`

变为

- `19289553aa0929bdb5803a8a3eaa96b38a651b3d52da441fb6298f2ac9228b55`。

`bb02498a` 的 `run.json` 绑定的是旧树 `ee99225c…`。源码身份一旦变更，该 run 的全部 member / checkpoint / prediction **不能**再进入正式证据链。

**不是**「还剩 8 个 control member 续跑未完成」的问题；续跑 `bb02498a` 在新树下是非法的。

## 3. Lineage（当前权威 vs 已作废）

| 角色 | run_id | source_sha256 | 状态 |
|---|---|---|---|
| **当前权威 completed formal run** | `7cb2bfb18c1f9aa3dba7` | `19289553aa0929bdb5803a8a3eaa96b38a651b3d52da441fb6298f2ac9228b55` | `PASS_FORMAL_STAGE09_COMPLETE`；completion receipt file SHA-256 `07a0dd1e54cfcc96179c8adaffe2d987776c210e27972faafca90cec6b12f111`；当前无相关进程 |
| **已作废（保留磁盘）** | `bb02498a8396ea7c6110` | `ee99225c55b2ceacdac6fdf596f0b452d5dbdb4d417edb81e234732b81521ab1` | VOID；约 27/35 control member 曾完成；~1.1 GB |
| **更早 audit-only** | `f1ab4da5736f5c25e2b0` | （与当前权威不同） | 仅只读对照，不得 promote |
| **计划文档曾误写** | `74aba73df5559f626c91` | （过时快照） | 历史误记；不是当前权威 |

路径：

- 当前权威候选：`outputs/runs/09_usgs_experiment/7cb2bfb18c1f9aa3dba7/`
- VOID：`outputs/runs/09_usgs_experiment/bb02498a8396ea7c6110/`

相关提交（工作树 `feat/route-a-completion`，HEAD ≈ `bc6d4eac`）：

- `c0c556f7` — metadata 上限（曾保持 `ee99225c…`）
- `fdd70c42` — gitignore runs/tables
- `008e1f34` — R0-2 M-01/M-04 → **新 hash `19289553…`，void `bb02498a`**
- `41d9485f` — ops watcher fail-closed（拒 void run）
- `6f27b552` / `bc6d4eac` — docs / paper claim SHA 回退

## 4. 已付代价（账要记清）

| 项 | 事实 |
|---|---|
| 已付机器时间 | 约 **~12 机器小时**（`bb02498a` 在旧树下推进到 ~27/35） |
| 进度回退 | 从约 **27/35** control member 回到新 run 的 **0/35** |
| 决策正确性 | 用户选择改码重跑以消掉 M-01/M-04；账上是整链重算，不是「迟到回执」或「续跑 8 个」 |
| ETA 含义 | 新 run 的墙钟回到完整 Stage-09 量级，不能沿用「剩 2–4 小时」的旧估计 |

## 5. 禁止事项（fail-closed）

1. **禁止 resume** `bb02498a8396ea7c6110`（含任何「接着跑剩 8 个 member」的操作）。
2. **禁止 promote** 该 run 的 member / prediction / checkpoint 进入 `outputs/models/` completion receipt 或 formal pointer。
3. **禁止** 把 `EXPECTED_STAGE09_RUN_ID=bb02498a…` 写回 watcher；ops 已将其列入 `VOID_STAGE09_RUN_IDS`。
4. **禁止** 为「修绿」去改 `tests/**`（会再改 source hash，第三次废 run）。pytest hermeticity 排到 Phase-3 **M commit 之后**。
5. `7cb2bfb…` receipt 已落地，但本状态勘误**不授权删除** `bb02498a` 的 ~1.1 GB 目录；它仍是 M-01/M-04 前后唯一磁盘对照，保留/处置须另行审计决定。

## 6. 审稿人会问的三点（预答）

**Q1：为什么弃掉已经跑到 27/35 的 run？**  
A：用户授权修 M-01（calib inclusion）与 M-04（loss 量纲）。二者改动 hashed 源码，`source_sha256` 从 `ee99225c…` 变为 `19289553…`。旧 run 身份与新树不一致，不能续写正式证据。

**Q2：那 ~12 机器小时和 1.1 GB 是不是白费？**  
A：对正式发表链而言该 run **不能**充当 completion evidence，因此进度回到 0/35 是真实代价。磁盘保留至新 receipt，用于 audit / 前后对照，不是为了以后 resume 回来。

**Q3：现在哪个 run 算数？能不能混用旧 LightGBM / ThermoRoute 缓存？**  
A：`7cb2bfb18c1f9aa3dba7`（绑定 `19289553…`）是当前已完成的 formal run，且其 receipt 已验证。旧树产物不得混入该 identity closure。若未来获授权修复受保护源码，必须建立新 source identity；当前 receipt 只能作为 `19289553…` 下的历史完成证据，不能跨 hash 提升。

## 7. 与 pytest 证据的关系

早先并发全量 pytest 在 live Stage-09 写 `lightgbm_shards` 时出现恰 1 个 `_artifact_snapshot` mtime 失败；静止窗口旧树曾全绿。详见：

- `outputs/logs/PYTEST_19289553_CONCURRENT_RACE_NOTE.md`

这证明新树在更苛刻条件下仍可洗清；**不**构成现在改 `tests/**` 的理由。

## 8. 历史状态勘误（2026-08-01）

`7cb2bfb18c1f9aa3dba7` 后来完成了 35/35 control-member 缓存，但 Python
在 prepublication validation 阶段以约 24.1 GiB RSS 被 OOM kill。当前没有
`scripts/09_usgs_experiment.py` 进程，仅有 PID `596058` 的 Phase-2 watcher；
`outputs/models/route_a_stage09_completion.json` 仍不存在。源码哈希仍为
`19289553aa0929bdb5803a8a3eaa96b38a651b3d52da441fb6298f2ac9228b55`。

因此本文较早的“live”称谓只描述当时 lineage 角色，不得解释为该日仍在
运行。该段描述的是 2026-08-01 的历史快照；其后的 guarded completion 已取代
`7cb2...` 的 STUCK 状态，但不改变 `bb02498a8396ea7c6110` 的永久 VOID 状态。

## 9. 当前状态勘误（2026-08-02）

guarded entrypoint 其后在完全相同的 source identity 下完成
`7cb2bfb18c1f9aa3dba7`。`outputs/models/route_a_stage09_completion.json` 的状态为
`PASS_FORMAL_STAGE09_COMPLETE`，receipt 文件 SHA-256 为
`07a0dd1e54cfcc96179c8adaffe2d987776c210e27972faafca90cec6b12f111`。

watcher 先验证该回执，才启动 09b run `a930214d93fb7bdca83e`。09b 在 formal
precompute freeze 时发现 `ArmSpec` 的 `variables`/`seeds` 仍为 tuple，而契约要求
JSON-list 形状，遂在任何成员训练前 fail-closed。该失败没有产生 authorization、work
order、成员缓存或 completion receipt，当前没有 Stage-09、09b 或 watcher 进程；没有
请求或读取 2021--2023 outcome。

下一门禁是受保护源码的 config-shape 修复授权，而非重试旧 09b run。任何该修复都会
改变 source hash，因而必须在新身份下重新建立所需 Stage-09→09b lineage；不得把
`7cb2...` 的已完成 receipt 跨 source hash 重标或推广。上述要求不影响
`bb02498a8396ea7c6110` 的永久 VOID 状态。
