# MULTICORE V2 LINEAGE — 实验状态记录（2026-08-03 11:15 UTC）

状态：**进行中（Stage-09 已完成并出正式 receipt；Stage-09b 正在运行）**
适用对象：导师 Review 前的工作进展记录。

## 1. 当前 lineage 定义

| 项 | 值 |
|---|---|
| 分支 | `feat/multicore`（worktree: `thermoroute-water-temperature-multicore`） |
| 最新 commit | `2440230`（ops 绑定 `bdc11ad7`）→ 主树合并 commit `39372b2` |
| source_sha256 | `bdc11ad7a22edbfe1d332a142b8c9815810183a39331b4ae3533d19484e62f37` |
| 数值策略 | `protocols/route_a_numerical_policy_v2.json`：**全角色统一 8 线程** |
| 政策 amendment | `route_a_numerical_policy_amendment_v2.json` + seal（supersedes v1） |
| 复现级别 | 固定种子 + 锁依赖版本 + 指标级复现（内部回放校验全部保留：LGB 1e-12 / 神经 1e-5） |

## 2. 运行历程（三轮）

| 轮次 | source | 结果 | 说明 |
|---|---|---|---|
| v1 | `631382c3` | 主动停止 | 16/2 线程身份冲突（导师 B-01），控制臂必然 fail-closed |
| v2 | `c207b618` | 跑到最后一步失败 | **35/35 控制成员全部通过新授权校验**（B-01 修复验证成功）；死于 `model_suite.py` 硬编码 `n_jobs=1` 与元数据 `n_jobs=8` 不一致 |
| **v3（当前）** | `bdc11ad7` | **Stage-09 正式完成** | 修复：`_frozen_lightgbm_n_jobs()` 从政策文档派生期望值 |

## 3. 当前实验进度

- ✅ **Stage-09 完成**：receipt `PASS_FORMAL_STAGE09_COMPLETE`
  - run_id：`34a1f59ed8a012d0f6ec`
  - source：`bdc11ad7a22e`；runtime：`7ad5ed86e71c`
  - 5/5 种子 + 35/35 控制成员；从启动到 receipt 约 3.5 小时（串行版同实验 7h18m）
- 🟢 **Stage-09b 运行中**：run_id `0c411da014382085fa92`，member receipts 24/45（16 并发 × 8 线程）
- ⏳ **Stage-16 / Stage-25**：链式脚本自动串联（`ops/stage09/chain_stage09_multicore.sh`，完成时打印 `MODEL_TRAINING_SUBCHAIN_COMPLETE (4/19)`）

## 4. 导师 Review 修复对照（B-01 ~ B-10）

| 编号 | 修复内容 | 状态 |
|---|---|---|
| B-01 | 全角色统一 8 线程，父子 runtime 契约一致 | ✅ 已验证（35/35 控制通过） |
| B-02 | 授权文档删除 `one_native_thread_per_member_process`，绑定 cap+政策文档 sha256 | ✅ |
| B-03 | amendment v2 修正 v1 矛盾（rationale / replay tolerance） | ✅ |
| B-04 | v2 seal 已创建；claim-registry/model-suite 正式绑定列为后续步骤 | 🔶 部分 |
| B-05 | `assert_role_thread_cap`：环境变量必须等于政策文档冻结值 | ✅ |
| B-06/B-07 | 链式脚本诚实命名（4/19 步）+ 内存遥测；`run_all.sh` 跟随政策 | ✅ |
| B-08 | 不 rsync；worktree 即 lineage 家（方案 A） | ✅ |
| B-09 | 串行存档补全 1.2GB 完整 closure（`~/archives/stage09-serial-c5d4aebc/`） | ✅ |
| B-10 | GFS / 2021-2023 数据未下载（冻结至 M commit 后） | ✅ |
| Review 2.3 | seed/LSTM launcher 非零退出码 fail-fast | ✅ |
| P0-6 | 新增 6 个回归测试（含 B-01 正/负例） | ✅ |
| P0-7 | CI/PR（GitHub Actions） | 🔶 待推远程 |

## 5. 验证与测试

- pytest 全量（v4）：**0 失败**（含新 `test_numerical_policy_v2.py`、09b、model_suite、lgb_shards）
- LightGBM `n_jobs=1` vs `16` 位级一致（实测 max diff = 0.0）
- fork+OpenMP 挂死实测 → 弃用 fork，隔离子进程架构
- 内存峰值（遥测中）：当前 09b 阶段 ~5GB / 125GB（预算 96GB）

## 6. 下一步

1. 09b → 16 → 25 自动串联完成（预计 14:30 UTC 前后），验证四份 receipt 链
2. 19 步完整 development chain 剩余 15 步（Stage-24 model-suite freeze、Stage-27 replay、manifest、chronology）
3. **Route B 设计决策**（HUC2 仅 22 个 vs 门限 30——需在 M commit 前冻结 HUC4/网络分离 cluster 定义，≥50-60 初始）
4. M → I → G → C → authorization → one-time opening（2021-2023）
5. CI：`git push -u origin feat/multicore` + PR 到 main

## 7. 关键文件位置

- 政策文档：`protocols/route_a_numerical_policy_v2.json`
- amendment v2：`protocols/route_a_numerical_policy_amendment_v2.{json,md}` + `_seal_v2.json`
- 链式运行器：`ops/stage09/chain_stage09_multicore.sh`（日志 `outputs/logs/multicore_chain.log`，内存 `multicore_chain.memory.csv`）
- 09 receipt：`outputs/models/route_a_stage09_completion.json`
- 串行历史存档：`~/archives/stage09-serial-c5d4aebc/`
