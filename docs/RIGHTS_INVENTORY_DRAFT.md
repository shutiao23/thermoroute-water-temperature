# ThermoRoute 权利 / FAIR 审计清单（草案）

| 字段 | 值 |
| --- | --- |
| 角色 | R2-2 Worker A（权利/FAIR 审计启动） |
| 仓库 | `https://github.com/shutiao23/thermoroute-water-temperature` |
| 可见性（gh 只读，2026-07-30） | **PUBLIC**（`isPrivate=false`，forks=0）— **仅事实快照，≠ 再分发/再许可授权** |
| 默认分支 tip | `origin/main` @ `510f5bc824ce4bb37895adbb2bea6be02af4d75b` |
| 其他公开 remote refs | `feat/route-a-completion`、`feat/yiqu-upgrade`、`fix/delta-leakage-lightgbm-parity-p0p1`、`pull/1/head` |
| Releases | 无（`gh release list` 为空） |
| Git LFS | 未配置 / 本机无 `git-lfs` |
| 范围 | 只读核查；**未**改 `src/`、`scripts/`、`tests/`、`protocols/`；**未** commit |
| 依据 | `LICENSE`、`README.md` License / public-remote governance stop、`paper/agu_submission/README.md`、`scripts/make_release_archive.sh` 门闩、`git ls-tree` / `git rev-list` / `gh repo view` |

> **声明**：公开可读 ≠ 再分发授权。本清单是本地审计草稿，不是权利结论或放行凭证。

---

## 1. 已知 public remote 暴露对象类别

按 **当前 tip 树** 与 **仍可达历史对象** 分类。尺寸来自 `git ls-tree -r --long origin/main`（约数）。

### 1.1 Tip 树仍直接暴露（`origin/main`）

| 类别 | 代表路径 | 规模（约） | 备注 |
| --- | --- | --- | --- |
| **Legacy 三站 CSV** | `data/b1.csv`, `data/p3.csv`, `data/s2.csv` | 3 文件 / ~0.94 MB | README 明确：来源与再分发授权**尚未文档化** |
| **USGS / 衍生 panel parquet** | `data_usgs/panel_usgs.parquet`, `panel_usgs_100.parquet`, `panel_usgs_120v2.parquet`, `panel_usgs_wind.parquet` | 4 文件 / ~34.9 MB | 含 WTEMP/FLOW 与 Daymet/gridMET 衍生列的冻结 panel；原始 HTTP 溯源对 legacy panel 标注为不可用 |
| **站点 / 拒绝名单 CSV** | `data_usgs/stations_meta*.csv`, `rejected_sites_120v2.csv` | 4 文件 / ~0.17 MB | 元数据与选站痕迹 |
| **旧 outputs（figures）** | `outputs/figures/*.{png,pdf}` | 27 文件 / ~3.4 MB | 含已撤回解读所依赖的图件 |
| **旧 outputs（tables）** | `outputs/tables/**`（含 `_retune_sweep_workers/`） | 25 文件 / ~0.53 MB | 分数、校准、HUC 表、`explain.npz` 等 |
| **旧 outputs（reports）** | `outputs/reports/*.md` | 15 文件 / ~0.16 MB | 含 adversarial / USGS experiment 叙述 |
| **outputs 清单** | `outputs/manifest.json` | 1 | legacy 产物索引 |
| **论文渲染物** | `paper/*.pdf`, `paper/*.docx`, `paper/agu_submission/ThermoRoute_WRR.pdf` | 多文件 / paper/ 合计 ~3.9 MB | README：active archive 应排除 withdrawn outputs / generated figures / rendered PDF·DOCX |
| **第三方 AGU class** | `paper/agu_submission/agujournal2019.cls`（及 template / trackchanges） | 在 main 与 feature 分支 tip 均存在 | MIT **不**覆盖其再分发条款 |

### 1.2 其他公开 tip（非 main）额外暴露

| 分支 tip | 额外类别 | 代表路径 |
| --- | --- | --- |
| `origin/feat/route-a-completion` | Daymet/gridMET **raw API 快照** | `data_usgs/raw_snapshots/**`（约 492 路径） |
| 同上 | predictor bridge parquet / 报告 | `data_usgs/development_predictor_bridge_v1/**` |
| 同上 | HUC / station registry | `huc_metadata_usgs_v1.csv(.provenance.json)`, `station_registry_v1.csv` |
| `origin/feat/yiqu-upgrade` | 同上类数据 + **legacy figures 仍在树中** | 与 main 类似的 `outputs/figures/**` 等 |

> `feat/route-a-completion` tip 已去掉大量 legacy `outputs/figures|tables|reports`，但 **数据与 AGU class 仍在**；且 raw_snapshots 扩大了暴露面。

### 1.3 Tip 已删除、但 history 仍可达（clone 可得）

| 类别 | 代表对象 | 说明 |
| --- | --- | --- |
| **WIP archive** | `_archive_wip/panel_usgs_100_wip.parquet`, `usgs_predictions_wip.parquet`, `usgs_scores_wip.csv`, … | tip 无路径，但 `git rev-list --objects origin/main` 仍列出 |
| **撤回的 Zenodo 元数据** | 历史 blob `.zenodo.json`（引入于 `2c0b7534`） | tip 上 absent；README：其 open/MIT 数据再分发等主张**未被当前证据授权**，不得复用 |
| **三站相关早期提交** | 如 `d0cd35ee data: raw 3-station cascade + …` | 与 withdrawn three-site interpretations 同属 governance stop 范围 |

### 1.4 明确未在 public tip 暴露（本地/ignore）

以下本地目录多数被 `.gitignore` 挡住，**当前 remote tip 未见**；但若曾误 push 过，仍须用完整 SHA inventory 核验：

- `outputs/runs/**`、`outputs/models/**`、`outputs/predictions/**`、`outputs/logs/**`
- `data_usgs/confirmatory_*`、部分 `n*.csv` / `site_*.csv`
- 本地证据包命名模式：`thermoroute_LOCAL_EVIDENCE_DO_NOT_DISTRIBUTE_*.zip`

---

## 2. MIT vs 数据 / AGU class 缺口

| 层 | 现状 | 缺口 |
| --- | --- | --- |
| **代码 LICENSE** | 根目录 `LICENSE` = MIT；`pyproject.toml` `license = MIT` | 仅覆盖 **Software**；README 已写明数据与第三方条款须另行审查 |
| **三站 case-study 数据** | 已在 public tip 的 `data/*.csv` | **无**来源、采集协议、再分发授权记录；不得因 MIT 而视为可公开再分发 |
| **USGS 观测衍生 panel** | parquet/CSV 在 remote | Provider 端「可下载」**不是**对本仓库已打包字节的再分发授权；**本仓库打包字节** ≠ 再打包/再许可权；attribution / terms 未做成独立 data-license 元数据 |
| **Daymet / gridMET 衍生与 raw_snapshots** | main：衍生列在 panel；feature 分支：原始 `response.bin` 等 | 网格产品条款与 API ToS **未**逐类记录；feature 分支扩大了 raw bytes 暴露 |
| **AGU LaTeX class** | `agujournal2019.cls` 等在所有主要 remote tip | README 要求：再分发条款须写入 third-party notice，**或**从任何 public code archive 中排除；当前 **两者皆缺** |
| **Zenodo / FAIR 发布** | `--distribution PUBLIC` **fail-closed**；deposit 元数据禁用 | 缺：verified creators、分数据类 license、准确 scope；历史 `.zenodo.json` 主张无效 |
| **作者 / DOI / funding（AGU 投稿包）** | `paper/agu_submission/README.md`：author block 故意无效 | 投稿前许可与 DOI 语言未就绪 |
| **FAIR 可发现性** | 无正式 public data release；无 release assets | Findable/Accessible 被 governance stop 阻断；Interoperable/Reusable 缺独立数据许可与 provenance 对 legacy panel 的 raw replay |

**结论（审计启动层）**：MIT 只覆盖代码意图；**数据类别与 AGU class 均未闭合**。在完成逐字节 rights review 前，任何「开源仓库 = 可公开分发数据包」的推断均不成立。

---

## 3. 下一步 SHA-256 inventory 命令草案

目标：对 **每个 remote ref 的 tip 树** + **可达历史对象（可选第二遍）** 生成 digest 清单，供权利审查与 remediation 对照。只读；不改 source hash。

### 3.1 Tip 树按路径 inventory（推荐先做）

```bash
REPO=/home/shutiao/workspace/projects_part_time/project1/thermoroute-water-temperature
OUT="$REPO/outputs/logs/rights_sha256_inventory"
mkdir -p "$OUT"
DATE=$(date -u +%Y%m%dT%H%M%SZ)

# Include every public remote tip named in the header (incl. pull/1/head).
for REF in origin/main origin/feat/route-a-completion origin/feat/yiqu-upgrade \
  origin/fix/delta-leakage-lightgbm-parity-p0p1 origin/pull/1/head; do
  if ! git -C "$REPO" rev-parse --verify "$REF" >/dev/null 2>&1; then
    echo "SKIP missing ref: $REF" >&2
    continue
  fi
  SAFE=$(echo "$REF" | tr '/:' '__')
  LIST="$OUT/${DATE}_${SAFE}_paths.txt"
  DIGEST="$OUT/${DATE}_${SAFE}_sha256.tsv"
  git -C "$REPO" ls-tree -r --name-only "$REF" > "$LIST"
  printf 'sha256\tsize\tpath\n' > "$DIGEST"
  while IFS= read -r path; do
    # blob bytes via git cat-file (no worktree checkout required)
    size=$(git -C "$REPO" cat-file -s "$REF:$path" 2>/dev/null) || continue
    sha=$(git -C "$REPO" cat-file -p "$REF:$path" | sha256sum | awk '{print $1}')
    printf '%s\t%s\t%s\n' "$sha" "$size" "$path" >> "$DIGEST"
  done < "$LIST"
  # Acceptance: digest must have header + ≥1 data row
  rows=$(wc -l < "$DIGEST")
  test "$rows" -ge 2 || { echo "FAIL empty digest: $DIGEST" >&2; exit 1; }
done
```

### 3.2 仅高风险前缀（更快；可直接跑）

```bash
REPO=/home/shutiao/workspace/projects_part_time/project1/thermoroute-water-temperature
OUT="$REPO/outputs/logs/rights_sha256_inventory"
mkdir -p "$OUT"
DATE=$(date -u +%Y%m%dT%H%M%SZ)
REF=origin/main
SAFE=$(echo "$REF" | tr '/:' '__')
LIST="$OUT/${DATE}_${SAFE}_high_risk_paths.txt"
DIGEST="$OUT/${DATE}_${SAFE}_high_risk_sha256.tsv"
git -C "$REPO" ls-tree -r --name-only "$REF" \
  | grep -E '^(data/|data_usgs/|outputs/|paper/|_archive_wip/|\.zenodo\.json)' \
  > "$LIST"
printf 'sha256\tsize\tpath\n' > "$DIGEST"
while IFS= read -r path; do
  size=$(git -C "$REPO" cat-file -s "$REF:$path" 2>/dev/null) || continue
  sha=$(git -C "$REPO" cat-file -p "$REF:$path" | sha256sum | awk '{print $1}')
  printf '%s\t%s\t%s\n' "$sha" "$size" "$path" >> "$DIGEST"
done < "$LIST"
test "$(wc -l < "$DIGEST")" -ge 2 || { echo "FAIL empty high-risk digest" >&2; exit 1; }
```

### 3.3 History 可达对象（含 tip 已删路径）

```bash
REPO=/home/shutiao/workspace/projects_part_time/project1/thermoroute-water-temperature
OUT="$REPO/outputs/logs/rights_sha256_inventory"
mkdir -p "$OUT"
DATE=$(date -u +%Y%m%dT%H%M%SZ)
REF=origin/main
# 列出 origin/main 历史中曾出现的敏感路径对象
# Note: rev-list --objects lines are "<sha> <path>"; do not anchor on ^path.
git -C "$REPO" rev-list --objects "$REF" \
  | grep -E '(_archive_wip/|\.zenodo\.json|[[:space:]]data/|[[:space:]]data_usgs/.*\.(parquet|csv)$|[[:space:]]outputs/(figures|tables|reports)/|agujournal2019\.cls)' \
  > "$OUT/${DATE}_history_sensitive_objects.txt"

# 对 blob SHA 做内容 digest（注意：git blob id ≠ SHA-256 of payload）
printf 'git_blob\tsha256\tsize\tpath\n' > "$OUT/${DATE}_history_sensitive_sha256.tsv"
while read -r blob path; do
  [ -z "$blob" ] && continue
  git -C "$REPO" cat-file -t "$blob" | grep -q blob || continue
  size=$(git -C "$REPO" cat-file -s "$blob")
  sha=$(git -C "$REPO" cat-file -p "$blob" | sha256sum | awk '{print $1}')
  printf '%s\t%s\t%s\t%s\n' "$blob" "$sha" "$size" "$path"
done < "$OUT/${DATE}_history_sensitive_objects.txt" \
  >> "$OUT/${DATE}_history_sensitive_sha256.tsv"
test "$(wc -l < "$OUT/${DATE}_history_sensitive_sha256.tsv")" -ge 2 \
  || { echo "FAIL empty history digest" >&2; exit 1; }
```

### 3.4 与 GitHub 可见性快照一并归档

```bash
REPO=/home/shutiao/workspace/projects_part_time/project1/thermoroute-water-temperature
OUT="$REPO/outputs/logs/rights_sha256_inventory"
mkdir -p "$OUT"
DATE=$(date -u +%Y%m%dT%H%M%SZ)
# Snapshot only — PUBLIC visibility is NOT a redistribution grant.
gh repo view shutiao23/thermoroute-water-temperature \
  --json visibility,isPrivate,defaultBranchRef,pushedAt,forksCount \
  > "$OUT/${DATE}_gh_repo_visibility.json"
git -C "$REPO" ls-remote origin > "$OUT/${DATE}_ls_remote.txt"
```

### 3.5 产出物建议命名

- `outputs/logs/rights_sha256_inventory/<UTC>_<ref>_sha256.tsv`
- 列：`sha256`, `size`, `path`（history 表另加 `git_blob`）
- 本草案文件本身：`outputs/logs/RIGHTS_INVENTORY_DRAFT.md`（已被 `.gitignore` 的 `outputs/logs/` 覆盖，默认不会进 Git）

---

## 4. 不能公开分发的明确列表（当前证据下）

在完成合格权利审查并形成可引用授权之前，下列类别 **明确不得** 作为 public redistribution / Zenodo PUBLIC / 第三方传递包发布（本地 `LOCAL_EVIDENCE_ONLY` 也不构成再分发许可）：

1. **`data/b1.csv`, `data/p3.csv`, `data/s2.csv`（legacy 三站）** — 来源与再分发授权未文档化。  
2. **任意打包了未审查数据类别的 release / Zenodo / mirror 包** — 含历史 `.zenodo.json` 所声称的「open/MIT data」主张。  
3. **`paper/agu_submission/agujournal2019.cls` 及未授权的 AGU/第三方 TeX 附属文件** — 缺 third-party notice 或正式排除决定前，不得进入 public code/data archive。  
4. **已渲染的 PDF/DOCX 与 withdrawn legacy figures/tables/reports** — README 将之排除在 active member namespace 之外；公开再分发前须单独 license/privacy 审查。  
5. **Git-history bundle 中的已删对象**（`_archive_wip/**`、旧 `.zenodo.json`、旧 outputs blob）— 仅作 provenance，不是可再分发内容。  
6. **`data_usgs/**` raw_snapshots / predictor bridge 原始 API 字节** — provider ToS 未逐类核验前不得再分发。  
7. **USGS panel parquet 与衍生 CSV 的「再许可为 MIT」** — MIT 文件头/根 LICENSE 不延伸为数据许可。  
8. **`thermoroute_LOCAL_EVIDENCE_DO_NOT_DISTRIBUTE_*.zip` 及任何 `--distribution PUBLIC` 产物** — PUBLIC 门闩 fail-closed；本地证据包明示禁止分发。  
9. **作者身份未核实、DOI/funding/redistribution 语言未就绪的 AGU 投稿包对外传播** — 见 `paper/agu_submission/README.md`。  
10. **将「GitHub PUBLIC 可读」本身当作分发授权或 FAIR 发布完成的声明**。

---

## 5. 建议的后续动作（非本 Worker 执行范围）

1. Owner + 合格权利审查人：限制 remote 访问策略评估（私有化 / 历史 remediation），**先**完成 §3 inventory 与审计副本。  
2. 为每一数据类填写：来源、许可/ToS URL、attribution、是否允许再打包。  
3. AGU class：取得可引用条款并写 `THIRD_PARTY_NOTICES`，或从公开树移除。  
4. 在权利闭合前：禁止新建 `.zenodo.json`、禁止 PUBLIC archive、禁止把数据说成 MIT。  
5. R0-2 等组若改 `src/scripts/tests/protocols`：本清单勿与之交叉提交，以免污染 source hash。

---

*草案结束。路径：`outputs/logs/RIGHTS_INVENTORY_DRAFT.md`*
