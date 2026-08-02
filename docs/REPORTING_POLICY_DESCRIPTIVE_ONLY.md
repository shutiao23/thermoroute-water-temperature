# Reporting policy — descriptive-only after opening（R2-3）

| 字段 | 值 |
| --- | --- |
| 角色 | R2-3 Worker B（对抗式报告政策） |
| 对齐 | `docs/B02_PERMANENT_DESCRIPTIVE_CLAIM_WORDING.md`（B-02 LOCKED） |
| 日期 | 2026-07-30 |
| 范围 | 只写 `docs/`；**不**改 `protocols/`、`src/`、`scripts/`、`tests/`；**不** commit |
| 效力 | 文稿 / 摘要 / 结论 / cover letter / highlights / 五比较正文的**报告政策**；**不是**协议冻结，也**不是**把当前 15-HUC2 Route A 升级为 confirmatory |

> **永久前提（不得弱化）**：Route A 冻结 cohort ≤ 15 HUC2（逆 Herfindahl 有效簇数 ≈ 9.54）在 outcome-free 门控下**必然失败**。正式五比较 confirmatory claim eligibility **永久**为 `DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED`。Route A 跑完 = descriptive benchmark，**不是** confirmatory 完成。

---

## 1. 开封后：允许写法

仅当标签与门控裁定一致时，允许：

| 允许 | 说明 |
| --- | --- |
| fixed-cohort descriptive effect / 固定 cohort 描述性效应 | 五比较主行的唯一 claim eligibility 口径 |
| `DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED` | 五比较正式裁定；可粘贴 B-02 中英 paste-ready 段 |
| assumption-conditional sensitivity / 假设条件敏感度 | 用于标注 bootstrap CI、sign-flip p、Holm 等**数值可出现但非决策证据** |
| exploratory / secondary（显式标签） | 架构对照、外部站点、时间覆盖敏感度、概率诊断、扰动、非官方基线等 |
| gate facts 重述 | ≤15 HUC2；有效簇 ≈ 9.54；`n_clusters≥30` 等门控失败事实（outcome-free） |
| 「门控与 outcome 无关、必然失败」 | 防止开封后用结果回改 eligibility |

数值（效应点估计、区间、p）可以**作为描述性或敏感度数字出现**，但必须同时满足 §3：不得写成 confirmatory 决策证据。

---

## 2. 开封后：禁止写法（五比较 confirmatory 行）

与 B-02 黑名单对齐；下列任一出现即视为报告缺陷：

| 禁止 | 近义陷阱 |
| --- | --- |
| superiority / non-inferiority / equivalence / parity 作为正式结论 | 「优效成立」「非劣成立」「等价/持平」 |
| confirmatory evidence of … | 「证实」「确认优越」 |
| nationally representative / U.S.-river superpopulation | 「可推广全美河流」「全国代表」 |
| gate nearly passes / effectively confirmatory / practically national | 「几乎通过门控」「实质 confirmatory」 |
| 把 p / CI / Holm 当决策证据 | 「Holm 显著故优越」「CI 不含 0 故确认」 |
| Route A 完成 = confirmatory 完成 | 「正式推断已通过」「15-HUC2 已可 confirmatory」 |

**禁止动词**（升级五比较描述性裁定时）：confirm、demonstrate、establish、support（作决策性支持）、achieve、outperform、prove、validate（作 confirmatory 验证）、show superiority / show non-inferiority、conclude equivalence / conclude parity。完整表见 B-02。

---

## 3. p / CI / Holm 规则（硬禁决策化）

| 物件 | 允许角色 | 禁止角色 |
| --- | --- | --- |
| whole-HUC2 bootstrap CI | assumption-conditional sensitivity；描述性展示 | confirmatory 决策证据；优效/非劣/等价判定依据 |
| exact sign-flip raw p | 同上 | 同上 |
| Holm-adjusted p | 同上 | 同上；「多重校正后仍显著故确认」 |
| 任何「p&lt;α ⇒ claim」模板 | **禁止**用于五比较 confirmatory eligibility | — |

开封后**不得**因有利点估计、狭窄 CI、或小 p 而把五比较从 `DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED` 改写为 `SUPERIORITY_SUPPORTED` / `NONINFERIORITY_SUPPORTED` 等 confirmatory 模板口径（那些模板存在于协议注册表中，**不等于**当前 cohort 有资格使用）。

---

## 4. Exploratory 标签规则

1. **必须显式标注** `exploratory` 或 `secondary`（或中文「探索性 / 次要」）；不得靠版面位置暗示其为正式五比较。  
2. **不得**把 exploratory 结果提升为 confirmatory 摘要句、highlights 主句、或 cover letter 的正式 claim。  
3. 门控**不抹掉** exploratory / secondary；也**不**自动把它升级。  
4. exploratory 可用普通描述动词，但若句子读起来像五比较正式结论，必须降级措辞并加标签。  
5. 禁止用 exploratory 的 p/CI「补救」五比较门控失败（「探索性分析显著，故总体可 confirmatory」——否决）。

---

## 5. 与 Route A / Route B 边界

| 陈述 | 政策裁定 |
| --- | --- |
| 当前 15-HUC2 fixed cohort | **仅** descriptive benchmark；**不可** confirmatory |
| Route A 跑完 / Stage09 产物 | 不改变 B-02；不暗示权利或推断放行 |
| 未来 ≥30 cluster Route B（若另案预注册） | **不得**回溯污染或改写当前 Route A 五比较措辞；本政策不授权立刻改 `protocols/` |
| 本文件 | reporting policy 草稿级落地；**非**协议冻结 |

---

## 6. 粘贴检查清单（开封后文稿自检）

- [ ] 摘要/结论是否仅使用 B-02 paste-ready 或等价 descriptive 口径？  
- [ ] 是否出现 superiority / non-inferiority / equivalence / parity / national 推广？  
- [ ] p/CI/Holm 是否被写成「因此确认 / 因此非劣」？  
- [ ] exploratory 是否都有标签且未进正式五比较 claim？  
- [ ] 是否暗示「15-HUC2 已可 confirmatory」或「门控几乎通过」？  
- [ ] 是否把 Route B 设计草稿写成已冻结协议或已放行？

任一项失败 → 改写后再提交，不「几乎合规」放行。

---

## 7. 本政策删除的危险措辞

| 已砍 / 禁止 | 替换为 |
| --- | --- |
| confirmatory results (对本 Route A 五比较) | fixed-cohort descriptive effects / `DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED` |
| statistically confirms / Holm confirms | assumption-conditional sensitivity only |
| ready for inferential claims | gate failed; descriptive only |
| Stage09 complete ⇒ confirmatory OK | Stage09 / Route A = descriptive benchmark only |
| exploratory implies confirmatory support | exploratory remains exploratory |
| protocol may be updated now to unlock confirmatory | 不授权改协议；另案 outcome-free 预注册后方可讨论 |

---

*R2-3 结束。对齐 B-02；当前 15-HUC2 永不可写作 confirmatory。*
