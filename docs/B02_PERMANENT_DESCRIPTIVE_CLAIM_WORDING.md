# B-02 — Permanent fixed-cohort descriptive claim wording (locked)

**Status:** LOCKED / 定稿（2026-07-30，R2-1；监督 Reviewer 最小修补对齐 Worker B）  
**Scope (narrow):** Locks **only** eligibility of the **formal five confirmatory
comparisons** (Route-A five-row family) for confirmatory claim wording in abstract,
conclusion, cover letter, highlights, and the five-row result prose.  
**Does not lock / does not erase:** predeclared exploratory or secondary analyses
(architecture controls, external-site cohort, temporal-coverage sensitivities,
probability diagnostics, perturbations, unofficial baselines, etc.). Those remain
reportable as exploratory / descriptive under their existing labels.  
**Does not modify:** `src/`, `scripts/`, `tests/`, `protocols/` (source-hash boundary).  
**Authority:** Outcome-free inference amendment small-cluster gate
(`n_clusters >= 30`, `effective_cluster_fraction >= 0.75`,
`largest_cluster_share < 0.25`); frozen cohort ≤ 15 HUC2; inverse-Herfindahl
effective cluster count ≈ 9.54.

## Permanent verdict (English)

Route A's **formal five confirmatory comparisons** are **permanently**
`DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED`: fixed-cohort descriptive effects only
for confirmatory claim eligibility. This is decided by the frozen cohort geography
**before any target outcome is viewed** and is not reopened by opening results,
favorable point estimates, or sensitivity tables. The gate does **not** delete or
rebrand exploratory / secondary work.

## Permanent verdict (中文)

Route A 的**正式五比较 confirmatory 行**在 claim eligibility 上**永久**裁定为
`DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED`：仅允许固定 cohort 的描述性效应。
该裁定由冻结 cohort 地理结构在**查看任何目标期 outcome 之前**即已确定；不得因
opening 数值、有利点估计或敏感度表而重新升级为 confirmatory 措辞。门控**不抹掉**
exploratory / secondary 分析，亦不禁止其按既有标签继续报告。

## Gate facts (must stay aligned)

| Fact | Value |
|---|---|
| Max HUC2 groups in frozen cohort | ≤ 15 |
| Inverse-Herfindahl effective cluster count | ≈ 9.54 |
| Gate: `n_clusters` | ≥ 30 (fails) |
| Gate: `effective_cluster_fraction` | ≥ 0.75 (fails; ≈ 9.54/15 ≈ 0.636) |
| Gate: `largest_cluster_share` | < 0.25 (cohort largest share ≈ 0.217 may pass this alone, but the conjunction fails) |
| What the gate locks | Five-comparison **confirmatory claim eligibility** only |
| Eligible five-row confirmatory verdict | `DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED` only |
| Exploratory / secondary | Remain; not erased; not upgraded to confirmatory |
| Post-opening rewrite ban (five rows) | No superiority / non-inferiority / equivalence / parity / national or U.S.-river superpopulation claim |
| p / CI / Holm as decision evidence | **Forbidden** for confirmatory claim decisions |

## Paste-ready — Abstract / Conclusion (English)

Because the frozen Route-A cohort contains at most 15 HUC2 groups (inverse-Herfindahl effective cluster count ≈ 9.54) and therefore fails the outcome-free inference gate requiring n_clusters ≥ 30, effective_cluster_fraction ≥ 0.75, and largest_cluster_share < 0.25, the formal five confirmatory comparisons are permanently restricted to fixed-cohort descriptive effects under DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED for confirmatory claim eligibility; exploratory and secondary analyses remain reportable as such and are not erased by the gate. Whole-HUC2 bootstrap intervals, exact sign-flip p-values, and Holm adjustments are assumption-conditional sensitivities only and must not be written as confirmatory decision evidence; after opening they must not be rewritten as superiority, non-inferiority, equivalence, parity, or national / U.S.-river generalization.

## Paste-ready — 摘要 / 结论（中文）

由于冻结的 Route-A cohort 至多只有 15 个 HUC2 组（逆 Herfindahl 有效簇数 ≈ 9.54），必然无法通过 outcome-free 推断门控（n_clusters ≥ 30、effective_cluster_fraction ≥ 0.75、largest_cluster_share < 0.25），因此正式五比较在 confirmatory claim eligibility 上永久限定为固定 cohort 的描述性效应，裁定为 DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED；exploratory / secondary 分析仍可按既有标签报告，门控不抹掉它们。全 HUC2 bootstrap 区间、精确 sign-flip p 值与 Holm 校正仅可作为假设条件敏感度，禁止写成 confirmatory 决策证据；开封后禁止改写为 superiority、non-inferiority、equivalence、parity，或全美/全国河流总体推广结论。

## Allowed phrases (执行派白名单)

- fixed-cohort descriptive effect / 固定 cohort 描述性效应
- `DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED`
- assumption-conditional sensitivity / 假设条件敏感度
- permanently descriptive (five confirmatory rows only) / 五比较永久描述性
- at most 15 HUC2; effective cluster ≈ 9.54
- gate cannot pass independent of outcomes / 与 outcome 无关、门控必然失败
- exploratory / secondary remains labelled exploratory or descriptive / exploratory 仍保留

## Forbidden phrases after opening (五比较 confirmatory 黑名单)

- superiority / non-inferiority / equivalence / parity（作为正式 confirmatory 结论）
- confirmatory evidence of superiority（或同义升级）
- nationally representative / national generalization / U.S.-river superpopulation
- “gate nearly passes” / “effectively confirmatory” / “practically national”
- treating p-values or CIs as confirmatory **decision evidence** / 把 p 或 CI 写成决策证据

## Forbidden verbs (禁止动词清单) — five confirmatory rows only

Do **not** use these verbs (or close synonyms) to upgrade the five-row descriptive
verdict into a confirmatory claim after opening:

| EN (base forms) | ZH gloss / 近义禁写 |
|---|---|
| confirm / confirms / confirmed | 确认、证实（作 confirmatory 结论） |
| demonstrate / demonstrates / demonstrated | 证明、表明（升级为正式成立） |
| establish / establishes / established | 确立、成立 |
| support / supports / supported（as confirmatory support） | 支持（作决策性支持） |
| achieve / achieves / achieved | 达成（优越/非劣等） |
| outperform / outperforms / outperformed | 优于、胜过 |
| prove / proves / proven | 证明 |
| validate / validates / validated（as confirmatory validation of superiority） | 验证为正式优越 |
| show superiority / show non-inferiority | 显示优越 / 显示非劣 |
| conclude equivalence / conclude parity | 断定等价 / 断定持平 |

Reporting exploratory / secondary results with ordinary descriptive verbs remains
allowed when the label stays exploratory or descriptive.

## Related in-repo anchors (read-only reminders)

- `paper/ThermoRoute_paper.md` Abstract L40–44; §4.2 L377–387; Conclusion L643–645
- `paper/cover_letter.md` L44–51
- `paper/highlights.md` L23, L49–51
- Claim template `DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED` in
  `protocols/route_a_claim_registry_v1.json` (do not edit from this ticket)
