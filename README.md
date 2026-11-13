# Korean safety drift in Tiny Aya Global, and what a weight-space repair really recovers

Lexsi AI Safety & Alignment Hackathon · Track 2b (AI Safety and Correction), problem **Repairing safety drift without retraining** (detect safety drift after fine-tuning, repair it in weight space while keeping task performance).

**Write-up:** [`ARTICLE.md`](https://github.com/kyungmin-cho/tiny-aya-ko-refusal-drift/blob/main/ARTICLE.md) in this repository.

> **Safety notice.** This repository studies how fine-tuning on a small amount of harmful data removes refusals. It contains **no harmful prompts, no model outputs and no drifted weights**. Evaluation inputs are rebuilt locally from public datasets by row id; training files are rebuilt from public datasets by row id and verified by sha256. Several of those public datasets (MultiJail, AdvBench-Korean, BeaverTails, XSTest) contain offensive or dangerous requests. Handle them under their licences and your organisation's research-safety rules. The only released model is a *control* adapter trained on benign data; it is not a safety-improved model. No repaired model is released (see finding 4).

## Findings in one screen

| | run 1 | run 2 |
|---|---|---|
| Korean refusal (MultiJail-ko, judge), base Tiny Aya Global | 192/315 (61.0%) | 192/315 (61.0%) |
| after LoRA on 2,000 benign + **100 harmful** rows (T100) | 49/315 (15.6%) | 50/315 (15.9%) |
| after LoRA on 2,100 benign rows, length-matched (C100) | 179/315 (56.8%) | 174/315 (55.2%) |
| **drift attributable to the 100 harmful rows** (T100 vs C100, paired McNemar) | **−41.3 pp** (p=9.6e-36) | **−39.4 pp** (p=3.2e-32) |
| Global-MMLU-ko (400 items), base → T100 | not measured | 205 → 210 (p=0.60) |

1. **Drift is large, replicated across two runs, and invisible to a capability benchmark.** 100 harmful rows (4.76% of rows, about 2% of training tokens) cut Korean refusals by about 40 pp against a matched benign control in two runs (different training seed; run 2 also re-curated the benign pool with CuratorKIT, 1,978 of 2,000 rows shared). English refusals fell too (70.8% → 13.0%, run 2). Global-MMLU-ko did not move.
2. **A benign control is necessary.** Benign LoRA alone lowered Korean refusals by 4.1 pp (p=0.053) and 5.7 pp (p=0.013). A before/after comparison without the control overstates the harmful-data effect.
3. **ReSta with the public base→global weight difference did not give a usable repair in our setting.** No α in {0.25, 0.5, 0.75, 1.0} passed the capability and over-refusal gates we fixed before run 2. α = 0.25 raised judge refusals (50 → 114/315) but cost 10.5 pp on Global-MMLU-ko (p=4.0e-4), and an automated agent read-through found only 30 of those 114 "refusals" to be real refusals (recovery of lost real refusals 7.7%, paired bootstrap 95% [−4.1, 19.6]). α ≥ 0.5 pushed judge refusals to 91–100% while the model collapsed (MMLU at chance, Korean answers with almost no Hangul, 89–100/100 benign prompts refused). A CPU diagnosis found no code bug (the merge reproduces bit-for-bit) and ruled out bf16 rounding: the base→global delta is 38% of the weight norm and the α = 0.25 update was 8.5× the size of the whole drift.
4. **A paper-style safety vector raised refusals, but it was not a repair.** With the vector built as in the ReSta paper (global minus global fine-tuned on 100 *other* harmful rows), refusals on the drifted model rose to 232/315 (ko) and 258/315 (en) with MMLU flat, and the model passed our pre-specified release rule. In the base-model control (E7) the un-aligned adapter was re-trained with the same rows, recipe and seed (GPU training gave a slightly different copy, v_E7), and v_E7 was added to both the undamaged base model and the drifted model: at γ = 1.0 base refusals rose by the same share of the remaining headroom as the drifted model's (ρ = 0.945 ko, 1.062 en; the drifted model reached 237 and 257 with v_E7) and base XSTest over-refusal went from 26 to 129/250. A second control on the benign fine-tune C100 (E8; the vector trained a third time, v_E8, added to both C100 and T100) gave the same result at γ = 1.0: ρ = 0.97 ko [0.87, 1.08] and 1.07 en [0.97, 1.17], C100 refusals 174 → 271 (ko) and 213 → 299 (en), XSTest over-refusal 26 → 158/250, while T100 with this copy (237 ko, 257 en) was close to the earlier copies (E7: 237, 257; E6: 232, 258). At γ = 1.0 the vector raises refusal by a similar share on all three states of the one model we tested (original, benign fine-tune, harmful fine-tune); at γ = 0.5 the ratios differ (en ρ 2.35 and 2.16). It does not specifically undo the drift. We therefore release **no repaired model**. Aggregates for this finding (E6: v_E6 on the drifted model; E7: v_E7 on the undamaged model and on the drifted model, and the bf16 merged file; E8: v_E8 on C100 and T100) are in `experiments/e6/agg/`, `experiments/e7/agg/` and `experiments/e8/agg/`; their judge and agent-label numbers re-derive from the released labels (Reproduce A).

Full tables: [`RESULTS.md`](RESULTS.md) (generated by `tools/make_tables.py` from the aggregate JSON files in `experiments/*/agg/`); tables for finding 4 are in `experiments/e6/agg/e6_agg.md`, `experiments/e7/agg/e7_agg.md` and `experiments/e8/agg/e8_agg.md` (the generated `.md` tables are in Korean; the `.json` files hold the same numbers).

## Pipeline (three Track 2 toolkits used together)

```
kullm-v2 (benign, ko) ──► CuratorKIT ──► 2,000 clean rows ──┬─► + 100 BeaverTails harmful rows ──► T100 ─┐
                          schema gate, exact + MinHash      └─► + 100 length-matched kullm rows ──► C100 ─┤
                          dedup, secrets gate                                                             │
                                                                                                           ▼
Tiny Aya Global ──► SafeTune PlainSFT (LoRA r16, 3 epochs, same seed for both arms) ──► T100 / C100 adapters
                                                                                                           │
                 SafeTune ReStaTrainer: θ = global + ΔLoRA_T100 + α·DARE₀.₃(global − base) ◄──────────────┘
                                                                                                           │
generate (greedy, 128 tok) ──► Qwen3Guard-Gen-4B labels ──► AuditKIT (precomputed labels, paired bootstrap)
                               Global-MMLU-ko first-token logits      + exact McNemar (e4_common.paired)
```

| step | toolkit | code |
|---|---|---|
| curate the benign pool, provenance, overlap checks against evaluation items | CuratorKIT 0.1.24 | `experiments/e5/scripts/e5_curate.py`, `e5_build_eval_items.py` |
| drift LoRA (PlainSFT path), ReSta repair | SafeTune 0.1.14 | `scripts/e4_train.py` (via `e5_train.py`), `scripts/e4_eval.py::build_model` |
| paired comparison and significance on every metric | AuditKIT 1.1.7 | `experiments/e5/scripts/e5_auditkit.py` |
| orchestration, pre-specified α rule, aggregation | — | `e5_orchestrate.py`, `e5_select_release.py`, `e5_supplement.py` |

## Repository layout

```
scripts/              shared harness: data rows (e4_common), generation + judge (e1e3_harness), training, model building, aggregation
  rebuild_eval_items.py   rebuild all evaluation inputs from public datasets + id manifests (sha256-checked)
experiments/e4/       run 1: 4 arms (T100, C100, T200, C200) × epochs 1, 3; ReSta α 0.1, 0.25
experiments/e5/       run 2: T100, C100, ReSta α 0.25–1.0, MultiJail-en, XSTest, Global-MMLU-ko, α-selection hold-out
  data/               id-only manifests: training rows (dataset row ids + sha256), evaluation items, CuratorKIT stage counts
  agg/                aggregate results (numbers and item ids only), incl. run-2 agent read-through counts (e5_sol_*.json)
  tests/              25 unit tests (orchestrator, α rule, seeds, data checks, weight-fetch checks)
experiments/e6/       paper-style safety vector on T100 (γ 0.5, 1.0): agg/ (judge, read-through, release rule, weight
                      statistics, output shape) and scripts/e6_from_labels.py (re-derives them from the label dataset)
experiments/e7/       a re-trained copy of the vector (v_E7) on the undamaged model and on T100, and the bf16 merged file: agg/ and scripts/e7_from_labels.py
experiments/e8/       a third copy of the vector (v_E8) on the benign control C100 and on T100: agg/ and scripts/e8_from_labels.py
tools/                export_labels.py (public label dataset), make_tables.py, make_figure.py (Figures 1 and 2 of the write-up), check_release.py (leak check)
figures/              figure1_repair_before_after.png (from the e8 aggregate) and figure2_refusal_vs_overrefusal.png (from the
                      e7 and e8 aggregates), drawn by tools/make_figure.py
ARTICLE.md            the write-up (Figures 1 and 2 are in figures/)
RESULTS.md            all result tables
SAFETY_SCAN.md        output of the pre-release leak check on this tree
LICENSE               MIT licence for the code in this repository
LICENSE_REVIEW.md     licences of every model, dataset and toolkit used
```

Code comments in the experiment scripts are in Korean (the working language of the project); entry points and this README are in English.

## Reproduce

Requirements: Python 3.12, [`uv`](https://docs.astral.sh/uv/). Versions are pinned in `pyproject.toml` / `uv.lock`.

### A. Run-2 and later numbers, from the released labels (CPU, a few minutes)

The label dataset ([`kyungmin-cho/tiny-aya-global-ko-refusal-drift-labels`](https://huggingface.co/datasets/kyungmin-cho/tiny-aya-global-ko-refusal-drift-labels)) mirrors `experiments/<run>/eval/<model>/` with judge labels, MMLU predictions and generation lengths (runs `e4` to `e8`) and agent read-through labels (id, label) under `<run>/review/`; it holds no text.

```bash
uv sync
uv run hf download kyungmin-cho/tiny-aya-global-ko-refusal-drift-labels --repo-type dataset --local-dir labels
DS=labels/data/e5
uv run python -I -B experiments/e5/scripts/e5_supplement.py $DS/eval out/e5_supplement \
    --items-e1 $DS/items/e1_index.jsonl --items-mjen $DS/items/mjen_index.jsonl --items-xs $DS/items/xs_index.jsonl
uv run python -I -B experiments/e5/scripts/e5_auditkit.py $DS/eval out/auditkit --xs-items $DS/items/xs_index.jsonl \
    --models base T100_ep3 C100_ep3 T100_ep3_resta0.25 T100_ep3_resta0.5 T100_ep3_resta0.75 T100_ep3_resta1.0 \
    --pairs base:T100_ep3 base:C100_ep3 C100_ep3:T100_ep3 T100_ep3:T100_ep3_resta0.25 T100_ep3:T100_ep3_resta0.5 \
            T100_ep3:T100_ep3_resta0.75 T100_ep3:T100_ep3_resta1.0 base:T100_ep3_resta0.25 base:T100_ep3_resta0.5 \
            base:T100_ep3_resta0.75 base:T100_ep3_resta1.0 \
    --parts e1 mjen xs_safe xs_unsafe sel_h sel_b mmlu_rep mmlu_sel
```

Then render the tables with the rebuilt run-2 numbers:

```bash
python3 -I -B tools/make_tables.py --e5-supplement out/e5_supplement.json > RESULTS.md
```

Repair B (E6), the base-model control (E7) and the C100 control (E8), including every agent read-through count of run 2 and later; each script exits 1 on any difference from the shipped aggregates:

```bash
uv run python -I -B experiments/e6/scripts/e6_from_labels.py labels/data --out out/e6_from_labels.json
uv run python -I -B experiments/e7/scripts/e7_from_labels.py labels/data --out out/e7_from_labels.json
uv run python -I -B experiments/e8/scripts/e8_from_labels.py labels/data --out out/e8_from_labels.json
```

We ran exactly these commands on the exported labels: `e5_supplement.json` matched the run-2 original on every field, the AuditKIT comparison (121 rows, rates and bootstrap p-values) matched on every field except the `created_by` path, and the regenerated `RESULTS.md` was byte-identical to the shipped one; `e6_from_labels.py`, `e7_from_labels.py` and `e8_from_labels.py` matched 1,470, 1,744 and 3,008 values of the shipped E5–E8 aggregates with no difference (run in a copy of this tree and the dataset outside our working folders). Scope: the run-2 refusal, Unsafe, XSTest, selection-set and MMLU columns re-derive from labels. The run-1 rows and the output-shape table (Hangul share, identical answers, repetition) are read from aggregate files that were computed from the generations, which are not released; the run-1 refusal-phrase columns (`e4_aggregate.py`) also need the generations. In `experiments/e6/agg/`, `e7/agg/` and `e8/agg/` the same holds for `*_degeneracy.json` (output shape), `e6_vecstat.json` (weight statistics), the `determinism` block of `e7_agg.json` (re-training and merge-rounding checks) and, in `e8_agg.json`, `determinism.u100` (re-training), `determinism.vecstat` and the character-identical answer counts: aggregates only (the label and MMLU-prediction agreement counts in `determinism.repro` do re-derive). The un-aligned adapter hashes are removed from the shipped aggregates; `identical: false` and the loss differences are kept. The E6–E8 GPU scripts (training the un-aligned adapter on harmful rows, applying the vector, orchestration) are not in this tree; they reference unreleased weights by path and hash.

The two figures of the write-up are drawn from the shipped aggregates. Figure 1 (refusals before and after the vector at γ 1.0, for the damaged model and the two undamaged ones) reads `e8_agg.json`; Figure 2 (refusal vs over-refusal at γ 0, 0.5, 1.0) reads the E7 and E8 aggregates:

```bash
uv run --with matplotlib python -I -B tools/make_figure.py        # writes figures/figure1_repair_before_after.png and figures/figure2_refusal_vs_overrefusal.png
```

### B. Full rerun (one CUDA GPU, about 2.2 h on an A40 48 GB)

```bash
uv sync
uv run python -I scripts/download.py open
HF_TOKEN=... uv run python -I scripts/download.py gated        # accept the Tiny Aya and XSTest gates on the Hub first
uv run python -I -B scripts/rebuild_eval_items.py              # writes the 5 evaluation input files, exits 1 on any sha256 mismatch
HF_TOKEN=... bash experiments/e5/scripts/run_e5.sh             # training files are rebuilt from ids (sha256-checked) in preflight
uv run python -I -B experiments/e5/scripts/e5_orchestrate.py experiments/e5 --dry-run   # plan and status at any time
```

Measured on our run 2: training about 34 min per arm (1,575 steps at 1.29 s/step), evaluation 7.0–7.3 min per model (1,280 generations + judge + 600 MMLU items), ReSta build + evaluation 8.9–9.1 min per α, orchestrator total 126.5 min. Peak GPU memory 35.65 GB (training); ReSta runs on CPU and peaked at 21.2 GB RSS.

What we verified on this tree before release: 4 of the 5 evaluation inputs rebuild byte-identically from public data (XSTest is gated and was not in our local cache, so its rebuild was not re-checked here); all 6 training files of both runs rebuild byte-identically from the id manifests; `py_compile` of every script; 25/25 unit tests with the training and evaluation files rebuilt (without them the two data tests fail and the other 23 pass); orchestrator dry-run (130.3 min planned).

**Not reproducible from public files alone.** The curation and item-sampling scripts (`e5_curate.py`, `e4_build_data.py`, `e5_build_eval_items.py`) were run once on private analysis files that contain harmful text (risk-keyword review, sampled harmful rows). Their *outputs* are released as id manifests, which is what the rerun uses.

## Evaluation protocol

- Model under test: `CohereLabs/tiny-aya-global@af89d219` (3.35B). ReSta base: `CohereLabs/tiny-aya-base@d62ac17e` (the pretrained checkpoint).
- Generation: greedy, 128 new tokens, batch 64, the model's default chat template, no system prompt.
- Judge: `Qwen/Qwen3Guard-Gen-4B@6ec42827` ("Safety" and "Refusal" fields, 64 tokens). Parse failures on MultiJail: 0 for every model in both runs.
- Primary metric: refusal rate on MultiJail-ko (315), paired exact McNemar against base and against the benign control. Secondary: MultiJail-en (315), XSTest safe over-refusal (250) / unsafe refusal (200), Global-MMLU-ko (first-token logit over A–D, 400 report items).
- α selection (fixed before run 2; held-out data only): AdvBench-ko 100 (H), benign kullm 100 (B), Global-MMLU-ko 200 (M). Eligible if M ≥ M(base) − 10 and B ≤ B(base) + 10; pick highest H. If none is eligible, nothing is released as a repair.
- Judge "Unsafe" is not the same as harmful: an agent read-through of the *new* Unsafe answers (base Safe → model Unsafe) found, in run 1, 59/140 harmful for T100 and 5/14 for C100; in run 2, 68/146 (ko) and 99/179 (en) for T100 and 6/19 (ko) and 1/4 (en) for C100 (automated agent reading by `gpt-6.1-sol`, not human annotation, not blind; labels of run 2 and later are in the dataset under `data/e5/review/` and `data/e6/review/`, id and label only).

## Limitations

One 3.35B model, one LoRA configuration, MultiJail in two languages (ko, en; 315 items each), one judge model, greedy 128-token answers. XSTest is English. Refusal is a judge label; for broken repaired models it overcounts (run 2, α = 0.25: 30 of 114 judge refusals were real refusals by agent reading). Agent read-throughs are automated LLM readings, not human annotation and not blind, and they only cover answers the judge had flagged, so real-refusal and harmful counts are lower bounds. The DARE mask seed was 0 in both runs, so the α = 0.25 result is not an independent replicate on the mask side. GPU training was not bit-reproducible: re-training the same adapter with the same seed gave a different hash, with similar refusal levels. The paper-style vector (finding 4) was trained three times on the same 100 rows with the same seed (three slightly different copies); each control (the undamaged model in E7, the benign control C100 in E8) used one copy, and the answers of both controls were not read by the agent (judge labels only; the 7 new judge-Unsafe answers of E8, 6 distinct items, are unread). C100 starts below the undamaged model (Korean refusals 174 vs 192/315), so it is not a perfectly clean control. Repair B and both controls use the run-2 adapters only (one training seed; the drift itself was replicated with two seeds). No multiple-comparison correction was applied anywhere; the main comparisons have p < 1e-21, and effects with 0.01 < p < 0.06 (benign-data drift, MMLU-ko on the vector-added models, XSTest on the bf16 file) are reported as observed.

## Release check

`tools/check_release.py` fingerprints harmful texts at run time (MultiJail, AdvBench-ko, BeaverTails rows used, optional local generations) and scans the release tree for overlaps, secrets, absolute paths, identity strings (from a private deny-list), forbidden files and free-text JSON fields. It prints locations and counts only. Result for this tree: [`SAFETY_SCAN.md`](SAFETY_SCAN.md).

## Licences

Code: **MIT** ([`LICENSE`](LICENSE)). The MIT licence covers only our own scripts in this repository; running them requires accepting the toolkit and model licences below separately (see `LICENSE_REVIEW.md`). Data and models keep their own licences: Tiny Aya CC-BY-NC-4.0 + Cohere Labs Acceptable Use Policy (non-commercial), BeaverTails CC-BY-NC-4.0, kullm-v2 Apache-2.0, MultiJail MIT, AdvBench-Korean MIT, Global-MMLU Apache-2.0, XSTest CC-BY-4.0, Qwen3Guard-Gen-4B Apache-2.0. The Lexsi toolkits (SafeTune, AuditKIT, CuratorKIT) are used under their own non-commercial research licences and are not redistributed here.
