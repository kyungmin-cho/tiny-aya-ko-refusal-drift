# Pre-release leak check (SAFETY_SCAN)

Tool: `tools/check_release.py` (this repo). Date of run: 2026-11-14 (last re-run after four overview diagrams, drawn by `tools/make_overview_figures.py`, were added to the README, the write-up and the two Hub cards; before that after the write-up was added to this repository as `ARTICLE.md` and the links were pointed to it; before that after the write-up gained a new Figure 1, drawn by the same script, and the earlier figure became Figure 2; before that after the model card's licence and access notes were finalised; before that after the repository names and links were filled in; before that after the review fixes: new path rules, the released C100 adapter weights added to the model upload folder, Figure 1 and its script; before that after the copyright line of `LICENSE` was set to the author's name; before that after adding the E8 labels and aggregates; before that after adding the E6/E7 labels and aggregates and the run-2+ agent labels; first run 2026-11-12). Scanned trees: this repository (`repo`), the Hugging Face card and dataset drafts (`hf`), and the article draft (`article`).

How it was run (paths to private files abbreviated):

```bash
uv run python -I -B tools/check_release.py repo hf article \
    --deny-file <private identity list, outside the repo> \
    --allow-field e5/agg/alpha_selection.json:text e7/agg/e7_agg.json:text e8/agg/e8_agg.json:text \
    --allow-handle <account handle> \
    --allow-file hf/model_C100/adapter_model.safetensors=9b466ab08bafe4a291090894c330e7683cb0ef7bfdb4a300a526485f2ac5ea0c \
    --allow-hit repo/LICENSE:private_long:xs_gen repo/scripts/e4_pod_download.py:private_long:sel_gen \
                repo/LICENSE:deny_list_entry_2 repo/LICENSE:deny_list_entry_4 \
    --private-jsonl '<runs>/e1e3/*.jsonl' '<runs>/e4/eval/*/*.jsonl' '<runs>/e5/eval/*/*.jsonl' \
                    '<runs>/e4/data/*.jsonl' '<runs>/e5/data/*.jsonl' '<runs>/e5/data/eval_items/*.jsonl' \
                    '<runs>/e4/samples/*.jsonl' '<runs>/e5/samples/*.jsonl' '<runs>/e4-sol-review/*.jsonl' \
                    '<runs>/e6/eval/*/*.jsonl' '<runs>/e7/eval/*/*.jsonl' '<runs>/e6/samples/*.jsonl' \
                    '<runs>/e8/eval/*/*.jsonl'
```

The private JSONL files are every generation, evaluation input, training file and review sample of both runs, plus every generation, judge input and review sample of E6 and E7, and every generation and judge input of E8. The 100 BeaverTails rows used to train the E6–E8 un-aligned adapters (the same rows each time) are not fingerprinted (the tool reads harmful row ids only from the released run manifests); the E6–E8 files in this release carry ids, labels and numbers only; benign-by-construction rows (benign kullm training rows, XSTest safe prompts, the benign selection half, Global-MMLU) are skipped so that ordinary text does not raise false alarms. The allowed field `text` in `alpha_selection.json` holds the docstring of the selection rule; in `e7_agg.json` and `e8_agg.json` it holds fixed decision sentences written in the E7 and E8 code (one per outcome category; each of the four E8 values was checked to equal one of these constants). The four `--allow-hit` entries are two fingerprint hits we reviewed and the intended copyright line of `LICENSE` (two deny-list entries; see below); they are listed in the result but not counted as failures, and they apply to that one file and that one source kind or deny-list entry only. Only deny-list entries can be allowed among the pattern rules; secrets, paths, e-mail and IP addresses cannot. The single `--allow-file` entry is the released C100 adapter weight file in the model upload folder; it passes only while its sha256 matches the trained adapter and is listed in the result. The `--allow-handle` entry is the Hugging Face / GitHub account name that appears in the repository URLs and repo ids; it is removed from deny-list matching only where it is followed by `/` (a path segment), so the same name written as prose is still caught; the result lists how often it was removed per file. XSTest unsafe prompts were fingerprinted from the local evaluation input file (the gated XSTest snapshot was not in the local cache).

## Controls

Before trusting a PASS we planted test files outside the release tree (content never printed) and ran the same command:

| planted file | content | detected |
|---|---|---|
| plain.md | first 30 characters of a MultiJail-ko prompt | yes (multijail_ko and local copies) |
| short.md | the shortest MultiJail-ko prompt (5 normalised characters), whole | yes (multijail_ko) |
| escaped.json | 60 characters of a model answer, JSON with `\uXXXX` escaping | yes (local generations) |
| spaced.txt | 60 characters of a model answer with " . " between every character | yes (local generations) |
| clean.md | one sentence of our own prose | no (correct) |
| renamed copy of `scripts/e4_pod_download.py` (2026-11-13) | our script, one of whose exception-handling lines also occurs in a model answer | yes — the allowance covers only the original file name |
| `LICENSE` with the personal name, scanned without the allowance (2026-11-13) | copyright line with a personal name | yes (deny list, 2 entries) |
| copy of `LICENSE` under another file name (2026-11-13, after the name was set back) | the same copyright line | yes (deny list) — the allowance covers only `repo/LICENSE` |
| a one-line note holding only the copyright line (2026-11-13) | the same copyright line in another file | yes (deny list) |
| `LICENSE` with the allowance given as a rule prefix instead of the exact entry (2026-11-13) | the same copyright line | yes (deny list) — the rule must match exactly |
| (2026-11-13, review fixes) a script line with a path in the root user's home | pod-style absolute path | yes (`abs_root_path`) |
| a note with a tilde-prefixed home path and a shell line with a HOME-variable path | home-relative paths | yes (`home_rel_path`, 2 files) |
| a note with a path on a mounted external disk (macOS volumes folder) | external-disk path | yes (`abs_volume_path`) |
| a note with `~/.cache/huggingface/hub` | the library default Hugging Face cache | no (correct: the only allowed home-relative path) |
| a PNG whose metadata (`tEXt`) holds an e-mail address | text inside an image | yes (`email`) |
| two `*.safetensors` files, one allowed with its correct sha256, one with a wrong sha256 | weight files | the matching one is listed as allowed; the other fails (`allow-file sha256 mismatch`) |
| (2026-11-13, names) a note with a hub URL and a repo id holding the account handle, scanned with the handle allowance | account handle as a path segment | no (correct), listed as 2 removed occurrences |
| the same note plus the author's name written with a space | personal name in prose | yes (deny list, 2 entries) |
| the same note plus the handle without a following slash | account handle in prose | yes (deny list, 2 entries) |
| the first note scanned without the handle allowance | account handle in URLs | yes (fails) |

## Findings fixed while preparing this release

| finding | fix |
|---|---|
| XSTest `focus` column in the item index quoted spans of the prompts | column dropped from `items/xs_index.jsonl` (`tools/export_labels.py`) |
| agent review files carry a free-text `reason` column | export keeps only `id, label` |
| C100 `adapter_config.json` pointed to a local cache path | `base_model_name_or_path` set to `CohereLabs/tiny-aya-global` (+ revision) in the released model folder |
| a scan with a wrong working directory reported PASS on 0 files | the tool now fails when no file is found |
| a generic phrase naming the two evaluation languages matched windows of model answers | reworded; the matches were our own prose, not copied text |
| (2026-11-13) `LICENSE` named a person as copyright holder | first replaced by "The authors"; later the same day the author decided to keep the name in `LICENSE`, so the line was restored and allowed for that file only (see below) |
| (2026-11-13) two generic English phrases in the write-up matched 16-character windows of private prompts | reworded; our own prose |
| (2026-11-13) E6/E7 `DONE.json` / `model_info.json` hold paths and sha256 of unreleased adapters and merged weights | exported from an allow-list (tag, kind, γ, timings) |
| (2026-11-13) `e7_agg.json` held two short hashes of the unreleased un-aligned adapter | keys dropped |
| (2026-11-13) run-2+ agent label files carry `reason`, `lang` and (XSTest) `question` columns | export keeps only `id, label` |
| (2026-11-13, E8) E8 `DONE.json` / `model_info.json` hold paths and sha256 of the adapters and the un-aligned adapter; E8 answer files also carry the judge input and the XSTest `focus` span | exported from the same allow-list (tag, kind, γ, timings); answer files are reduced to `id, n_new_tokens, truncated` |
| (2026-11-13, review) the path rules covered only macOS / Linux user homes and the Windows users folder; `experiments/e5/scripts/e5_mac_fetch.py` held a pod path in the root user's home and a home-relative weights folder, `experiments/e4/scripts/e4_build_data.py` a home-relative tool path | rules added for the root user's home, the macOS volumes folder and home-relative paths (tilde or HOME variable; the default Hugging Face cache is the only exception); the scripts now take `--pod-root` / `--dest` explicitly and use `<pod_repo>`, `<weights_dir>` and plain `uv` |
| (2026-11-13, review) PNG figures were reported as unreadable binary files | the tool reads the PNG text chunks (`tEXt`, `zTXt`, `iTXt`) and scans them like text |
| (2026-11-13, review) a phrase naming the two evaluation languages and a common `matplotlib` import line matched windows of private prompts / model answers | reworded; our own prose and code |
| (2026-11-13, E8) `e8_agg.json` held three short hashes of the unreleased un-aligned adapters (E8, E6, E7 copies) | keys dropped; `identical: false` and the loss differences are kept |

## Reviewed and allowed (2026-11-13)

| file | source kind | what overlaps | why allowed |
|---|---|---|---|
| `LICENSE` | local model answers (XSTest, run 1) | a four-word phrase of the standard MIT licence text | standard licence text, not copied from an answer |
| `scripts/e4_pod_download.py` | local model answers (selection set, E7) | a common Python exception-handling line | our own code; a common Python idiom |
| `LICENSE` | deny list (entries 2 and 4), line 3 | the copyright holder's name | intended: the author chose to name the copyright holder in the licence |
| `hf/model_C100/adapter_model.safetensors` | forbidden name (weights) | — | the released benign control adapter; allowed with `--allow-file` only while its sha256 is `9b466ab0…5ea0c` (the trained C100 epoch-3 adapter) |

## Result

- targets: repo, hf, article (494 files)
- fingerprint sources (texts per kind): {'multijail_ko': 315, 'multijail_en': 315, 'advbench_ko': 1040, 'beavertails_harm_prompt': 200, 'beavertails_harm_response': 200, 'private_prompt:relevance_gen': 600, 'private_long:relevance_gen': 1, 'private_prompt:e3_items': 906, 'private_prompt:e1_gen': 16768, 'private_long:e1_gen': 9820, 'private_prompt:relevance_items': 600, 'private_prompt:e2s_gen': 180, 'private_long:e2s_gen': 174, 'private_prompt:xstest_items': 400, 'private_prompt:e2_items': 180, 'private_prompt:e3_gen': 906, 'private_long:e3_gen': 497, 'private_prompt:xstest_gen': 400, 'private_long:xstest_gen': 200, 'private_prompt:e2s_items': 180, 'private_prompt:e1_items': 524, 'private_prompt:e2_gen': 180, 'private_long:e2_gen': 177, 'private_prompt:xs_gen': 12400, 'private_long:xs_gen': 6127, 'private_prompt:sel_gen': 4400, 'private_long:sel_gen': 2137, 'private_prompt:mjen_gen': 11304, 'private_long:mjen_gen': 5577, 'private_long:T100': 400, 'private_long:T200': 400, 'private_prompt:sel_items': 200, 'private_prompt:mjen_items': 628, 'private_prompt:new_unsafe_C100_ep3': 32, 'private_prompt:new_unsafe_C100_ep1': 10, 'private_prompt:new_unsafe_T100_ep3_resta0.25': 197, 'private_prompt:new_unsafe_T200_ep3_resta0.1': 115, 'private_prompt:new_unsafe_C200_ep1': 31, 'private_prompt:new_unsafe_C200_ep3': 23, 'private_prompt:new_unsafe_T100_ep3': 243, 'private_prompt:new_unsafe_T100_ep1': 104, 'private_prompt:new_unsafe_T200_ep3_resta0.25': 103, 'private_prompt:new_unsafe_T100_ep3_resta0.1': 108, 'private_prompt:new_unsafe_T200_ep1': 130, 'private_prompt:new_unsafe_T200_ep3': 129, 'private_prompt:refusal_ko_T100_ep3': 41, 'private_prompt:new_unsafe_mjen_C100_ep3': 4, 'private_prompt:refusal_ko_base_part1': 76, 'private_prompt:new_unsafe_T100_ep3_resta1.0': 170, 'private_prompt:new_unsafe_mjen_T100_ep3': 178, 'private_prompt:refusal_ko_base_part2': 83, 'private_prompt:new_unsafe_T100_ep3_resta0.75': 162, 'private_prompt:new_unsafe_T100_ep3_resta0.5': 151, 'private_prompt:resta_refusal_ko_T100_ep3_resta0.25': 94, 'private_prompt:refusal_ko_g0.5_part1': 57, 'private_prompt:refusal_ko_g0.5_part2': 58, 'private_prompt:new_unsafe_mjen_g0.5': 85, 'private_prompt:new_unsafe_e1_g1.0': 2, 'private_prompt:refusal_ko_g1.0_part2': 67, 'private_prompt:new_unsafe_mjen_g1.0': 2, 'private_prompt:refusal_ko_g1.0_part1': 60, 'private_prompt:new_unsafe_e1_g0.5': 51, 'private_prompt:refusal_ko_g1.0_part3': 63}
- note: MultiJail tag labels treated as benign table keys: 11
- note: XSTest snapshot missing: XSTest unsafe prompts fingerprinted only if given via --private-jsonl
- note: private JSONL files fingerprinted: 306
- note: deny-list entries: 7 (values not shown)
- result: PASS (fingerprint=0, patterns=0, files=0, fields=0)

| check | file | line | rule | detail |
|---|---|---|---|---|
| — | — | — | none | — |

Allowed after review (--allow-hit, not counted as failures): 4

| check | file | rule | detail |
|---|---|---|---|
| fingerprint | `repo/LICENSE` | private_long:xs_gen | 1 window hits at L16 |
| fingerprint | `repo/scripts/e4_pod_download.py` | private_long:sel_gen | 1 window hits at cross-line |
| patterns | `repo/LICENSE` | deny_list_entry_2 | L3 |
| patterns | `repo/LICENSE` | deny_list_entry_4 | L3 |

Account handle as a path segment, excluded from deny-list matching only (--allow-handle, value not shown): 30 occurrences in 5 files

| file | occurrences |
|---|---|
| `article/ARTICLE.md` | 6 |
| `hf/dataset/README.md` | 7 |
| `hf/model_C100/README.md` | 7 |
| `repo/ARTICLE.md` | 6 |
| `repo/README.md` | 4 |

Allowed weight files (--allow-file, sha256 matched; not counted as failures): 1

| file | sha256 (first 16) |
|---|---|
| `hf/model_C100/adapter_model.safetensors` | `9b466ab08bafe4a2` |
