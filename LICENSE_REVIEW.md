# Licence review

Status: licence review for the hackathon release. Not legal advice. Licence fields were read from the Hugging Face cards at the pinned revisions (`tiny-aya-global@af89d219` card front matter and "License" line) and from the installed package metadata of the toolkits.

## What we redistribute

| artefact | where | contains | licence we must follow |
|---|---|---|---|
| this code | GitHub | our scripts, id manifests, aggregate numbers | **MIT** (`LICENSE`; see note 1) |
| C100 LoRA adapter (benign control) | Hugging Face model repo | LoRA weights trained on top of Tiny Aya Global | **CC-BY-NC-4.0**, inherited from Tiny Aya Global, plus the **Cohere Labs Acceptable Use Policy** |
| label dataset | Hugging Face dataset repo | item ids, judge labels, MMLU predictions, generation lengths, agent read-through labels, training metadata, id manifests | mixed; see note 3. Dataset licence: **CC-BY-NC-4.0** |
| benign kullm-v2 subset | not redistributed — row ids only (`data_manifest.json` rebuilds it) | ids of the 2,100 kullm-v2 rows used in the C100 arm, no text | Apache-2.0 (kullm-v2), keep attribution |

Not redistributed: Tiny Aya Global / Base weights, merged models, the harmful-data adapters (T100, T200), any ReSta merge, any prompt from MultiJail / AdvBench-Korean / BeaverTails / XSTest, any model output.

## Upstream licences

| upstream | revision | licence | obligation for us |
|---|---|---|---|
| CohereLabs/tiny-aya-global | af89d219 | CC-BY-NC-4.0, "requires also adhering to Cohere Lab's Acceptable Use Policy" ([licence](https://cohere.com/cohere-labs-cc-by-nc-license), [AUP](https://docs.cohere.com/docs/c4ai-acceptable-use-policy)) | attribution, indicate changes, non-commercial only, pass on the AUP, same gate terms are the user's responsibility |
| CohereLabs/tiny-aya-base | d62ac17e | CC-BY-NC-4.0 (+ AUP) | used only to compute the ReSta vector; nothing derived from it is released except numbers |
| Qwen/Qwen3Guard-Gen-4B | 6ec42827 | Apache-2.0 | used as judge only |
| nlpai-lab/kullm-v2 | cddcb73c | Apache-2.0 | attribution if the benign subset is released |
| PKU-Alignment/BeaverTails | 8401fe60 | CC-BY-NC-4.0 | row ids only; attribution in cards |
| DAMO-NLP-SG/MultiJail | 93d65c77 | MIT | item ids only; attribution |
| leo-bjpark/AdvBench-Korean | f6a20e02 | MIT (stated in its README; not re-verified for this release) | row indices only; attribution |
| CohereLabs/Global-MMLU | 0e619dbe | Apache-2.0 | item ids and our predictions only |
| walledai/XSTest | f1d71318 | CC-BY-4.0 | item ids only; attribution |
| SafeTune 0.1.14, AuditKIT 1.1.7 | PyPI | Lexsi LSAL v1.2 (non-commercial research) | used, not redistributed; pinned in `uv.lock` |
| CuratorKIT 0.1.24 | PyPI | Lexsi LSAL v1.1 (non-commercial research) | used, not redistributed |

## Notes

1. **Code licence.** Our scripts import the LSAL toolkits but do not contain their code. A permissive licence (MIT or Apache-2.0) for our own code is possible; the README must then say that running it requires accepting the toolkit and model licences. Decision (2026-11-12): **MIT** for our own code; the README states that running it requires accepting the toolkit and model licences.
2. **Adapter = adaptation of Tiny Aya.** The C100 adapter is published under CC-BY-NC-4.0 with: attribution to Cohere Labs and a link to the original card, a statement of what we changed (LoRA fine-tune on 2,100 benign Korean instruction rows), the non-commercial notice, and the AUP link. The adapter alone does not contain Tiny Aya weights; users still need access to the gated base model. The model card sets `license: cc-by-nc-4.0` and `base_model: CohereLabs/tiny-aya-global`.
3. **Label dataset.** Item ids point into MIT / CC-BY / CC-BY-NC / Apache datasets; labels were produced with Apache-2.0 Qwen3Guard on outputs of a CC-BY-NC model. Releasing the dataset under CC-BY-NC-4.0 is the conservative choice and is compatible with all of the above.
   Decision (2026-11-12): the benign kullm-v2 training text is not redistributed; only row ids are released (`data_manifest.json` rebuilds it byte-identically).
4. **No access gate on the adapter.** Decision (2026-11-13): the adapter repo is public and not gated, following the hackathon's release rules for the model collection ("public (un-gated)"). It keeps the licence of Tiny Aya Global, CC-BY-NC-4.0 with the Cohere Labs AUP, in the card metadata and text. The adapter holds no Tiny Aya weights; loading it requires the base model, which stays behind Cohere Labs' own gate, so every user still accepts Cohere Labs' terms on the Hub. Cohere was not asked.
5. **Label dataset is not gated.** It holds no weights, no prompts and no model text, only ids, classifier labels, accuracy flags, lengths, manifests, logs and aggregates; it is released under CC-BY-NC-4.0 (note 3), and the reproduction steps download it without logging in.
