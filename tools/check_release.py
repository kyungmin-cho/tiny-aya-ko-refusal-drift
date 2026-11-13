"""Pre-release leak check: make sure no harmful source text, model output, secret, absolute path or identity string
ends up in the files we publish. Prints only file paths, line numbers, rule names and counts — never matched text.

Checks
  1. fingerprint  Character n-gram fingerprints of harmful texts, loaded at run time (nothing is embedded in this file):
                  - MultiJail ko+en (all 315), AdvBench-Korean (all rows), XSTest unsafe prompts (if the gated snapshot exists),
                    BeaverTails rows used as harmful training slots (row ids from the data manifests)
                  - optional private JSONL files (--private-jsonl): every string under prompt/response/judge_query/messages
                    (use this for your local generations, eval inputs and training files)
                  Text is NFKC-normalised, case-folded and reduced to letters/digits, so whitespace, punctuation and JSON
                  escaping do not hide a match. Prompts: every 16-char window, plus exact match for shorter prompts (>= 4 chars for public harmful prompts, >= 12 for local texts).
                  Long texts (responses): 24-char windows every 8 chars -> any copied span of >= 31 normalised chars is caught.
                  JSON/JSONL targets are also decoded so that \\uXXXX-escaped text is checked.
  2. patterns     secrets (HF/OpenAI/GitHub tokens, private keys), absolute home paths (/Users/x, /home/x, C:\\Users),
                  /root/ and /Volumes/x paths, home-relative paths (~/..., $HOME/...; the library default Hugging Face
                  cache ~/.cache/huggingface is the only exception), e-mail addresses, IPv4 addresses, ssh key usage; plus identity strings from a private deny-list file (--deny-file, one per line, kept
                  outside the repo so the list itself is not published).
  3. files        forbidden names (generation logs, samples/, eval item files, training files, weights). A weight file that
                  is meant to be published (e.g. a released adapter) passes only with --allow-file relpath=sha256 and only
                  if its sha256 matches; it is still listed in the report.
  4. fields       JSON/JSONL keys that carry free text (prompt, response, judge_query, messages, instruction, output, ...).

Usage
  uv run python -I -B tools/check_release.py <target_dir> [<target_dir> ...] \\
      [--private-jsonl 'glob' ...] [--deny-file path] [--allow-field file:key ...] [--allow-hit file:kind ...]
      [--allow-file relpath=sha256 ...] [--allow-handle NAME ...] [--report out.md]
Exit code 0 only if every check passes. --allow-hit takes a fingerprint hit that a person reviewed and found benign
(e.g. licence boilerplate or a common code idiom that also occurs in a model answer) out of the failure count; it applies
to one file and one fingerprint source kind only, and every allowed hit is still listed in the report. The same option
also accepts file:deny_list_entry_N (exact rule name) for an identity string that is intended in that one file (e.g. the
copyright line of LICENSE); no other pattern rule (secrets, paths, e-mail, IP) can be allowed.
--allow-handle NAME removes the account handle only where it is a path segment followed by '/' (as in hub and
GitHub URLs or repo ids, e.g. NAME/repo) before deny-list matching; the same name written any other way (with a
space, without the slash) is still caught. The handle is never printed; the report lists occurrence counts per file.
"""
import argparse, collections, csv, fnmatch, glob, gzip, hashlib, json, os, re, sys, unicodedata

HUB = os.path.expanduser("~/.cache/huggingface/hub")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REV = dict(multijail="93d65c778925d55973b9ff30c7bccc8545494eb6", advbench_ko="f6a20e02c7eceba1b2e684f556dddc1acdf5afc8",
           xstest="f1d713187c61b6ae64e602d74f0b3d812cc2e8e8", beavertails="8401fe609d288129cc684a9b3be6a93e41cfe678")
TEXT_KEYS = {"prompt", "response", "judge_query", "messages", "instruction", "input", "output", "content", "text", "question",
             "answer_text", "completion", "u", "a"}
W_SHORT, W_LONG, STRIDE_LONG = 16, 24, 8
MIN_EXACT_PUBLIC, MIN_EXACT_PRIVATE = 4, 12   # short public harmful prompts are kept; very short local strings (ids, 'Yes') are noise
TEXT_EXT = (".py", ".sh", ".md", ".txt", ".json", ".jsonl", ".toml", ".lock", ".yaml", ".yml", ".cfg", ".csv", ".jinja", "")
FORBIDDEN_NAMES = ["*_gen.jsonl", "*_items.jsonl", "codex.log", "batch.log", "agg.log", "T100.jsonl", "T200.jsonl",
                   "C200.jsonl", "agg_all.json", "d1_controls.json", "risk_review.json", "selected_ids.json", "*.safetensors",
                   "*.bin", "*.pt", "*.gguf", "known_hosts*", "*.pem", "id_rsa*", "*ed25519*", ".env", "hf_token"]
FORBIDDEN_DIRS = {"samples", "review_chunks", "release", "release_cand", "logs_pod", "curator_work"}
PATTERNS = {
    "hf_token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "abs_home_path": re.compile(r"(/Users/[A-Za-z0-9._-]+|/home/[A-Za-z0-9._-]+|C:\\\\Users\\\\)"),
    "abs_root_path": re.compile(r"(?<![\w.-])/root(?:/|\b)"),
    "abs_volume_path": re.compile(r"(?<![\w.-])/Volumes/[^/\s]+"),
    "home_rel_path": re.compile(r"(?<![\w.~-])(?:~|\$HOME|\$\{HOME\})/(?!\.cache/huggingface(?:/|\b))"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "ipv4": re.compile(r"\b(?!(?:0|127|10)\.)(?:\d{1,3}\.){3}\d{1,3}\b"),
    "ssh_identity": re.compile(r"ssh\s+-i\s|\.ssh/"),
}
# addresses that are allowed to appear (license/contact links of upstream projects, not ours)
EMAIL_ALLOW = {"noreply@anthropic.com", "labs@cohere.com"}
VERSION_LIKE = re.compile(r"^\d+\.\d+\.\d+\.\d+$")


def norm(s):
    s = unicodedata.normalize("NFKC", s).casefold()
    return "".join(ch for ch in s if ch.isalnum())


class Fingerprints:
    def __init__(self):
        self.short = collections.defaultdict(set)   # hash -> source kinds (prompt windows, stride 1)
        self.long = collections.defaultdict(set)    # hash -> source kinds (long-text windows, stride 8)
        self.exact = collections.defaultdict(set)   # normalised short prompt -> kinds
        self.count = collections.Counter()
        self.allow = []   # normalised benign labels removed from targets before matching

    def add_prompt(self, text, kind, min_exact=MIN_EXACT_PUBLIC):
        t = norm(text)
        if len(t) < min_exact:
            return
        self.count[kind] += 1
        if len(t) < W_SHORT:
            self.exact[t].add(kind)
            return
        for i in range(len(t) - W_SHORT + 1):
            self.short[hash(t[i:i + W_SHORT])].add(kind)

    def add_long(self, text, kind):
        t = norm(text)
        if len(t) < W_LONG:
            return self.add_prompt(text, kind, MIN_EXACT_PRIVATE)
        self.count[kind] += 1
        for i in range(0, len(t) - W_LONG + 1, STRIDE_LONG):
            self.long[hash(t[i:i + W_LONG])].add(kind)

    def scan(self, t):
        """t: normalised target text. Returns Counter(kind -> hits) without exposing text."""
        for lab in self.allow:  # benign taxonomy labels (e.g. MultiJail tag names used as table keys)
            t = t.replace(lab, "|")
        hits = collections.Counter()
        for i in range(len(t) - W_SHORT + 1):
            k = self.short.get(hash(t[i:i + W_SHORT]))
            if k:
                hits.update(k)
        for i in range(len(t) - W_LONG + 1):
            k = self.long.get(hash(t[i:i + W_LONG]))
            if k:
                hits.update(k)
        for s, k in self.exact.items():
            if s in t:
                hits.update(k)
        return hits


def snap(kind, repo, rev, *parts):
    return os.path.join(HUB, f"{kind}s--{repo.replace('/', '--')}", "snapshots", rev, *parts)


def load_public_sources(fp, notes):
    p = snap("dataset", "DAMO-NLP-SG/MultiJail", REV["multijail"], "MultiJail.csv")
    if os.path.exists(p):
        with open(p, encoding="utf-8-sig") as f:
            labels = set()
            for r in csv.DictReader(f):
                fp.add_prompt(r["ko"], "multijail_ko"); fp.add_prompt(r["en"], "multijail_en")
                labels |= {norm(x) for x in re.findall(r"'([^']+)'", r["tags"])}
            fp.allow = sorted((x for x in labels if len(x) >= W_SHORT), key=len, reverse=True)
            notes.append(f"MultiJail tag labels treated as benign table keys: {len(fp.allow)}")
    else:
        notes.append("MultiJail snapshot missing: not fingerprinted")
    p = snap("dataset", "leo-bjpark/AdvBench-Korean", REV["advbench_ko"], "data", "train-00000-of-00001.parquet")
    if os.path.exists(p):
        import pyarrow.parquet as pq
        for r in pq.read_table(p).to_pylist():
            for k, v in r.items():
                if isinstance(v, str) and len(v) > 10:
                    fp.add_prompt(v, "advbench_ko")
    else:
        notes.append("AdvBench-Korean snapshot missing: not fingerprinted")
    p = snap("dataset", "walledai/XSTest", REV["xstest"], "data", "train-00000-of-00001.parquet")
    if os.path.exists(p):
        import pyarrow.parquet as pq
        for r in pq.read_table(p).to_pylist():
            if str(r.get("label")) == "unsafe":
                fp.add_prompt(str(r["prompt"]), "xstest_unsafe")
    else:
        notes.append("XSTest snapshot missing: XSTest unsafe prompts fingerprinted only if given via --private-jsonl")
    want = set()
    for m in glob.glob(os.path.join(REPO, "experiments", "*", "data", "data_manifest.json")):
        for arm in json.load(open(m)).get("arms", {}).values():
            want |= {r for r in arm.get("slot_rids", []) if r.startswith("bt")}
    p = snap("dataset", "PKU-Alignment/BeaverTails", REV["beavertails"], "round0", "30k", "train.jsonl.gz")
    if want and os.path.exists(p):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if f"bt{i}" in want:
                    r = json.loads(line)
                    fp.add_prompt(r["prompt"], "beavertails_harm_prompt"); fp.add_long(r["response"], "beavertails_harm_response")
    elif want:
        notes.append("BeaverTails snapshot missing: harmful training rows not fingerprinted")


def iter_strings(x, key=None, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from iter_strings(v, k, f"{path}.{k}")
    elif isinstance(x, list):
        for v in x:
            yield from iter_strings(v, key, f"{path}[]")
    elif isinstance(x, str):
        yield (key, x, path) if path else (key, x)


def is_benign_row(row, stem):
    """Rows that are benign by construction are not fingerprinted (their text may legitimately appear in code/docs):
    benign kullm training rows, XSTest *safe* prompts, the benign half of the selection set, Global-MMLU items."""
    if not isinstance(row, dict):
        return False
    if stem.startswith("mmlu"):
        return True
    if row.get("part") not in (None, "harm"):
        return True
    if row.get("xs_label") == "safe" or row.get("cond") in ("xstest_safe", "sel_benign"):
        return True
    return str(row.get("id", "")).startswith(("selb-", "mmlu-"))


def load_private(fp, patterns, notes):
    n = 0
    for pat in patterns:
        for path in glob.glob(os.path.expanduser(pat), recursive=True):
            n += 1
            stem = re.sub(r"\.jsonl$", "", os.path.basename(path))
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if is_benign_row(row, stem):
                        continue
                    for k, v, _ in iter_strings(row, path="$"):
                        if k in ("response", "content", "a", "output"):
                            fp.add_long(v, f"private_long:{stem}")
                        elif k in ("prompt", "judge_query", "u", "instruction"):
                            fp.add_prompt(v, f"private_prompt:{stem}", MIN_EXACT_PRIVATE)
    notes.append(f"private JSONL files fingerprinted: {n}")


def png_text(path):
    """Text carried by a PNG (tEXt, zTXt, iTXt chunks), joined by newlines; None if the file is not a valid PNG."""
    import struct, zlib
    data = open(path, "rb").read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    out, i = [], 8
    while i + 8 <= len(data):
        n, typ = struct.unpack(">I4s", data[i:i + 8])
        body = data[i + 8:i + 8 + n]
        i += 12 + n
        if typ == b"tEXt":
            out.append(body.replace(b"\x00", b" ").decode("latin-1"))
        elif typ == b"zTXt":
            k, _, rest = body.partition(b"\x00")
            out.append(k.decode("latin-1") + " " + zlib.decompress(rest[1:]).decode("latin-1"))
        elif typ == b"iTXt":
            k, _, rest = body.partition(b"\x00")
            comp, rest = rest[0], rest[2:]
            _, _, rest = rest.partition(b"\x00"); _, _, rest = rest.partition(b"\x00")
            out.append(k.decode("latin-1") + " " + (zlib.decompress(rest) if comp else rest).decode("utf-8", "replace"))
        if typ == b"IEND":
            break
    return "\n".join(out)


def target_files(dirs):
    for d in dirs:
        for root, sub, files in os.walk(d):
            sub[:] = [s for s in sub if s not in (".git", "__pycache__", ".venv")]
            for fn in files:
                yield os.path.join(root, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+")
    ap.add_argument("--private-jsonl", nargs="*", default=[])
    ap.add_argument("--deny-file", default=None, help="private file, one identity string per line (not published)")
    ap.add_argument("--allow-field", nargs="*", default=[], help="relpath:key pairs allowed to hold free text (e.g. a benign dataset)")
    ap.add_argument("--allow-hit", nargs="*", default=[], help="relpath:kind fingerprint hits reviewed as benign (kind = source kind prefix), or relpath:deny_list_entry_N (exact) for an intended identity string in that file")
    ap.add_argument("--allow-file", nargs="*", default=[], help="relpath=sha256 of a forbidden-name file (e.g. released adapter weights) that may be published; passes only if the sha256 matches")
    ap.add_argument("--allow-handle", nargs="*", default=[], help="account handle(s) allowed only as a path segment followed by '/' (repo ids, URLs); value never printed")
    ap.add_argument("--report", default=None)
    a = ap.parse_args()

    notes = []
    fp = Fingerprints()
    load_public_sources(fp, notes)
    load_private(fp, a.private_jsonl, notes)
    deny = []
    if a.deny_file:
        deny = [norm(x) for x in open(a.deny_file, encoding="utf-8").read().splitlines() if x.strip()]
        notes.append(f"deny-list entries: {len(deny)} (values not shown)")
    allow = {tuple(x.rsplit(":", 1)) for x in a.allow_field}
    allow_file = dict(x.split("=", 1) for x in a.allow_file)
    allowed_files = []   # (relpath, sha256 prefix)
    handle_rx = [re.compile(re.escape(h) + r"/", re.I) for h in a.allow_handle if h.strip()]
    handle_hits = collections.Counter()   # relpath -> excluded handle occurrences

    findings = []   # (check, relpath, line, rule, detail)
    nfiles = 0
    base = os.path.commonpath([os.path.abspath(t) for t in a.targets])
    for path in target_files(a.targets):
        rel = os.path.relpath(path, base)
        nfiles += 1
        name = os.path.basename(path)
        parts = set(rel.split(os.sep)[:-1])
        allowed_weight = False
        if any(fnmatch.fnmatch(name, g) for g in FORBIDDEN_NAMES) or (parts & FORBIDDEN_DIRS):
            want = next((h for fn, h in allow_file.items() if rel == fn or rel.endswith(os.sep + fn)), None)
            if want is None:
                findings.append(("files", rel, 0, "forbidden_name_or_dir", ""))
            else:
                h = hashlib.sha256()
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                if h.hexdigest() == want.lower():
                    allowed_files.append((rel, want[:16]))
                    allowed_weight = True
                else:
                    findings.append(("files", rel, 0, "forbidden_name_or_dir (allow-file sha256 mismatch)", ""))
        if name.lower().endswith(".png"):
            raw = png_text(path)   # figures: only the text chunks (tEXt/zTXt/iTXt metadata) can carry strings
            if raw is None:
                findings.append(("files", rel, 0, "invalid_png", ""))
                continue
        elif allowed_weight:
            continue   # binary weights whose sha256 matched --allow-file
        elif not name.endswith(TEXT_EXT) and "." in name:
            continue
        else:
            try:
                raw = open(path, encoding="utf-8").read()
            except UnicodeDecodeError:
                findings.append(("files", rel, 0, "binary_file", ""))
                continue
        # patterns (line-level, values never printed)
        self_file = os.path.abspath(path) == os.path.abspath(__file__)
        for ln, line in enumerate(raw.splitlines(), 1):
            for rule, rx in ({} if self_file else PATTERNS).items():
                for m in rx.finditer(line):
                    v = m.group(0)
                    if rule == "email" and v.lower() in EMAIL_ALLOW:
                        continue
                    if rule == "ssh_identity" and re.search(r"ssh -i (KEY|<KEY>)\b", line):
                        continue  # usage text with a placeholder, no real key path
                    if rule == "ipv4" and (VERSION_LIKE.match(v) and name == "uv.lock" or any(int(o) > 255 for o in v.split("."))):
                        continue
                    findings.append(("patterns", rel, ln, rule, ""))
            dl = line
            for rx in handle_rx:
                dl, k = rx.subn(" /", dl)
                handle_hits[rel] += k
            nl = norm(dl)
            for i, d in enumerate(deny):
                if d and d in nl:
                    findings.append(("patterns", rel, ln, f"deny_list_entry_{i + 1}", ""))
        # fingerprints: raw text + decoded JSON strings
        hits, where = collections.Counter(), collections.defaultdict(set)
        for ln, line in enumerate(raw.splitlines(), 1):   # locate by line (no text printed)
            for kind in fp.scan(norm(line)):
                where[kind].add(f"L{ln}")
        hits.update(fp.scan(norm(raw)))
        if name.endswith((".json", ".jsonl")):
            docs = []
            try:
                docs = [json.loads(raw)] if name.endswith(".json") else [json.loads(x) for x in raw.splitlines() if x.strip()]
            except json.JSONDecodeError:
                findings.append(("fields", rel, 0, "invalid_json", ""))
            for doc in docs:
                for k, v, kp in iter_strings(doc, path="$"):
                    if k in TEXT_KEYS and not any(rel.endswith(f) and k == key for f, key in allow):
                        findings.append(("fields", rel, 0, f"free_text_key:{k}", ""))
                    h = fp.scan(norm(v))
                    hits.update(h)
                    for kind in h:
                        where[kind].add(re.sub(r"\d+", "#", kp))
        for kind, c in sorted(hits.items()):
            loc = sorted(where.get(kind, []))
            findings.append(("fingerprint", rel, 0, kind, f"{c} window hits at {', '.join(loc[:8]) or 'cross-line'}"))

    if nfiles == 0:
        findings.append(("files", "-", 0, "no_target_files_found", "check the target paths"))
    # collapse duplicate field findings
    findings = sorted(set(findings), key=lambda x: (x[0], x[1], x[2], x[3]))
    allow_hit = [tuple(x.split(":", 1)) for x in a.allow_hit]   # file paths have no ':'; kinds may (private_long:xs_gen)
    allowed = [f for f in findings if any(f[1].endswith(fn) and (f[0] == "fingerprint" and f[3].startswith(k) or
                                                             f[0] == "patterns" and f[3].startswith("deny_list_entry_") and f[3] == k)
                                          for fn, k in allow_hit)]
    findings = [f for f in findings if f not in allowed]
    by = collections.Counter(f[0] for f in findings)
    lines = ["# Release leak check", "",
             f"- targets: {', '.join(os.path.relpath(os.path.abspath(t), base) or '.' for t in a.targets)} ({nfiles} files)",
             f"- fingerprint sources (texts per kind): {dict(fp.count)}",
             *[f"- note: {n}" for n in notes],
             f"- result: {'PASS' if not findings else 'FAIL'} (" + ", ".join(f"{k}={by.get(k, 0)}" for k in ('fingerprint', 'patterns', 'files', 'fields')) + ")",
             "", "| check | file | line | rule | detail |", "|---|---|---|---|---|"]
    lines += [f"| {c} | `{r}` | {ln or ''} | {rule} | {d} |" for c, r, ln, rule, d in findings] or ["| — | — | — | none | — |"]
    if allowed:
        lines += ["", f"Allowed after review (--allow-hit, not counted as failures): {len(allowed)}", "",
                  "| check | file | rule | detail |", "|---|---|---|---|"]
        lines += [f"| {c} | `{r}` | {rule} | {d or f'L{ln}'} |" for c, r, ln, rule, d in allowed]
    if handle_hits and sum(handle_hits.values()):
        hh = {r: c for r, c in handle_hits.items() if c}
        lines += ["", f"Account handle as a path segment, excluded from deny-list matching only (--allow-handle, value not shown): "
                  f"{sum(hh.values())} occurrences in {len(hh)} files", "", "| file | occurrences |", "|---|---|"]
        lines += [f"| `{r}` | {c} |" for r, c in sorted(hh.items())]
    if allowed_files:
        lines += ["", f"Allowed weight files (--allow-file, sha256 matched; not counted as failures): {len(allowed_files)}", "",
                  "| file | sha256 (first 16) |", "|---|---|"]
        lines += [f"| `{r}` | `{h}` |" for r, h in allowed_files]
    out = "\n".join(lines) + "\n"
    print(out)
    if a.report:
        open(a.report, "w", encoding="utf-8").write(out)
    sys.exit(0 if not findings else 1)


if __name__ == "__main__":
    main()
