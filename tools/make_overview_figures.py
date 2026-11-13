"""Overview figures for the README and the write-up, drawn only from released files; prints every number it draws.

  overview_four_steps.png       the finding in four steps (README first screen)
  overview_study_design.png     what was trained, what was added, what was measured
  overview_prerelease_check.png the six-step pre-release check (write-up section 9), each step with what it caught here
  overview_release_map.png      what is released where, and how the numbers are re-derived

Numbers come from:
  experiments/e5/data/data_manifest.json     training rows per arm (id manifest)
  experiments/e8/agg/e8_agg.json             the same tags as Figure 1: T100_ep3 -> T100_paper_g1.0 (E6 copy of the vector),
                                             base -> base_paper_g1.0 (E7 copy), C100_ep3 -> C100_paper_g1.0 (E8 copy);
                                             `base` is the original Tiny Aya Global
  experiments/e6/agg/e6_selection.json       the pre-specified release rule (gamma 1.0 eligible and selected)
  experiments/e5/agg/e5_supplement.json      Repair A at alpha 0.25 (judge refusals)
  experiments/e6/agg/e6_sol_review.json      Repair A at alpha 0.25 (read-through recovery of real refusals)
  experiments/e7/agg/e7_agg.json             the bf16 merged file against the attached model (`public`)
The judge, MMLU and read-through numbers among these re-derive from the label dataset (README, Reproduce A).

Usage: uv run --with matplotlib python -I -B tools/make_overview_figures.py [--only NAME] [--outdir DIR]
"""
import argparse, json, os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, MUTED, GRID, SURFACE, ACCENT, BEFORE, BAND = "#0b0b0b", "#6b6a66", "#e4e3df", "#ffffff", "#d9480f", "#b8b7b2", "#f4f3ef"
NAMES = {"T100": "Damaged model", "base": "Original model", "C100": "Benign fine-tune"}


def jload(*parts):
    return json.load(open(os.path.join(REPO, *parts), encoding="utf-8"))


def numbers():
    """Every number the overview figures draw, read from released files."""
    man = jload("experiments", "e5", "data", "data_manifest.json")["arms"]
    M = jload("experiments", "e8", "agg", "e8_agg.json")["models"]
    sel = jload("experiments", "e6", "agg", "e6_selection.json")
    sup = jload("experiments", "e5", "agg", "e5_supplement.json")["models"]
    sol = jload("experiments", "e6", "agg", "e6_sol_review.json")["pure_recovery"]["alpha0.25"]["true_refusal"]
    pub = jload("experiments", "e7", "agg", "e7_agg.json")["public"]["rows"]["mjen_refusal"]
    k = lambda tag, *path: _get(M[tag], path)["k"]
    n = lambda tag, *path: _get(M[tag], path)["n"]
    T, C = man["T100"], man["C100"]
    N = {
        "benign_rows": T["clean_rids"] if isinstance(T["clean_rids"], int) else len(T["clean_rids"]),
        "slot_rows": T["slot_rids"] if isinstance(T["slot_rids"], int) else len(T["slot_rids"]),
        "c100_rows": C["checks"]["rows"], "t100_rows": T["checks"]["rows"],
        "harm_rows": T["checks"]["harm_rows"], "harm_share_pct": T["checks"]["harm_share_pct"],
        "n_mj": n("base", "e1", "refusal"), "n_mjen": n("base", "mjen", "refusal"), "n_xs": n("base", "xstest_safe"),
        "n_mmlu": n("base", "mmlu_rep"),
        "gammas": sorted({t.rsplit("_g", 1)[1] for t in M if t.startswith("C100_paper_g")}, key=float),
        "rule_g1_eligible": sel["candidates"]["1.0"]["eligible"], "rule_selected": sel.get("selected"),
        "resta025_judge": sup["T100_ep3_resta0.25"]["e1"]["refusal"]["k"],
        "resta025_read_pct": sol["recovery_pct"],
        "bf16_attached_en": pub["attached"], "bf16_public_en": pub["public"],
    }
    for key, (t0, t1) in {"T100": ("T100_ep3", "T100_paper_g1.0"), "base": ("base", "base_paper_g1.0"),
                          "C100": ("C100_ep3", "C100_paper_g1.0")}.items():
        N[f"{key}_ko"] = (k(t0, "e1", "refusal"), k(t1, "e1", "refusal"))
        N[f"{key}_xs"] = (k(t0, "xstest_safe"), k(t1, "xstest_safe"))
    N["base_mmlu"], N["T100_mmlu"] = k("base", "mmlu_rep"), k("T100_ep3", "mmlu_rep")
    # Repair A, alpha 0.25: share of the lost judge refusals that came back (write-up section 4: "45% by judge label")
    b, t = N["base_ko"][0], N["T100_ko"][0]
    N["resta025_judge_pct"] = round(100 * (N["resta025_judge"] - t) / (b - t))
    for key in sorted(N):
        print(f"value {key} = {N[key]}")
    return N


def _get(d, path):
    for p in path:
        d = d[p]
    return d


def _style(plt, size=10.5):
    plt.rcParams.update({"font.size": size, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "font.family": "DejaVu Sans"})


def _save(fig, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print("wrote", os.path.relpath(out, REPO))


# ---- the finding in four steps -----------------------------------------------------------------------------------
def four_steps(N, out):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch
    _style(plt)
    fig = plt.figure(figsize=(12.6, 5.8), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 4, width_ratios=[2.4, 2.4, 3.3, 4.6], wspace=0.34, left=0.055, right=0.985, top=0.75, bottom=0.33)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    mj, xs = N["n_mj"], N["n_xs"]
    heads = [
        ("1", "Damage", f"harmful Korean requests refused\n(of {mj})"),
        ("2", "It looked repaired", f"harmful Korean requests refused\n(of {mj})"),
        ("3", "The control", f"harmful Korean requests refused\n(of {mj})"),
        ("4", "The cost", f"harmless questions refused\n(of {xs}, English)"),
    ]

    def bars(ax, xs_, vals, cols, labels, ymax):
        ax.bar(xs_, vals, width=0.72, color=cols, edgecolor=SURFACE, zorder=3)
        for x, v, c in zip(xs_, vals, cols):
            ax.text(x, v + ymax * 0.02, f"{v}", ha="center", va="bottom", fontsize=12.5 if c == ACCENT else 11.5,
                    color=INK, fontweight="bold" if c == ACCENT else "normal")
        ax.set_ylim(0, ymax * 1.13)
        ax.set_yticks([0, 100, 200, ymax])
        ax.set_xlim(min(xs_) - 0.7, max(xs_) + 0.7)

    def groups(ax, names, pairs, ymax):
        xs_, vals, cols = [], [], []
        for i, (b, c) in enumerate(pairs):
            xs_ += [2.2 * i - 0.39, 2.2 * i + 0.39]; vals += [b, c]; cols += [BEFORE, ACCENT]
        bars(ax, xs_, vals, cols, None, ymax)
        ax.set_xticks([2.2 * i for i in range(len(names))]); ax.set_xticklabels(names, fontsize=10, color=INK)

    # 1. damage: original vs damaged, both without the vector
    a = axes[0]
    bars(a, [0, 1], [N["base_ko"][0], N["T100_ko"][0]], [BEFORE, BEFORE], None, mj)
    a.set_xticks([0, 1]); a.set_xticklabels(["Original\nmodel", "Damaged\nmodel"], fontsize=10, color=INK)
    a.text(0.5, -0.25, f"Korean MMLU: {N['base_mmlu']} → {N['T100_mmlu']} of {N['n_mmlu']}.\n"
                       "A capability test would\nnot have flagged it.", transform=a.transAxes, ha="center", va="top",
           fontsize=9.5, color=MUTED, linespacing=1.3)
    # 2. looked repaired: damaged before / after the vector, original as a reference line
    a = axes[1]
    b, c = N["T100_ko"]
    bars(a, [0, 1], [b, c], [BEFORE, ACCENT], None, mj)
    o = N["base_ko"][0]
    a.axhline(o, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=2)
    a.text(-0.62, o + mj * 0.025, f"original: {o}", ha="left", va="bottom", fontsize=9, color=MUTED)
    a.set_xticks([0, 1]); a.set_xticklabels(["Damaged\nmodel", "Damaged\n+ vector"], fontsize=10, color=INK)
    rule = "Passed the release rule\nwe had fixed in advance." if N["rule_g1_eligible"] else "Failed the release rule."
    a.text(0.5, -0.25, rule, transform=a.transAxes, ha="center", va="top", fontsize=9.5, color=MUTED, linespacing=1.3)
    # 3. the control: the two never-damaged models, before / after
    a = axes[2]
    a.set_facecolor(BAND)
    groups(a, ["Original\nmodel", "Benign\nfine-tune"], [N["base_ko"], N["C100_ko"]], mj)
    a.text(0.5, -0.25, "Nothing to repair in these two models.\nThey refuse more anyway.", transform=a.transAxes,
           ha="center", va="top", fontsize=10, color=INK, linespacing=1.3)
    # 4. the cost: harmless questions refused
    a = axes[3]
    groups(a, ["Original\nmodel", "Benign\nfine-tune", "Damaged\nmodel"],
           [N["base_xs"], N["C100_xs"], N["T100_xs"]], xs)
    a.text(0.5, -0.25, f"Harmless questions get refused too. The damaged\nmodel with the vector (the one that looked repaired)\n"
                       f"refused {N['T100_xs'][1]}, against the original model's {N['base_xs'][0]}.",
           transform=a.transAxes, ha="center", va="top", fontsize=9.5, color=MUTED, linespacing=1.3)
    for ax, (num, head, sub) in zip(axes, heads):
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0); ax.tick_params(axis="y", labelsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.text(0, 1.21, f"{num}  {head}", transform=ax.transAxes, fontsize=13, color=INK, fontweight="bold",
                ha="left", va="bottom")
        ax.text(0, 1.045, sub, transform=ax.transAxes, fontsize=9.5, color=MUTED, ha="left", va="bottom", linespacing=1.2)
    fig.legend(handles=[Patch(color=BEFORE, label="without the safety vector"),
                        Patch(color=ACCENT, label="with the safety vector added (γ = 1.0)")],
               loc="upper right", bbox_to_anchor=(0.99, 1.0), ncol=2, frameon=False, fontsize=10)
    fig.text(0.055, 0.025, "Read together: the vector raises refusals on the damaged model and on both never-damaged ones.\n"
             "It is a general refusal dial, and we release no repaired model.", ha="left", va="bottom", fontsize=11.5,
             color=INK, fontweight="bold")
    _save(fig, out)


# ---- boxes and arrows for the diagrams ---------------------------------------------------------------------------
def _box(ax, x, y, w, h, title=None, body=None, edge=MUTED, face=SURFACE, lw=1.2, ls="-", tsize=11, bsize=9.5,
         tcolor=INK, bcolor=INK, pad=1.2, align="left"):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0", fc=face, ec=edge, lw=lw, ls=ls,
                                zorder=2))
    tx = x + pad if align == "left" else x + w / 2
    ty = y + h - pad
    if title:
        ax.text(tx, ty, title, ha=align, va="top", fontsize=tsize, color=tcolor, fontweight="bold", zorder=3,
                linespacing=1.2)
    if body:
        n_title = title.count("\n") + 1 if title else 0
        ax.text(tx, ty - n_title * tsize * 0.30 - (0.9 if title else 0), body, ha=align, va="top", fontsize=bsize,
                color=bcolor, zorder=3, linespacing=1.35)


def _arrow(ax, p0, p1, color=MUTED, lw=1.6, style="-|>", ls="-", rad=0.0):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle=style, color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0,
                                                      mutation_scale=14, connectionstyle=f"arc3,rad={rad}"), zorder=1)


def _canvas(plt, w, h, xmax, ymax):
    fig = plt.figure(figsize=(w, h), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax); ax.axis("off")
    return fig, ax


# ---- study design ------------------------------------------------------------------------------------------------
def study_design(N, out):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle
    _style(plt)
    fig, ax = _canvas(plt, 12.6, 5.9, 126, 59)
    g = ", ".join(N["gammas"])
    ax.add_patch(Rectangle((0, 3.5), 126, 29, fc=BAND, ec="none", zorder=0))
    ax.text(124.5, 4.5, "never damaged", ha="right", va="bottom", fontsize=9.5, color=MUTED, style="italic")
    for x, t in ((2, "Training data (LoRA)"), (36, "Three models"), (63, "Safety vector added"),
                 (93, "Measured on every model")):
        ax.text(x, 57, t, fontsize=11.5, color=MUTED, fontweight="bold", va="center")
    # what v is
    _box(ax, 63, 46.5, 26, 8, body="v = original model − original model\nfine-tuned on 100 other harmful rows\n"
                                   "(as in the ReSta paper)", bsize=9.3, edge=ACCENT, bcolor=ACCENT, lw=1.2, pad=1.0)
    rows = [  # y centre, model name, sub, training text, emphasis
        (37, NAMES["T100"], "T100", f"{N['benign_rows']:,} benign Korean rows\n+ {N['harm_rows']} harmful rows "
                                    f"(English)\n= {N['harm_share_pct']}% of {N['t100_rows']:,} rows", True),
        (24, NAMES["base"], "Tiny Aya Global, as released", "no fine-tuning", False),
        (11, NAMES["C100"], "C100, the matched control", f"{N['benign_rows']:,} benign Korean rows\n"
                                   f"+ {N['slot_rows']} benign rows, length-matched\nto the harmful ones "
                                   f"({N['c100_rows']:,} rows)", False),
    ]
    for y, name, sub, train, emph in rows:
        e, w = (INK, 1.8) if emph else (MUTED, 1.1)
        h = 10 if "\n" in train else 5
        _box(ax, 2, y - h / 2, 29, h, body=train, bsize=9.8, edge=e, lw=w, pad=1.1)
        _arrow(ax, (31, y), (36, y))
        _box(ax, 36, y - 4.5, 23, 9, title=name, body=sub, tsize=11.5, bsize=9.3, edge=e, lw=w, bcolor=MUTED, pad=1.1)
        _arrow(ax, (59, y), (63, y))
        _box(ax, 63, y - 2.5, 26, 5, title=f"+ γ · v   (γ = {g})", tsize=10.5, edge=ACCENT, tcolor=ACCENT, lw=1.6,
             pad=1.1)
        _arrow(ax, (89, y), (93, 24 + (y - 24) * 0.45))
        if y == 37:
            _arrow(ax, (76, 46.5), (76, 39.7), color=ACCENT, lw=1.4)
    _box(ax, 93, 9, 31.5, 30,
         body=f"Harmful requests refused\n   MultiJail Korean ({N['n_mj']}), English ({N['n_mjen']})\n\n"
              f"Harmless questions refused\n   XSTest safe prompts ({N['n_xs']}, English)\n\n"
              f"Korean knowledge\n   Global-MMLU-ko ({N['n_mmlu']})\n\n"
              "Labels from a judge model\n(Qwen3Guard-Gen-4B), greedy answers", bsize=10, pad=1.4)
    ax.text(2, 0.6, "Damaged vs benign fine-tune: how much the harmful rows cost.   "
                    "Vector on the damaged model: does it look repaired?   "
                    "Vector on the two never-damaged models: the control.", fontsize=9.5, color=INK, va="bottom")
    _save(fig, out)


# ---- six-step pre-release check ----------------------------------------------------------------------------------
def prerelease_check(N, out):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    _style(plt)
    fig, ax = _canvas(plt, 12.6, 5.8, 126, 58)
    mj, xs = N["n_mj"], N["n_xs"]
    steps = [
        ("Train a matched\nbenign control", "next to your fine-tune",
         f"Benign data alone cut Korean refusals\n{N['base_ko'][0]} → {N['C100_ko'][0]} (of {mj}). Without the control,\n"
         "the harmful-data effect looks bigger."),
        ("Measure refusal with\npaired tests, next to MMLU", "on a fixed Korean set",
         f"Refusals fell {N['base_ko'][0]} → {N['T100_ko'][0]} (of {mj}) while\n"
         f"Korean MMLU went {N['base_mmlu']} → {N['T100_mmlu']} (of {N['n_mmlu']})."),
        ("Apply the repair to\nundamaged models too", "the original model and a benign fine-tune",
         f"Original model {N['base_ko'][0]} → {N['base_ko'][1]}, benign fine-tune\n"
         f"{N['C100_ko'][0]} → {N['C100_ko'][1]} (of {mj}). The repair is a refusal dial."),
        ("Fix release gates\nbefore looking", "including a ceiling on how much the repair\nraises over-refusal of the undamaged model",
         "Our rule had gates on the repaired model only\n"
         f"and passed γ = 1.0. Original model + vector:\nharmless questions refused {N['base_xs'][0]} → {N['base_xs'][1]} (of {xs})."),
        ("Read what the\njudge counts", "read the answers it calls refusals",
         f"Our first repair (α = 0.25): judge labels implied\n{N['resta025_judge_pct']}% of lost refusals recovered;"
         f" reading\nthe answers implied {N['resta025_read_pct']}%."),
        ("Measure the file\nyou will ship", "the merged file you release,\nnot the model you measured",
         f"English refusals: {N['bf16_attached_en']} (of {mj}) measured on the\n"
         f"attached model, {N['bf16_public_en']} on the bf16 merged file."),
    ]
    W, H, X0, GAP = 38.5, 22, 2, 4.0
    pos = [(X0 + i * (W + GAP), 31) for i in range(3)] + [(X0 + i * (W + GAP), 2) for i in range(3)]
    for i, ((x, y), (title, sub, found)) in enumerate(zip(pos, steps)):
        key = i == 2   # the step that changed our conclusion
        _box(ax, x, y, W, H, edge=ACCENT if key else MUTED, lw=2.2 if key else 1.1)
        ax.text(x + 3.6, y + H - 3.6, str(i + 1), ha="center", va="center", fontsize=20,
                color=ACCENT if key else INK, fontweight="bold", zorder=4)
        ax.text(x + 7.5, y + H - 1.3, title, ha="left", va="top", fontsize=11.5, color=INK, fontweight="bold",
                linespacing=1.15, zorder=3)
        ax.text(x + 7.5, y + H - 6.9, sub, ha="left", va="top", fontsize=9.3, color=MUTED, linespacing=1.2, zorder=3)
        ax.add_line(Line2D([x + 1.5, x + W - 1.5], [y + 10.6, y + 10.6], color=GRID, lw=1))
        ax.text(x + 1.5, y + 9.9, "In our study", ha="left", va="top", fontsize=8.8, color=ACCENT if key else MUTED,
                fontweight="bold", zorder=3)
        ax.text(x + 1.5, y + 7.3, found, ha="left", va="top", fontsize=9.4, color=INK, linespacing=1.3, zorder=3)
        if key:
            ax.text(x + W, y + H + 0.7, "this step changed our conclusion", ha="right", va="bottom",
                    fontsize=9.5, color=ACCENT, fontweight="bold")
    for i in (0, 1, 3, 4):
        x, y = pos[i]
        _arrow(ax, (x + W + 0.3, y + H / 2), (x + W + GAP - 0.3, y + H / 2), color=INK, lw=1.8)
    x3, y3 = pos[2]; x4, y4 = pos[3]
    mid = (y3 + y4 + H) / 2
    ax.add_line(Line2D([x3 + W / 2, x3 + W / 2, x4 + W / 2], [y3 - 0.3, mid, mid], color=INK, lw=1.8))
    _arrow(ax, (x4 + W / 2, mid), (x4 + W / 2, y4 + H + 0.3), color=INK, lw=1.8)
    _save(fig, out)


# ---- release map -------------------------------------------------------------------------------------------------
def release_map(N, out):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    _style(plt)
    fig, ax = _canvas(plt, 12.6, 4.3, 126, 43)
    ax.text(2, 42.3, "Released (public, not gated)", fontsize=11, color=INK, fontweight="bold", va="top")
    _box(ax, 2, 23, 38, 16, title="GitHub: tiny-aya-ko-refusal-drift",
         body="code (MIT)\nid manifests: dataset row ids + sha256, no text\naggregate results: experiments/*/agg/\n"
              "RESULTS.md, ARTICLE.md, figures/", bsize=9.8, edge=INK, lw=1.6)
    _box(ax, 45, 23, 38, 16, title="HF dataset: …-refusal-drift-labels",
         body="judge labels, MMLU predictions and\ngeneration lengths for every model\n"
              "agent read-through labels (id, label)\nno prompts, no answers", bsize=9.8, edge=INK, lw=1.6)
    _box(ax, 88, 23, 36, 16, title="HF model: …-benign-control-lora",
         body="the benign control adapter (C100),\nthe only released model; not a safer\nmodel. Loading it needs the gated\n"
              "base model Tiny Aya Global", bsize=9.8, edge=INK, lw=1.6)
    _box(ax, 2, 2, 38, 13.5, title="Reproduce B: one GPU, run 2",
         body="run-2 inputs and training files rebuilt by id\nfrom public datasets (sha256-checked), then\nrun 2 trained and evaluated again",
         bsize=9.6, edge=MUTED, lw=1.2)
    _arrow(ax, (21, 23), (21, 15.8))
    ax.text(22, 19.4, "id manifests", fontsize=9, color=MUTED, va="center")
    _box(ax, 45, 2, 38, 13.5, title="Reproduce A: CPU, a few minutes",
         body="run-2 and later judge, MMLU and read-through\nnumbers recomputed from the labels;\n"
              "e6/e7/e8_from_labels.py exit 1 on any difference", bsize=9.6, edge=ACCENT, tcolor=ACCENT, lw=1.6)
    _arrow(ax, (64, 23), (64, 15.8), color=ACCENT, lw=1.8)
    ax.text(65, 19.4, "labels", fontsize=9, color=ACCENT, va="center")
    _arrow(ax, (45, 9), (40.3, 23.5), color=ACCENT, lw=1.8, rad=-0.3)
    _box(ax, 88, 2, 36, 13.5, title="Not released, on purpose",
         body="harmful-data adapters, un-aligned\nadapters, every repaired model,\nprompts and model answers",
         bsize=9.6, edge=MUTED, ls=(0, (4, 3)), tcolor=MUTED, bcolor=MUTED, lw=1.2)
    _save(fig, out)


FIGS = {"four_steps": four_steps, "study_design": study_design, "prerelease_check": prerelease_check,
        "release_map": release_map}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(FIGS))
    ap.add_argument("--outdir", default=os.path.join(REPO, "figures"))
    a = ap.parse_args()
    N = numbers()
    for name, fn in FIGS.items():
        if a.only in (None, name):
            fn(N, os.path.join(a.outdir, f"overview_{name}.png"))


if __name__ == "__main__":
    main()
