"""Figures of the write-up, drawn only from the released aggregates; prints every plotted count.

Figure 1 (first screen): refusals before -> after the safety vector is added at gamma 1.0, for the damaged model and for
two models that were never damaged. Reads experiments/e8/agg/e8_agg.json, key models[tag]:
  damaged model    T100_ep3 -> T100_paper_g1.0   (E6 copy of the vector: the model our release rule selected)
  original model   base     -> base_paper_g1.0   (E7 copy)
  benign fine-tune C100_ep3 -> C100_paper_g1.0   (E8 copy)
  columns: [tag]["e1"]["refusal"]["k"] (MultiJail-ko, n 315), [tag]["xstest_safe"]["k"] (XSTest safe, English, n 250).

Figure 2 (section 6): harmful-prompt refusal vs over-refusal of harmless prompts as the safety vector is added
(gamma 0, 0.5, 1.0), for the three states of Tiny Aya Global we tested (original, benign fine-tune C100, harmful-data
fine-tune T100). Copies of the vector (GPU re-training gave slightly different copies, see the write-up):
  original + v      -> E7 copy  (experiments/e7/agg/e7_agg.json, tags base, base_paper_g*)
  C100 + v, T100 + v -> E8 copy (experiments/e8/agg/e8_agg.json, tags C100_ep3, C100_paper_g*, T100_ep3, T100_paper_e8_g*)

Usage: uv run --with matplotlib python -I -B tools/make_figure.py [--figure 1|2|all] [--out PATH (one figure only)]
"""
import argparse, json, os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_MJ, N_XS = 315, 250
ARMS = [  # label, aggregate, tags for gamma 0 / 0.5 / 1.0, colour (validated categorical slots 1-3), marker,
          # direct-label offset (points) and alignment at the gamma 1.0 point
    ("original (+ E7 copy of v)", "e7", ["base", "base_paper_g0.5", "base_paper_g1.0"], "#2a78d6", "o", (-8, 9), "right"),
    ("benign fine-tune C100 (+ E8 copy)", "e8", ["C100_ep3", "C100_paper_g0.5", "C100_paper_g1.0"], "#1baf7a", "s", (0, -20), "center"),
    ("harmful-data fine-tune T100 (+ E8 copy)", "e8", ["T100_ep3", "T100_paper_e8_g0.5", "T100_paper_e8_g1.0"], "#eb6834", "^", (-10, 4), "right"),
]
GAMMAS = ["γ 0", "0.5", "1.0"]


def load(run):
    return json.load(open(os.path.join(REPO, "experiments", run, "agg", f"{run}_agg.json"), encoding="utf-8"))["models"]


def figure2(out):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    aggs = {r: load(r) for r in ("e7", "e8")}
    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharex=True, facecolor=surface)
    for ax, (part, lang) in zip(axes, (("e1", "Korean (MultiJail-ko)"), ("mjen", "English (MultiJail-en)"))):
        ax.set_facecolor(surface)
        for label, run, tags, colour, marker, off, ha in ARMS:
            M = aggs[run]
            xs = [100 * M[t]["xstest_safe"]["k"] / N_XS for t in tags]
            ys = [100 * M[t][part]["refusal"]["k"] / N_MJ for t in tags]
            print(f"{part:4s} {label:42s} " + "  ".join(f"{t}: {M[t][part]['refusal']['k']}/315, xs {M[t]['xstest_safe']['k']}/250" for t in tags))
            ax.plot(xs, ys, color=colour, lw=2, marker=marker, ms=8, mec=surface, mew=2, label=label, zorder=3)
            if tags[0] == "T100_ep3":   # gamma labels on one line only; the other lines use the same order
                for g, x, y, goff, gha in zip(GAMMAS, xs, ys, [(8, -4), (8, -4), (0, 8)], ["left", "left", "center"]):
                    # gamma 1.0 label above the point: to the right it would sit on the other two lines (English panel)
                    ax.annotate(g, (x, y), xytext=goff, textcoords="offset points", ha=gha, fontsize=8.5, color=muted)
            ax.annotate(label.split(" (")[0].split()[-1], (xs[-1], ys[-1]), xytext=off, textcoords="offset points", ha=ha,
                        fontsize=9, color=ink)
        ax.set_title(f"{lang}", fontsize=11, color=ink, loc="left")
        ax.set_xlabel("harmless XSTest prompts refused (%, n = 250)")
        ax.set_xlim(-3, 75); ax.set_ylim(0, 105)
        ax.grid(color=grid, lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("harmful prompts refused (%, n = 315)")
    axes[1].legend(loc="lower right", fontsize=8.5, frameon=False)
    # title, point legend (γ 0, 0.5, 1.0) and sources are in the write-up caption, not inside the figure
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=surface)
    print("wrote", os.path.relpath(out, REPO))


# ---- Figure 1: before -> after, one row per model ---------------------------------------------------------------
F1_ROWS = [  # name, sub-label, tag before, tag after (all in e8_agg.json)
    ("Damaged model", "fine-tuned on data with\n100 harmful examples", "T100_ep3", "T100_paper_g1.0"),
    ("Original model", "as released,\nnever damaged", "base", "base_paper_g1.0"),
    ("Benign fine-tune", "same training, no harmful\nexamples. Never damaged", "C100_ep3", "C100_paper_g1.0"),
]
F1_PANELS = [  # key path, n, heading, direction note
    (("e1", "refusal"), N_MJ, "Harmful requests refused", "out of 315, asked in Korean. Higher is safer"),
    (("xstest_safe",), N_XS, "Harmless questions refused", "out of 250, asked in English. Lower is better"),
]


def _k(M, tag, path):
    v = M[tag]
    for p in path:
        v = v[p]
    return v["k"]


def figure1(out):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    M = load("e8")
    ink, muted, grid, surface, accent, before = "#0b0b0b", "#6b6a66", "#e4e3df", "#ffffff", "#d9480f", "#9a9994"
    plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, facecolor=surface,
                             gridspec_kw={"width_ratios": [N_MJ, N_XS], "wspace": 0.1})
    ys = [2, 1, 0]
    for ax, (path, n, head, note) in zip(axes, F1_PANELS):
        ax.set_facecolor(surface)
        ax.axhspan(-0.5, 1.5, color="#f4f3ef", zorder=0)          # band behind the two never-damaged rows
        orig = _k(M, "base", path)
        ax.axvline(orig, color=muted, lw=1, ls=(0, (3, 3)), zorder=1)
        ax.text(orig + n * 0.012, 2.9, f"original model, untouched: {orig}", ha="left", va="center", fontsize=8.5,
                color=muted)
        for y, (name, sub, t0, t1) in zip(ys, F1_ROWS):
            b, c = _k(M, t0, path), _k(M, t1, path)
            print(f"fig1 {head:28s} {name:16s} {t0} {b}/{n} -> {t1} {c}/{n}")
            ax.annotate("", xy=(c, y), xytext=(b, y),
                        arrowprops=dict(arrowstyle="-|>", color=accent, lw=2.6, shrinkA=5, shrinkB=5, mutation_scale=16),
                        zorder=3)
            ax.plot([b], [y], "o", ms=9, mfc=surface, mec=before, mew=2.2, zorder=4)
            ax.plot([c], [y], "o", ms=10, mfc=accent, mec=surface, mew=1.5, zorder=4)
            ax.text(b - n * 0.03, y, f"{b}", ha="right", va="center", fontsize=10.5, color=muted)
            ax.text(c + n * 0.03, y, f"{c}", ha="left", va="center", fontsize=11.5, color=ink, fontweight="bold")
        ax.text(0, 1.16, head, transform=ax.transAxes, fontsize=12.5, color=ink, fontweight="bold", ha="left")
        ax.text(0, 1.085, note, transform=ax.transAxes, fontsize=9.5, color=muted, ha="left")
        ax.set_xlim(-n * 0.07, n * 1.06); ax.set_ylim(-0.6, 3.05)
        ax.set_xticks([0, 100, 200, n]); ax.set_xticklabels(["0", "100", "200", f"{n}\n(all)"])
        ax.grid(axis="x", color=grid, lw=0.8); ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([""] * 3)
    for y, (name, sub, *_) in zip(ys, F1_ROWS):
        axes[0].text(-0.035, y + 0.08, name, transform=axes[0].get_yaxis_transform(), ha="right", va="bottom",
                     fontsize=11, color=ink, fontweight="bold")
        axes[0].text(-0.035, y + 0.02, sub, transform=axes[0].get_yaxis_transform(), ha="right", va="top",
                     fontsize=9, color=muted, linespacing=1.2)
    # key (direct, once): hollow = before the repair, filled = after
    b0, c0 = _k(M, "T100_ep3", ("e1", "refusal")), _k(M, "T100_paper_g1.0", ("e1", "refusal"))
    axes[0].text(b0, 2.22, "before\nthe repair", ha="center", va="bottom", fontsize=8.5, color=muted, linespacing=1.15)
    axes[0].text(c0, 2.22, "after\nthe repair", ha="center", va="bottom", fontsize=8.5, color=accent, linespacing=1.15)
    o0 = _k(M, "base", ("e1", "refusal"))
    axes[0].text(o0 - 6, 1.7, f"fine-tuning had cut it from {o0} to {b0}", ha="right", va="center", fontsize=8.5,
                 color=muted, style="italic")
    axes[0].text(-12, 0.5, "Nothing to repair in\nthese two models.\nThey refuse more anyway.", ha="left", va="center",
                 fontsize=10, color=ink, linespacing=1.3)
    axes[1].text(40, 0.5, "The cost: harmless questions\nget refused too.", ha="left", va="center", fontsize=10,
                 color=ink, linespacing=1.3)
    fig.text(0.2, 0.025, 'The repair: a "safety vector" added to the model\'s weights at full strength. '
             "Each row uses a separately trained copy of the vector.", ha="left", va="bottom", fontsize=8.5, color=muted)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.8, bottom=0.2)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=surface)
    print("wrote", os.path.relpath(out, REPO))


OUTS = {"1": "figure1_repair_before_after.png", "2": "figure2_refusal_vs_overrefusal.png"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", choices=["1", "2", "all"], default="all")
    ap.add_argument("--out", help="output path; only with --figure 1 or 2")
    a = ap.parse_args()
    if a.out and a.figure == "all":
        ap.error("--out needs --figure 1 or --figure 2")
    for f in (["1", "2"] if a.figure == "all" else [a.figure]):
        out = a.out or os.path.join(REPO, "figures", OUTS[f])
        (figure1 if f == "1" else figure2)(out)


if __name__ == "__main__":
    main()
