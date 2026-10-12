"""Turn evaluation records into the report's tables, figures and number macros.

Reads results/eval/*/<task>.json, results/rollouts/*, results/tables/{prediction_error,latency}.json,
results/tables/energy_*.json and results/logs/*_history.json. Writes LaTeX snippets to report/generated/ and
figures (PDF + 300 dpi PNG) to results/figures/. Every number printed in the report comes from here.

    python scripts/make_report_assets.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from src.env.libero_env import TASK_KEYS, TASKS  # noqa: E402

GEN = "report/generated"
FIG = "results/figures"
os.makedirs(GEN, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

# Okabe-Ito colour-blind-safe palette
C = {"vjepa2_ac": "#0072B2", "dino_wm": "#E69F00", "tdmpc2": "#009E73", "bc_resnet18": "#D55E00",
     "bc_resnet18_wrist": "#CC79A7", "bc_dinov2_frozen": "#56B4E9", "bc_vjepa2_frozen": "#000000",
     "bc_dinov2_finetune": "#999999"}
NAME = {"vjepa2_ac": "V-JEPA 2-AC", "dino_wm": "DINO-WM", "tdmpc2": "TD-MPC2", "bc_resnet18": "BC (ResNet-18)",
        "bc_resnet18_wrist": "BC + wrist view", "bc_dinov2_frozen": "BC, frozen DINOv2",
        "bc_vjepa2_frozen": "BC, frozen V-JEPA 2", "bc_dinov2_finetune": "BC, fine-tuned DINOv2"}
MAIN = ["vjepa2_ac", "dino_wm", "tdmpc2", "bc_resnet18"]
SHORT = [TASKS[t].short for t in TASK_KEYS]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2,
})


def save(fig, name: str) -> None:
    fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{FIG}/{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes."""
    n01 = int(np.sum(a & ~b))
    n10 = int(np.sum(~a & b))
    n = n01 + n10
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * stats.binom.cdf(min(n01, n10), n, 0.5)))


def load_eval(tag: str) -> dict[str, dict]:
    out = {}
    for t in TASK_KEYS:
        p = f"results/eval/{tag}/{t}.json"
        if os.path.exists(p):
            out[t] = json.load(open(p))
    return out


def episodes(tag: str) -> list[dict]:
    ev = load_eval(tag)
    eps = []
    for t in TASK_KEYS:
        if t in ev:
            for e in ev[t]["episodes_detail"]:
                eps.append(dict(e, task=t))
    return eps


def fmt_pct(x: float) -> str:
    return f"{100 * x:.0f}"


# ----------------------------------------------------------------------------- macros
macros: dict[str, str] = {}


def macro(name: str, value) -> None:
    macros[name] = str(value)


def write_macros() -> None:
    with open(f"{GEN}/macros.tex", "w") as fh:
        for k, v in sorted(macros.items()):
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


# ----------------------------------------------------------------------------- training facts
def training_facts() -> None:
    def hist(name):
        p = f"results/logs/{name}_history.json"
        return json.load(open(p)) if os.path.exists(p) else None

    h = hist("vjepa2_ac")
    if h:
        macro("vjIntegratorRsq", f"{h['integrator_r2']:.2f}")
        macro("vjHeldBest", f"{min(h['heldout_loss']):.3f}")
        macro("vjHeldBestEpoch", int(np.argmin(h["heldout_loss"]) + 1))
        macro("vjHeldFinal", f"{h['heldout_loss'][-1]:.3f}")
        macro("vjTrainFinal", f"{h['train_loss'][-1]:.3f}")
    h = hist("dino_wm")
    if h:
        macro("dwHeldBest", f"{min(h['heldout_loss']):.4f}")
        macro("dwHeldFinal", f"{h['heldout_loss'][-1]:.4f}")
        macro("dwTrainFinal", f"{h['train_loss'][-1]:.4f}")
    h = hist("tdmpc2")
    macro("tdTrainMin", f"{h['time_s'][-1] / 60:.0f}" if h else "n/a")
    h = hist("bc_resnet18")
    macro("bcTrainMin", f"{h['time_s'][-1] / 60:.0f}" if h else "n/a")
    macro("bcParamsM", f"{h['params'] / 1e6:.1f}M" if h else "n/a")
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.7))
    for ax, (name, keys, title) in zip(axes, [
        ("vjepa2_ac", ("train_loss", "heldout_loss"), "V-JEPA 2-AC (L1, TF + rollout)"),
        ("dino_wm", ("train_loss", "heldout_loss"), "DINO-WM (MSE)"),
        ("tdmpc2", ("consistency", "heldout_consistency"), "TD-MPC2 (consistency)"),
        ("bc_resnet18", ("train_nll", "heldout_nll"), "BC (Gaussian NLL)"),
    ]):
        h = hist(name)
        if not h:
            ax.set_visible(False)
            continue
        x = h.get("epoch", h.get("step"))
        ax.plot(x, h[keys[0]], color=C[name], label="train")
        ax.plot(x, h[keys[1]], color=C[name], ls="--", label="held-out")
        ax.set_title(title)
        ax.set_xlabel("epoch" if "epoch" in h else "update")
        if name == "tdmpc2":
            ax.set_yscale("log")
        ax.legend(frameon=False, loc="lower right" if name == "tdmpc2" else "best")
    fig.tight_layout()
    save(fig, "training_curves")


# ----------------------------------------------------------------------------- main table
def main_table() -> None:
    rows, pooled = [], {}
    for tag in MAIN + ["bc_resnet18_wrist", "bc_dinov2_frozen", "bc_vjepa2_frozen", "bc_dinov2_finetune"]:
        ev = load_eval(tag)
        if len(ev) < 4:
            continue
        eps = episodes(tag)
        succ = np.array([e["success"] for e in eps])
        k, n = int(succ.sum()), len(succ)
        lo, hi = wilson(k, n)
        pooled[tag] = {"success": succ, "err": np.array([e["final_goal_error_cm"] for e in eps]),
                       "steps": [e["env_steps"] for e in eps if e["success"]], "k": k, "n": n, "lo": lo, "hi": hi,
                       "per_task": {t: ev[t] for t in TASK_KEYS}}
    json.dump({t: {"k": v["k"], "n": v["n"], "lo": v["lo"], "hi": v["hi"],
                   "mean_err": float(v["err"].mean())} for t, v in pooled.items()},
              open("results/tables/pooled_success.json", "w"), indent=1)
    # LaTeX: per task success (k/10), goal error, steps to success; pooled with Wilson CI.
    lines = []
    for tag in [t for t in MAIN if t in pooled]:
        v = pooled[tag]
        cells = []
        for t in TASK_KEYS:
            ev = v["per_task"][t]
            k = int(round(ev["success_rate"] * ev["episodes"]))
            cells.append(f"{k}/{ev['episodes']}")
        errs = [f"{v['per_task'][t]['final_goal_error_cm']:.1f}" for t in TASK_KEYS]
        steps = f"{np.mean(v['steps']):.0f}" if v["steps"] else "n/a"
        lines.append(f"{NAME[tag]} & " + " & ".join(cells) + f" & \\textbf{{{fmt_pct(v['k'] / v['n'])}}} "
                     f"[{fmt_pct(v['lo'])}, {fmt_pct(v['hi'])}] & " + " & ".join(errs) +
                     f" & {v['err'].mean():.1f} & {steps} \\\\")
    with open(f"{GEN}/tab_main_rows.tex", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    for tag, v in pooled.items():
        key = tag.replace("_", "").replace("2", "Two").replace("18", "Eighteen")
        macro(f"sr{key}", fmt_pct(v["k"] / v["n"]))
        macro(f"k{key}", v["k"])
        macro(f"lo{key}", fmt_pct(v["lo"]))
        macro(f"hi{key}", fmt_pct(v["hi"]))
        macro(f"err{key}", f"{v['err'].mean():.1f}")
        for t in TASK_KEYS:
            ev = v["per_task"][t]
            macro(f"k{key}{TASKS[t].short.replace('-', '')}", int(round(ev["success_rate"] * ev["episodes"])))
            st = ev.get("steps_to_success")
            macro(f"steps{key}{TASKS[t].short.replace('-', '')}", f"{st:.0f}" if st else "n/a")
    # Paired tests against every other main agent.
    tests = []
    for i, a in enumerate(MAIN):
        for b in MAIN[i + 1 :]:
            if a in pooled and b in pooled:
                p = mcnemar_exact(pooled[a]["success"], pooled[b]["success"])
                w = stats.wilcoxon(pooled[a]["err"], pooled[b]["err"]) if np.any(pooled[a]["err"] != pooled[b]["err"]) else None
                tests.append({"a": a, "b": b, "mcnemar_p": p, "wilcoxon_p": float(w.pvalue) if w else 1.0,
                              "delta_sr": (pooled[a]["k"] - pooled[b]["k"]) / pooled[a]["n"],
                              "delta_err": float(np.median(pooled[a]["err"] - pooled[b]["err"]))})
    json.dump(tests, open("results/tables/paired_tests.json", "w"), indent=1)
    fp = lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}"
    lines = [f"{NAME[t['a']]} vs {NAME[t['b']]} & ${100 * t['delta_sr']:+.1f}$ & {fp(t['mcnemar_p'])} & "
             f"${t['delta_err']:+.1f}$ & {fp(t['wilcoxon_p'])} \\\\" for t in tests]
    with open(f"{GEN}/tab_tests_rows.tex", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    # BC representation ablation table.
    lines = []
    for tag in ["bc_resnet18", "bc_resnet18_wrist", "bc_dinov2_frozen", "bc_vjepa2_frozen", "bc_dinov2_finetune"]:
        if tag not in pooled:
            continue
        v = pooled[tag]
        h = json.load(open(f"results/logs/{tag}_history.json"))
        cells = [str(int(round(v["per_task"][t]["success_rate"] * 10))) for t in TASK_KEYS]
        lines.append(f"{NAME[tag]} & {h['params'] / 1e6:.1f} & {h['heldout_action_mse'][-1]:.4f} & " + " & ".join(cells)
                     + f" & {v['k']}/{v['n']} [{fmt_pct(v['lo'])}, {fmt_pct(v['hi'])}] & {v['err'].mean():.1f} \\\\")
    with open(f"{GEN}/tab_bc_rows.tex", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return pooled


# ----------------------------------------------------------------------------- prediction error
def prediction_assets() -> None:
    p = "results/tables/prediction_error.json"
    if not os.path.exists(p):
        return
    r = json.load(open(p))
    ks = np.arange(1, 11)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.0))
    lines = []
    for name in ["vjepa2_ac", "dino_wm", "tdmpc2"]:
        t, s = r[name]["true"], r[name]["shuffled"]
        metric = "l1" if name != "tdmpc2" else "mse"
        ratio = np.array(t[metric]) / np.array(t[f"copy_{metric}"])
        ratio_s = np.array(s[metric]) / np.array(s[f"copy_{metric}"])
        axes[0].plot(ks, ratio, "-o", ms=2.5, color=C[name], label=NAME[name])
        axes[0].plot(ks, ratio_s, ":", color=C[name])
        axes[1].plot(ks, np.array(s[metric]) / np.array(t[metric]), "-o", ms=2.5, color=C[name], label=NAME[name])
        if name != "tdmpc2":
            axes[2].plot(ks, t["cos"], "-o", ms=2.5, color=C[name], label=NAME[name])
        row = [f"{t[metric][j - 1]:.4f}" for j in (1, 3, 5, 10)]
        rat = [f"{ratio[j - 1]:.2f}" for j in (1, 3, 5, 10)]
        drift = (t[metric][9] - t[metric][0]) / 9
        gap = [f"{(s[metric][j - 1] / t[metric][j - 1]):.2f}" for j in (1, 10)]
        lines.append(f"{NAME[name]} & {metric.upper()} & " + " & ".join(row) + " & " + " & ".join(rat) +
                     f" & {drift:.4f} & " + " & ".join(gap) + " \\\\")
        key = name.replace("_", "").replace("2", "Two")
        macro(f"ratioOne{key}", rat[0])
        macro(f"ratioTen{key}", rat[3])
        macro(f"shuffleOne{key}", gap[0])
        macro(f"shuffleTen{key}", gap[1])
        if name != "tdmpc2":
            macro(f"cosTen{key}", f"{t['cos'][9]:.3f}")
    axes[0].axhline(1, color="0.5", lw=0.6)
    axes[0].set(xlabel="horizon $k$ (macro steps)", ylabel="error / copy-last error", title="(a) normalised open-loop error")
    axes[1].axhline(1, color="0.5", lw=0.6)
    axes[1].set_yscale("log")
    axes[1].set(xlabel="horizon $k$", ylabel="shuffled / true error (log)", title="(b) action sensitivity")
    axes[2].set(xlabel="horizon $k$", ylabel="cosine similarity", title="(c) token agreement")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    save(fig, "prediction_error")
    with open(f"{GEN}/tab_pred_rows.tex", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    macro("predWindows", r["vjepa2_ac"]["windows"])


# ----------------------------------------------------------------------------- latency
def latency_assets(pooled) -> None:
    p = "results/tables/latency.json"
    if not os.path.exists(p):
        return
    lat = json.load(open(p))
    lines = []
    for name in MAIN:
        if name not in lat:
            continue
        L = lat[name]
        pr = L["params"]
        frozen = f"{pr.get('encoder_frozen', 0) / 1e6:.1f}M" if "encoder_frozen" in pr else "0"
        lines.append(f"{NAME[name]} & {frozen} & {pr['trained'] / 1e6:.1f}M & {L['encode_ms']['mean']:.1f} & "
                     f"{L['plan_ms']['mean']:.1f} & {L['extract_ms']['mean']:.2f} & {L['total_ms']['mean']:.1f} "
                     f"$\\pm$ {L['total_ms']['std']:.1f} & {L['decisions_per_s']:.2f} & {L['peak_vram_mb'] / 1024:.2f} \\\\")
        key = name.replace("_", "").replace("2", "Two").replace("18", "Eighteen")
        macro(f"lat{key}", f"{L['total_ms']['mean']:.0f}")
        macro(f"vram{key}", f"{L['peak_vram_mb'] / 1024:.1f}")
    with open(f"{GEN}/tab_latency_rows.tex", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    if "vjepa2_ac" in lat and "bc_resnet18" in lat:
        macro("latRatio", f"{lat['vjepa2_ac']['total_ms']['mean'] / lat['bc_resnet18']['total_ms']['mean']:.0f}")
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for name in MAIN:
        if name in lat and name in pooled:
            v = pooled[name]
            x = lat[name]["total_ms"]["mean"]
            y = v["k"] / v["n"]
            ax.errorbar(x, 100 * y, yerr=[[100 * (y - v["lo"])], [100 * (v["hi"] - y)]], fmt="o", color=C[name],
                        capsize=2, ms=4, label=NAME[name])
    ax.set_xscale("log")
    ax.set(xlabel="latency per decision (ms, log scale)", ylabel="success rate (%), 95% CI")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    save(fig, "pareto")


# ----------------------------------------------------------------------------- ablations
def ablation_assets() -> None:
    base = load_eval("vjepa2_ac").get("turn_on_the_stove")
    rows = [("Main", "$H{=}10$, $N{=}256$, $\\kappa{=}0.05$, L1", base, "vjepa2_ac")]
    spec = [("Horizon", "$H{=}3$", "abl_H3"), ("Horizon", "$H{=}5$", "abl_H5"), ("Horizon", "$H{=}15$", "abl_H15"),
            ("Samples", "$N{=}32$", "abl_N32"), ("Samples", "$N{=}64$", "abl_N64"), ("Samples", "$N{=}128$", "abl_N128"),
            ("Samples", "$N{=}512$", "abl_N512"), ("Elites", "$\\kappa{=}0.1$", "abl_K0.1"),
            ("Elites", "$\\kappa{=}0.2$", "abl_K0.2"), ("Distance", "cosine", "abl_cost_cosine"),
            ("Distance", "Euclidean $L_2$", "abl_cost_l2"), ("Distance", "Mahalanobis (diag.)", "abl_cost_mahalanobis")]
    for group, label, tag in spec:
        rows.append((group, label, load_eval(tag).get("turn_on_the_stove"), tag))
    lines, data = [], {}
    for group, label, ev, tag in rows:
        if ev is None:
            lines.append(f"{group} & {label} & \\multicolumn{{5}}{{c}}{{pending}} \\\\")
            continue
        eps = ev["episodes_detail"]
        k = sum(e["success"] for e in eps)
        lo, hi = wilson(k, len(eps))
        err = np.mean([e["final_goal_error_cm"] for e in eps])
        minerr = np.mean([e["min_goal_error_cm"] for e in eps])
        st = np.mean([e["env_steps"] for e in eps if e["success"]]) if k else float("nan")
        pl = ev["latency_ms"]["plan_ms"]
        data[tag] = (k, len(eps), err, pl)
        mk = {"vjepa2_ac": "Main", "abl_H3": "Hthree", "abl_H5": "Hfive", "abl_H15": "Hfifteen", "abl_N32": "Nthirtytwo",
              "abl_N64": "Nsixtyfour", "abl_N128": "Nonetwentyeight", "abl_N512": "Nfivetwelve", "abl_K0.1": "Kten",
              "abl_K0.2": "Ktwenty", "abl_cost_cosine": "Cosine", "abl_cost_l2": "Ltwo",
              "abl_cost_mahalanobis": "Mahal"}[tag]
        macro(f"kabl{mk}", k)
        macro(f"errabl{mk}", f"{err:.1f}")
        lines.append(f"{group} & {label} & {k}/{len(eps)} [{fmt_pct(lo)}, {fmt_pct(hi)}] & {err:.1f} & {minerr:.1f} & "
                     f"{'n/a' if np.isnan(st) else f'{st:.0f}'} & {pl / 1000:.1f} \\\\")
    with open(f"{GEN}/tab_ablation_rows.tex", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    json.dump(data, open("results/tables/ablation_summary.json", "w"), indent=1)


def write_all() -> None:
    training_facts()
    pooled = main_table()
    prediction_assets()
    latency_assets(pooled)
    ablation_assets()
    write_macros()




# ----------------------------------------------------------------------------- qualitative assets
def rollout(tag: str, task: str, ep: int):
    p = f"results/rollouts/{tag}/{task}__ep{ep:02d}.npz"
    return np.load(p) if os.path.exists(p) else None


def pick_episode(tag: str, task: str, success: bool):
    ev = load_eval(tag).get(task)
    if ev is None:
        return None
    for j, e in enumerate(ev["episodes_detail"]):
        if e["success"] == success and os.path.exists(f"results/rollouts/{tag}/{task}__ep{j:02d}.npz"):
            return j
    return None


def filmstrips() -> None:
    """8 evenly spaced frames per episode: success and failure for each recorded agent on two tasks."""
    rows = []
    for task in ["turn_on_the_stove", "push_the_plate_to_the_front_of_the_stove"]:
        for tag in ["vjepa2_ac", "dino_wm", "bc_resnet18"]:
            for succ in (True, False):
                j = pick_episode(tag, task, succ)
                if j is not None:
                    rows.append((tag, task, j, succ))
    if not rows:
        return
    n = 8
    fig, axes = plt.subplots(len(rows), n + 1, figsize=(7.0, 0.80 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (tag, task, j, succ) in enumerate(rows):
        d = rollout(tag, task, j)
        fr = d["frames"]
        idx = np.linspace(0, len(fr) - 1, n).round().astype(int)
        ev = load_eval(tag)[task]["episodes_detail"][j]
        for c, i in enumerate(idx):
            ax = axes[r, c]
            ax.imshow(fr[i])
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.text(3, 13, f"t={5 * i}", color="w", fontsize=5.2, fontweight="bold",
                    bbox=dict(boxstyle="square,pad=0.1", fc="black", ec="none", alpha=0.45))
        ax = axes[r, n]
        ax.imshow(d["goal"])
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#0062A0" if succ else "#D55E00")
            s.set_linewidth(1.5)
        if r == 0:
            ax.set_title("goal", fontsize=6.5, pad=2)
        axes[r, 0].set_ylabel(f"{NAME[tag]}\n{TASKS[task].short}, {'success' if succ else 'failure'}\n"
                              f"{ev['final_goal_error_cm']:.1f} cm", fontsize=5.8, rotation=0, ha="right", va="center",
                              labelpad=2)
    fig.subplots_adjust(wspace=0.03, hspace=0.06, left=0.17, right=0.995, top=0.95, bottom=0.01)
    save(fig, "filmstrips")


def goal_latents(tag: str, task: str, eps: list) -> dict:
    """Pooled goal embeddings, computed with the frozen encoder and cached per episode."""
    cache = f"results/tables/goal_latents_{tag}_{task}.npz"
    out = {}
    if os.path.exists(cache):
        d = np.load(cache)
        out = {int(k[2:]): d[k] for k in d.files}
    missing = [(j, d) for j, d in eps if j not in out]
    if missing:
        import torch
        from src.models.encoders import build_encoder

        enc = build_encoder("vjepa2" if tag == "vjepa2_ac" else "dinov2")
        with torch.no_grad():
            for j, d in missing:
                out[j] = enc(torch.from_numpy(d["goal"][None])).mean(1)[0].numpy()
        np.savez(cache, **{f"ep{j}": v for j, v in out.items()})
    return out


def manifolds() -> None:
    """PCA and t-SNE of realised latents, imagined plans and goal latents along recorded Stove episodes.

    Also reports how far imagined final frames sit from the realised manifold: the distance from each imagined
    plan end to its nearest realised latent, divided by the median nearest-neighbour distance among realised
    latents (1 = as close as real frames are to each other)."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    task = "turn_on_the_stove"
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.9))
    for row, tag in enumerate(["vjepa2_ac", "dino_wm"]):
        eps = [(j, rollout(tag, task, j)) for j in range(10)]
        eps = [(j, d) for j, d in eps if d is not None and "realized_pooled" in d.files]
        if not eps:
            continue
        ev = load_eval(tag).get(task)
        goals = goal_latents(tag, task, eps)
        real = np.concatenate([d["realized_pooled"] for _, d in eps])
        imag_end = np.concatenate([d["imagined_pooled"][:, -1] for _, d in eps])
        gl = np.stack([goals[j] for j, _ in eps])
        # off-manifold distance of imagined plan ends
        from scipy.spatial import cKDTree
        tree = cKDTree(real)
        nn_real = tree.query(real, k=2)[0][:, 1]
        nn_imag = tree.query(imag_end, k=1)[0]
        ratio = float(np.median(nn_imag) / np.median(nn_real))
        spread = float(np.mean(np.linalg.norm(imag_end - imag_end.mean(0), axis=1)) /
                       np.mean(np.linalg.norm(real - real.mean(0), axis=1)))
        key = tag.replace("_", "").replace("2", "Two")
        macro(f"offManifold{key}", f"{ratio:.1f}")
        macro(f"imagSpread{key}", f"{spread:.2f}")
        pca = PCA(2, random_state=0).fit(real)
        ts = TSNE(2, random_state=0, init="pca", perplexity=30).fit_transform(np.concatenate([real, imag_end, gl]))
        # draw at most three successful and three failed episodes so the paths stay readable;
        # the ratios above use every episode
        succ_ids = [j for j, _ in eps if ev and ev["episodes_detail"][j]["success"]][:3]
        fail_ids = [j for j, _ in eps if not (ev and ev["episodes_detail"][j]["success"])][:3]
        shown = set(succ_ids + fail_ids)
        off = 0
        for j, d in eps:
            n_ = len(d["realized_pooled"])
            if j in shown:
                succ = j in succ_ids
                col = "#0072B2" if succ else "#D55E00"
                pr = pca.transform(d["realized_pooled"])
                pi = pca.transform(d["imagined_pooled"][:, -1])
                axes[row, 0].plot(pr[:, 0], pr[:, 1], "-", color=col, lw=0.9, alpha=0.9)
                axes[row, 0].scatter(pi[:, 0], pi[:, 1], s=9, color=col, alpha=0.55, marker="x", lw=0.7)
                axes[row, 0].scatter(*pr[0], s=14, color="k", zorder=3)
                g = pca.transform(goals[j][None])[0]
                axes[row, 0].scatter(*g, s=70, marker="*", color="#009E73", edgecolor="k", lw=0.5, zorder=4)
                tr = ts[off : off + n_]
                ti = ts[len(real) + off : len(real) + off + n_]
                axes[row, 1].plot(tr[:, 0], tr[:, 1], "-", color=col, lw=0.9, alpha=0.9)
                axes[row, 1].scatter(ti[:, 0], ti[:, 1], s=9, color=col, alpha=0.55, marker="x", lw=0.7)
            off += n_
        tg = ts[len(real) + len(imag_end) :]
        axes[row, 1].scatter(tg[:, 0], tg[:, 1], s=70, marker="*", color="#009E73", edgecolor="k", lw=0.5, zorder=4)
        # imagined H-step path from a few decisions of the first episode, against what followed
        j0, d0 = next((j, d) for j, d in eps if j in succ_ids) if succ_ids else eps[0]
        for i in range(0, len(d0["realized_pooled"]) - 1, 8):
            path = pca.transform(np.concatenate([d0["realized_pooled"][i : i + 1], d0["imagined_pooled"][i]]))
            axes[row, 0].plot(path[:, 0], path[:, 1], "--", color="0.3", lw=0.8)
        nm = NAME[tag]
        axes[row, 0].set_title(f"({'ab'[row]}) {nm}: PCA (imagined/real NN ratio {ratio:.1f})")
        axes[row, 1].set_title(f"({'cd'[row]}) {nm}: t-SNE")
        for c in (0, 1):
            axes[row, c].set_xticks([]), axes[row, c].set_yticks([])
    from matplotlib.lines import Line2D
    axes[0, 1].legend(handles=[Line2D([], [], color="#0072B2", label="realised path, success"),
                               Line2D([], [], color="#D55E00", label="realised path, failure"),
                               Line2D([], [], color="0.4", marker="x", ls="", label="imagined plan end"),
                               Line2D([], [], color="0.35", ls="--", label="imagined 10-step plan"),
                               Line2D([], [], color="#009E73", marker="*", ls="", ms=7, label="goal embedding"),
                               Line2D([], [], color="k", marker="o", ls="", ms=3, label="start")],
                      frameon=False, fontsize=6, loc="best")
    fig.tight_layout()
    save(fig, "manifolds")


def energy_assets() -> None:
    """Predicted energy versus executed outcome over the first action's (dx, dy) command."""
    order = [("vjepa2_ac", "turn_on_the_stove", 10), ("vjepa2_ac", "turn_on_the_stove", 3),
             ("dino_wm", "turn_on_the_stove", 10), ("vjepa2_ac", "push_the_plate_to_the_front_of_the_stove", 10),
             ("dino_wm", "push_the_plate_to_the_front_of_the_stove", 10)]
    files = [(a_, t_, h_, f"results/tables/energy_{a_}_{t_}_ep0_H{h_}.json") for a_, t_, h_ in order]
    files = [x for x in files if os.path.exists(x[3])]
    if not files:
        return
    fig, axes = plt.subplots(2, len(files), figsize=(7.0, 3.3), squeeze=False)
    summary = {}
    for c, (agent, task, H, f) in enumerate(files):
        e = json.load(open(f))
        x = np.array(e["grid"])
        E, T = np.array(e["energy"]), np.array(e["true_goal_error_cm"])
        flat = float(np.ptp(T)) < 1e-6
        rho = float("nan") if flat else float(stats.spearmanr(E.ravel(), T.ravel()).correlation)
        iy, ix = np.unravel_index(np.argmin(E), E.shape)
        key = (agent.replace("_", "").replace("2", "Two") + TASKS[task].short.replace("-", "")
               + {3: "Hthree", 10: ""}[H])
        summary[key] = {"rho": rho, "energy_range_pct": float(100 * np.ptp(E) / E.mean()),
                        "true_min": float(T.min()), "true_max": float(T.max()), "true_at_energy_min": float(T[iy, ix])}
        macro(f"rho{key}", "undefined" if flat else f"{rho:.2f}")
        macro(f"erange{key}", f"{100 * np.ptp(E) / E.mean():.1f}")
        macro(f"tmin{key}", f"{T.min():.1f}")
        macro(f"tmax{key}", f"{T.max():.1f}")
        macro(f"tatmin{key}", f"{T[iy, ix]:.1f}")
        for r, (Z, lab) in enumerate([(E, "predicted energy"), (T, "true goal error (cm)")]):
            ax = axes[r, c]
            if r == 1 and flat:
                im = ax.imshow(np.zeros_like(Z), extent=(-1, 1, -1, 1), cmap="Greys", vmin=0, vmax=4, origin="lower",
                               aspect="auto")
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
                cb.ax.set_visible(False)  # keep the panel the same size as its neighbours
                ax.text(0, -0.72, f"{T.min():.1f} cm at every point", ha="center", va="center", fontsize=6,
                        bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
            else:
                cs = ax.contourf(x, x, Z, levels=12, cmap="viridis")
                cb = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.03)
                cb.ax.tick_params(labelsize=5)
            path = np.array(e["cem_path_xy"])
            ax.plot(path[:, 0], path[:, 1], "-o", color="w" if not (r == 1 and flat) else "k", ms=1.8, lw=0.7)
            ax.plot(path[-1, 0], path[-1, 1], "*", color="#E69F00", ms=6)
            ax.plot(x[ix], x[iy], "s", mfc="none", mec="#D55E00", ms=4.5)
            ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
            ax.tick_params(labelsize=5.5)
            if r == 1:
                ax.set_xlabel("first action $\\Delta x$", fontsize=6.5)
            if c == 0:
                ax.set_ylabel(f"{lab}\n$\\Delta y$", fontsize=6.5)
            if r == 0:
                rt = "n/a" if flat else f"{rho:.2f}"
                ax.set_title(f"{NAME[agent]}\n{TASKS[task].short}, $H$={H}, $\\rho_s$={rt}", fontsize=6.5)
    fig.tight_layout(w_pad=0.6, h_pad=0.6)
    save(fig, "energy_landscapes")
    json.dump(summary, open("results/tables/energy_summary.json", "w"), indent=1)


# ----------------------------------------------------------------------------- failure taxonomy
def contact_distance(ep: dict) -> np.ndarray:
    return np.array([100 * np.linalg.norm(np.array(m["ee"]) - np.array(m["contact"])) for m in ep["measurement_trace"]])


def latent_diagnostics(tag: str, task: str, j: int, ep: dict) -> dict | None:
    d = rollout(tag, task, j)
    if d is None or "goal_cost_now" not in d.files:
        return None
    energy = d["goal_cost_now"]
    err = np.array(ep["goal_error_trace"][: len(energy)])
    rho = float(stats.spearmanr(energy, err).correlation) if len(energy) > 3 and np.std(err) > 0 else float("nan")
    real = d["realized"].astype(np.float32)
    imag = d["imagined_next"].astype(np.float32)
    n = min(len(imag) - 1, len(real) - 1)
    pred_err = np.abs(imag[:n] - real[1 : n + 1]).mean((1, 2))
    copy_err = np.abs(real[:n] - real[1 : n + 1]).mean((1, 2))
    believed = float(np.median((d["goal_cost_now"] - d["plan_cost"]) / d["goal_cost_now"]))
    return {"rho": rho, "drift_ratio": float(np.median(pred_err / np.maximum(copy_err, 1e-6))), "believed_gain": believed}


def failure_taxonomy() -> None:
    # Contact radius per task: 1.5 x the median gripper-to-part distance at the moment of success, pooled over
    # every successful episode of every evaluated agent, including the BC variants (8 cm without any success).
    radius = {}
    for t in TASK_KEYS:
        ds = []
        for tag in MAIN + ["bc_resnet18_wrist", "bc_dinov2_frozen", "bc_vjepa2_frozen", "bc_dinov2_finetune"]:
            ev = load_eval(tag).get(t)
            if ev:
                ds += [contact_distance(e)[-1] for e in ev["episodes_detail"] if e["success"]]
        radius[t] = 1.5 * float(np.median(ds)) if ds else 8.0
    rows, counts, diag = [], {}, {}
    for tag in MAIN:
        c = {"approach": 0, "contact": 0, "partial": 0, "success": 0}
        flags = {"ambiguity": 0, "drift": 0, "n_fail": 0}
        dg_all = []
        for t in TASK_KEYS:
            ev = load_eval(tag).get(t)
            if not ev:
                continue
            for j, e in enumerate(ev["episodes_detail"]):
                if e["success"]:
                    c["success"] += 1
                    dg = latent_diagnostics(tag, t, j, e) if tag in ("vjepa2_ac", "dino_wm") else None
                    if dg:
                        dg_all.append(dict(dg, success=True, task=t))
                    continue
                reached = contact_distance(e).min() <= radius[t]
                prog = (e["initial_goal_error_cm"] - e["final_goal_error_cm"]) / max(e["initial_goal_error_cm"], 1e-6)
                cat = "approach" if not reached else ("contact" if prog < 0.25 else "partial")
                c[cat] += 1
                if tag in ("vjepa2_ac", "dino_wm"):
                    dg = latent_diagnostics(tag, t, j, e)
                    if dg:
                        flags["n_fail"] += 1
                        flags["ambiguity"] += int(not dg["rho"] >= 0.3)
                        flags["drift"] += int(dg["drift_ratio"] > 1.0)
                        dg_all.append(dict(dg, success=False, task=t, category=cat))
        counts[tag], diag[tag] = c, {"flags": flags, "episodes": dg_all}
    json.dump({"radius_cm": radius, "counts": counts, "latent": diag}, open("results/tables/failures.json", "w"), indent=1)
    for t in TASK_KEYS:
        macro(f"radius{TASKS[t].short.replace('-', '')}", f"{radius[t]:.1f}")
    for tag in MAIN:
        key = tag.replace("_", "").replace("2", "Two").replace("18", "Eighteen")
        for cat in ["approach", "contact", "partial"]:
            macro(f"fail{cat.capitalize()}{key}", counts[tag][cat])
        if tag in ("vjepa2_ac", "dino_wm"):
            f = diag[tag]["flags"]
            macro(f"failAmbig{key}", f["ambiguity"])
            macro(f"failDrift{key}", f["drift"])
            macro(f"failN{key}", f["n_fail"])
            eps = diag[tag]["episodes"]
            for lab, sel in [("Succ", True), ("Fail", False)]:
                v = [e["rho"] for e in eps if e["success"] == sel and not np.isnan(e["rho"])]
                macro(f"rho{lab}{key}", f"{np.median(v):.2f}" if v else "n/a")
                g = [e["believed_gain"] for e in eps if e["success"] == sel]
                macro(f"gain{lab}{key}", f"{100 * np.median(g):.1f}" if g else "n/a")
                dr = [e["drift_ratio"] for e in eps if e["success"] == sel]
                macro(f"drift{lab}{key}", f"{np.median(dr):.2f}" if dr else "n/a")
    # Figure: (a) outcome categories per model, (b) latent diagnostics of the planners, (c) snapshots.
    fig = plt.figure(figsize=(7.0, 3.3))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.85])
    ax = fig.add_subplot(gs[0, :2])
    cats = [("success", "success", "#0072B2"), ("partial", "partial progress", "#56B4E9"),
            ("contact", "contact, no progress", "#E69F00"), ("approach", "never reached part", "#D55E00")]
    left = np.zeros(len(MAIN))
    for key, lab, col in cats:
        vals = np.array([counts[t][key] for t in MAIN])
        ax.barh(range(len(MAIN)), vals, left=left, color=col, label=lab, height=0.6)
        left += vals
    ax.set_yticks(range(len(MAIN)), [NAME[t] for t in MAIN])
    ax.invert_yaxis()
    ax.set_xlabel("episodes (of 40)")
    ax.set_title("(a) outcome of every episode")
    ax.legend(frameon=False, fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.45, -0.28))
    ax2 = fig.add_subplot(gs[0, 2:])
    for tag, mk in [("vjepa2_ac", "o"), ("dino_wm", "s")]:
        eps = diag[tag]["episodes"]
        for sel, col in [(True, "#0072B2"), (False, "#D55E00")]:
            pts = [(e["rho"], 100 * e["believed_gain"]) for e in eps if e["success"] == sel and not np.isnan(e["rho"])]
            if pts:
                p = np.array(pts)
                ax2.scatter(p[:, 0], p[:, 1], marker=mk, s=14, facecolor=col if mk == "o" else "none", edgecolor=col,
                            lw=0.8, label=f"{NAME[tag]}, {'success' if sel else 'failure'}")
    ax2.axvline(0.3, color="0.5", lw=0.6, ls="--")
    ax2.set(xlabel=r"energy-error rank correlation $\rho_s$ (low: goal ambiguity)",
            ylabel="predicted energy drop (%)", title="(b) latent diagnostics per planner episode")
    ax2.legend(frameon=False, fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    # snapshots: last frame of one failure per category for V-JEPA 2-AC, with the goal frame
    snaps = []
    for e in diag["vjepa2_ac"]["episodes"]:
        if not e["success"] and e.get("category") not in [s_[0] for s_ in snaps]:
            snaps.append((e["category"], e["task"]))
    for i, (cat, task) in enumerate(snaps[:2]):
        ev = load_eval("vjepa2_ac")[task]["episodes_detail"]
        for j, ep in enumerate(ev):
            if ep["success"]:
                continue
            reached = contact_distance(ep).min() <= radius[task]
            prog = (ep["initial_goal_error_cm"] - ep["final_goal_error_cm"]) / max(ep["initial_goal_error_cm"], 1e-6)
            c_ = "approach" if not reached else ("contact" if prog < 0.25 else "partial")
            if c_ == cat:
                d = rollout("vjepa2_ac", task, j)
                axl = fig.add_subplot(gs[1, 2 * i])
                axl.imshow(d["frames"][-1]); axl.set_xticks([]); axl.set_yticks([]); axl.axis("off")
                axl.set_title(f"V-JEPA 2-AC, {TASKS[task].short}: final", fontsize=6.5)
                axr = fig.add_subplot(gs[1, 2 * i + 1])
                axr.imshow(d["goal"]); axr.set_xticks([]); axr.set_yticks([]); axr.axis("off")
                axr.set_title("goal (" + {"approach": "never reached part", "contact": "contact, no progress",
                                          "partial": "partial progress"}[cat] + ")", fontsize=6.5)
                break
    fig.tight_layout()
    save(fig, "failures")


def write_everything() -> None:
    write_all()
    filmstrips()
    manifolds()
    energy_assets()
    failure_taxonomy()
    energy_attribution()
    action_stats()
    extra_tests()
    write_macros()



def energy_attribution() -> None:
    """Does the planning energy track the arm or the object?

    For every planner episode, Spearman correlations along the episode between the energy of the observed frame
    and (i) the gripper's distance to its goal position (end-effector position in the final state of the
    held-out demonstration) and (ii) the task error of the manipulated part (object tasks only)."""
    from src.data.libero_data import load_demos

    goal_ee = {(d.task, d.index): d.state[-1][:3] for d in load_demos() if d.split == "test"}
    out = {}
    for tag in ["vjepa2_ac", "dino_wm"]:
        arm, obj = [], []
        for t in TASK_KEYS:
            ev = load_eval(tag).get(t)
            if not ev:
                continue
            for j, e in enumerate(ev["episodes_detail"]):
                d = rollout(tag, t, j)
                if d is None:
                    continue
                en = d["goal_cost_now"]
                n = len(en)
                ee = np.array([m["ee"] for m in e["measurement_trace"][:n]])
                ee_d = np.linalg.norm(ee - goal_ee[(t, e["demo"])], axis=1)
                if np.std(ee_d) > 0:
                    arm.append(stats.spearmanr(en, ee_d).correlation)
                if t != "turn_on_the_stove":
                    oe = np.array(e["goal_error_trace"][:n])
                    if np.std(oe) > 1e-3:
                        obj.append(stats.spearmanr(en, oe).correlation)
        out[tag] = {"arm_median": float(np.nanmedian(arm)), "arm_n": len(arm),
                    "obj_median": float(np.nanmedian(obj)) if obj else float("nan"), "obj_n": len(obj)}
        key = tag.replace("_", "").replace("2", "Two")
        macro(f"rhoArm{key}", f"{out[tag]['arm_median']:.2f}")
        macro(f"rhoObj{key}", f"{out[tag]['obj_median']:.2f}" if obj else "n/a")
        macro(f"nObj{key}", len(obj))
        macro(f"nArm{key}", len(arm))
    json.dump(out, open("results/tables/energy_attribution.json", "w"), indent=1)
    print("energy attribution", out)


def action_stats() -> None:
    """Executed planner actions versus demonstrated macro actions (main configuration)."""
    from src.data.libero_data import load_demos

    demos = [d for d in load_demos() if d.split == "train"]
    A = np.concatenate([np.stack([d.action[t : t + 5].mean(0) for t in range(0, len(d.action) - 5, 5)]) for d in demos])
    X = np.concatenate([np.load(f)["action"] for tag in ["vjepa2_ac", "dino_wm"]
                        for f in glob.glob(f"results/rollouts/{tag}/*.npz")])
    macro("rotDemo", f"{np.abs(A[:, 3:6]).mean():.2f}")
    macro("rotPlanner", f"{np.abs(X[:, 3:6]).mean():.2f}")
    macro("gripPlanner", f"{np.abs(X[:, 6]).mean():.2f}")
    macro("gripDemo", f"{np.abs(A[:, 6]).mean():.2f}")
    json.dump({"demo_abs_mean": np.abs(A).mean(0).tolist(), "planner_abs_mean": np.abs(X).mean(0).tolist(),
               "demo_std": A.std(0).tolist()}, open("results/tables/action_stats.json", "w"), indent=1)


def extra_tests() -> None:
    """Paired exact McNemar tests for the ablations (same held-out episodes)."""
    def succ(tag, tasks):
        out = []
        for t in tasks:
            ev = load_eval(tag).get(t)
            if ev is None:
                return None
            out += [e["success"] for e in sorted(ev["episodes_detail"], key=lambda e: e["demo"])]
        return np.array(out)

    res = {}
    pairs = [("pHthree", "abl_H3", "vjepa2_ac", ["turn_on_the_stove"]),
             ("pHfive", "abl_H5", "vjepa2_ac", ["turn_on_the_stove"]),
             ("pPriorVjepa", "vjepa2_ac_dataprior", "vjepa2_ac", TASK_KEYS),
             ("pPriorDino", "dino_wm_dataprior", "dino_wm", TASK_KEYS),
             ("pBcDinoFrozen", "bc_dinov2_frozen", "bc_resnet18", TASK_KEYS),
             ("pBcVjepaFrozen", "bc_vjepa2_frozen", "bc_resnet18", TASK_KEYS),
             ("pBcDinoFt", "bc_dinov2_finetune", "bc_resnet18", TASK_KEYS),
             ("pBcWrist", "bc_resnet18_wrist", "bc_resnet18", TASK_KEYS),
             ("pBcVjepaVsDino", "bc_vjepa2_frozen", "bc_dinov2_frozen", TASK_KEYS),
             ("pHthreeAllVjepa", "vjepa2_ac_H3", "vjepa2_ac", TASK_KEYS),
             ("pHthreeAllDino", "dino_wm_H3", "dino_wm", TASK_KEYS),
             ("pHthreeVjepaVsBc", "vjepa2_ac_H3", "bc_resnet18", TASK_KEYS),
             ("pHthreeDinoVsBc", "dino_wm_H3", "bc_resnet18", TASK_KEYS)]
    for name, a, b, tasks in pairs:
        sa, sb = succ(a, tasks), succ(b, tasks)
        if sa is None or sb is None:
            continue
        p = mcnemar_exact(sa, sb)
        res[name] = {"a": a, "b": b, "ka": int(sa.sum()), "kb": int(sb.sum()), "n": len(sa), "p": p}
        macro(name, f"={p:.3f}" if p >= 0.001 else "<0.001")
    for tag, key in [("vjepa2_ac_dataprior", "PriorVjepa"), ("dino_wm_dataprior", "PriorDino"),
                     ("vjepa2_ac_H3", "HthreeVjepa"), ("dino_wm_H3", "HthreeDino")]:
        ev = load_eval(tag)
        if len(ev) == 4:
            ks = [int(round(ev[t]["success_rate"] * 10)) for t in TASK_KEYS]
            for t, k in zip(TASK_KEYS, ks):
                macro(f"k{key}{TASKS[t].short.replace('-', '')}", k)
                macro(f"err{key}{TASKS[t].short.replace('-', '')}", f"{ev[t]['final_goal_error_cm']:.1f}")
            macro(f"k{key}", sum(ks))
    json.dump(res, open("results/tables/ablation_tests.json", "w"), indent=1)


if __name__ == "__main__":
    write_everything()
    print("macros:", len(macros))
