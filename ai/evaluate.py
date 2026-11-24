# avaliacao final: junta os resultados da regra e do modelo de janela e gera as figuras do relatorio
# precisa rodar antes: ai/baseline_rule.py e ai/train_window_model.py

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # salva sem abrir janela
import matplotlib.pyplot as plt

# pra conseguir importar ai/ rodando daqui
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai.baseline_rule import wilson_ci

RESULTS_DIR = ROOT / "docs" / "results"
FIGURES_DIR = ROOT / "docs" / "figures"

# prevalencias pro VPP: atencao primaria (premissa) e o nosso dataset (encaminhados)
PREVALENCES = {"primary care (assumed)": 0.15, "referral cohort (UCDDB)": 0.56}

# modelo de janela escolhido (o mais simples, empatou com o gboost)
CHOSEN_MODEL = "logistic"


# le os csv gerados pelos outros scripts
def load_results():
    rule = pd.read_csv(RESULTS_DIR / "baseline_rule_nights.csv", dtype={"subject": str})
    nights = pd.read_csv(RESULTS_DIR / "window_model_nights.csv", dtype={"subject": str})
    folds = pd.read_csv(RESULTS_DIR / "window_model_folds.csv", dtype={"subject": str})
    return rule, nights, folds


# sens, espec e IC a partir das colunas true_osa e pred_osa
def confusion(df):
    tp = int((df.true_osa & df.pred_osa).sum())
    fn = int((df.true_osa & ~df.pred_osa).sum())
    fp = int((~df.true_osa & df.pred_osa).sum())
    tn = int((~df.true_osa & ~df.pred_osa).sum())
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "sens": tp / (tp + fn), "sens_ci": wilson_ci(tp, tp + fn),
        "spec": tn / (tn + fp), "spec_ci": wilson_ci(tn, tn + fp),
        "correct": tp + tn, "n": len(df),
    }


# VPP = S*p / (S*p + (1-E)*(1-p))
def ppv(sens, spec, p):
    return sens * p / (sens * p + (1 - spec) * (1 - p))


# VPN = E*(1-p) / (E*(1-p) + (1-S)*p)
def npv(sens, spec, p):
    return spec * (1 - p) / (spec * (1 - p) + (1 - sens) * p)


# Bland-Altman: diferenca (ODI - AHI) contra a media dos dois
def bland_altman(rule, path):
    mean = (rule.odi3_per_h + rule.ahi) / 2
    diff = rule.odi3_per_h - rule.ahi
    bias = diff.mean()
    sd = diff.std(ddof=1)
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(mean, diff, color="tab:blue")
    ax.axhline(bias, color="black", lw=1.2, label=f"bias {bias:.1f}/h")
    ax.axhline(lo, color="gray", ls="--", lw=1, label=f"95% limits {lo:.1f} to {hi:.1f}/h")
    ax.axhline(hi, color="gray", ls="--", lw=1)
    ax.axhline(0, color="lightgray", lw=0.8)

    # marca os pontos q mais se afastam
    for _, r in rule.assign(mean=mean, diff=diff).iterrows():
        if abs(r["diff"] - bias) > 1.5 * sd:
            ax.annotate(r.subject, (r["mean"], r["diff"]), textcoords="offset points", xytext=(5, 4))

    ax.set_xlabel("mean of ODI3 and PSG AHI (events/h)")
    ax.set_ylabel("ODI3 - PSG AHI (events/h)")
    ax.set_title("Bland-Altman: ODI3 vs PSG AHI (25 nights)")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"bias": bias, "sd": sd, "loa_low": lo, "loa_high": hi}


# AUC de cada noite (LOSO) dos dois modelos
def plot_fold_auc(folds, path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    models = list(folds.model.unique())
    for i, m in enumerate(models):
        a = folds[folds.model == m].auc.dropna()
        x = np.full(len(a), i) + np.random.default_rng(0).uniform(-0.12, 0.12, len(a))
        ax.scatter(x, a, alpha=0.7)
        ax.hlines(a.mean(), i - 0.25, i + 0.25, color="black", lw=2)
        ax.text(i + 0.28, a.mean(), f"{a.mean():.2f} ± {a.std():.2f}", va="center")
    ax.axhline(0.5, color="lightgray", ls="--", lw=1)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_xlim(-0.5, len(models) - 0.2)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("window-level AUC (held-out night)")
    ax.set_title("Leave-one-subject-out: one point per night")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# o q acontece com 1000 pessoas testadas (quadradinhos), com sens e espec medidas
def plot_screening(sens, spec, p, path, title):
    pos = round(1000 * p)
    tp = round(pos * sens)
    fn = pos - tp
    fp = round((1000 - pos) * (1 - spec))
    tn = 1000 - pos - fp

    # cada quadradinho eh uma pessoa: 40 colunas x 25 linhas
    cats = [("true referral", tp, "#1b9e9e"), ("false referral", fp, "#e4572e"),
            ("missed", fn, "#b8860b"), ("correctly reassured", tn, "#d5dbe3")]
    colors = [c for _, k, c in cats for _ in range(k)]

    fig, ax = plt.subplots(figsize=(8, 5.2))
    for i, c in enumerate(colors):
        col, row = divmod(i, 25)
        ax.add_patch(plt.Rectangle((col, 24 - row), 0.85, 0.85, color=c))
    ax.set_xlim(0, 40)
    ax.set_ylim(-0.5, 25)
    ax.set_aspect("equal")
    ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in cats]
    ax.legend(handles, [f"{n} · {k}" for n, k, _ in cats], loc="upper center",
              bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False, fontsize=9)
    ax.set_title(f"{title}\n{tp + fp} referrals → PPV {tp / (tp + fp):.2f}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rule, nights, folds = load_results()

    # 1. regra x modelo de janela no nivel da noite
    rows = []
    rule_cm = confusion(rule)
    rows.append({"method": "rule ODI3 >= 15", **rule_cm})
    for m in nights.model.unique():
        rows.append({"method": f"window model ({m})", **confusion(nights[nights.model == m])})

    print("nivel da noite (AHI >= 15), 25 noites:")
    for r in rows:
        print(f"  {r['method']:<28} sens {r['tp']}/{r['tp'] + r['fn']} = {r['sens']:.2f} "
              f"(IC {r['sens_ci'][0]:.2f}-{r['sens_ci'][1]:.2f}) | "
              f"espec {r['tn']}/{r['tn'] + r['fp']} = {r['spec']:.2f} "
              f"(IC {r['spec_ci'][0]:.2f}-{r['spec_ci'][1]:.2f}) | acertos {r['correct']}/{r['n']}")

    # 2. AUC da janela (LOSO)
    print("\nAUC da janela (LOSO, uma noite por dobra):")
    for m in folds.model.unique():
        a = folds[folds.model == m].auc.dropna()
        print(f"  {m:<10} {a.mean():.3f} ± {a.std():.3f} (min {a.min():.3f}, max {a.max():.3f})")
    plot_fold_auc(folds, FIGURES_DIR / "window_auc_per_night.png")

    # 3. Bland-Altman do ODI3 contra o AHI
    ba = bland_altman(rule, FIGURES_DIR / "bland_altman_odi3_ahi.png")
    print(f"\nBland-Altman ODI3 - AHI: vies {ba['bias']:.1f}/h, limites {ba['loa_low']:.1f} a {ba['loa_high']:.1f}/h")

    # 4. VPP e VPN com a sens e espec MEDIDAS da regra (troca os 0.85 ilustrativos do mid-term)
    print("\nVPP / VPN da regra com os valores medidos:")
    ppv_rows = []
    for name, p in PREVALENCES.items():
        val_ppv = ppv(rule_cm["sens"], rule_cm["spec"], p)
        val_npv = npv(rule_cm["sens"], rule_cm["spec"], p)
        ppv_rows.append({"scenario": name, "prevalence": p, "ppv": round(val_ppv, 2), "npv": round(val_npv, 2)})
        print(f"  {name:<26} prevalencia {p:.2f}: VPP {val_ppv:.2f}, VPN {val_npv:.2f}")

    waffle = plot_screening(rule_cm["sens"], rule_cm["spec"], 0.15, FIGURES_DIR / "screening_1000_primary_care.png", "1,000 people in primary care (15% prevalence), measured sens/spec of the rule")
    print(f"  1000 pessoas na atencao primaria: {waffle}")

    # salva as tabelas
    summary = pd.DataFrame([{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("sens_ci", "spec_ci")} | {"sens_ci": f"{r['sens_ci'][0]:.2f}-{r['sens_ci'][1]:.2f}", "spec_ci": f"{r['spec_ci'][0]:.2f}-{r['spec_ci'][1]:.2f}"} for r in rows])
    summary.to_csv(RESULTS_DIR / "summary_night_level.csv", index=False)
    pd.DataFrame(ppv_rows).to_csv(RESULTS_DIR / "ppv_scenarios.csv", index=False)
    pd.DataFrame([ba]).round(2).to_csv(RESULTS_DIR / "bland_altman.csv", index=False)

    print("\nfiguras em docs/figures/ e tabelas em docs/results/")
    print("tudo certo")