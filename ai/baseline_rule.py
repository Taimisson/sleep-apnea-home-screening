# baseline sem ML com regra ODI3 >= 15/h nas 25 noites, comparada com o AHI da polissonografia
# responde "o q o modelo adiciona sobre a regra?"

import sys
from math import sqrt
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # salva a figura sem abrir janela
import matplotlib.pyplot as plt

# pra conseguir importar common/ e ai/ rodando daqui
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai.load_ucddb import list_subjects, load_spo2, load_labels
from common.features import summarize_night

# corte da regra e do rotulo
ODI3_CUTOFF = 15
AHI_CUTOFF = 15

# onde salvar a tabela e a figura
RESULTS_DIR = ROOT / "docs" / "results"
FIGURES_DIR = ROOT / "docs" / "figures"


# intervalo de confianca de Wilson pra uma proporcao (bom pra n pequeno)
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centro = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    margem = z * sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return (centro - margem, centro + margem)


# roda a regra em todas as noites e devolve uma tabela
def run_all_nights():
    labels = load_labels().set_index("subject")
    rows = []
    for sid in list_subjects():
        rec = load_spo2(sid)
        summ = summarize_night(rec["spo2"], rec["fs"])
        rows.append({
            "subject": sid,
            "ahi": labels.loc[sid, "ahi"],
            "odi3_per_h": summ["odi3_per_h"],
            "odi4_per_h": summ["odi4_per_h"],
            "valid_hours": summ["valid_hours"],
            "valid_fraction": summ["valid_fraction"],
            "t90_min": summ["t90_min"],
            "nadir_spo2": summ["nadir_spo2"],
            "true_osa": bool(labels.loc[sid, "osa_ahi15"]),
            "pred_osa": summ["odi3_per_h"] >= ODI3_CUTOFF,
        })
    return pd.DataFrame(rows)


# matriz de confusao e metricas, com intervalo de confianca
def metrics(df):
    tp = int((df.true_osa & df.pred_osa).sum())
    fn = int((df.true_osa & ~df.pred_osa).sum())
    fp = int((~df.true_osa & df.pred_osa).sum())
    tn = int((~df.true_osa & ~df.pred_osa).sum())

    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "sensitivity": sens, "sensitivity_ci": wilson_ci(tp, tp + fn),
        "specificity": spec, "specificity_ci": wilson_ci(tn, tn + fp),
        "accuracy": (tp + tn) / len(df),
        # baseline trivial: chutar sempre a classe mais comum
        "majority_accuracy": max(df.true_osa.mean(), 1 - df.true_osa.mean()),
    }


# grafico ODI3 x AHI com as linhas de corte
def plot_odi_vs_ahi(df, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    acertou = df.true_osa == df.pred_osa
    ax.scatter(df.ahi[acertou], df.odi3_per_h[acertou], label="correct", color="tab:blue")
    ax.scatter(df.ahi[~acertou], df.odi3_per_h[~acertou], label="wrong", color="tab:red", marker="x", s=70)

    # nome do sujeito do lado de cada erro
    for _, r in df[~acertou].iterrows():
        ax.annotate(r.subject, (r.ahi, r.odi3_per_h), textcoords="offset points", xytext=(5, 5))

    ax.axvline(AHI_CUTOFF, color="gray", ls="--", lw=1)
    ax.axhline(ODI3_CUTOFF, color="gray", ls="--", lw=1)
    lim = max(df.ahi.max(), df.odi3_per_h.max()) + 5
    ax.plot([0, lim], [0, lim], color="lightgray", lw=1)  # linha ODI = AHI
    ax.set_xlabel("PSG AHI (events/h)")
    ax.set_ylabel("ODI3 (events/h)")
    ax.set_title("Rule ODI3 >= 15 vs PSG AHI >= 15 (25 nights)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    df = run_all_nights()
    m = metrics(df)

    # salva a tabela e a figura
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "baseline_rule_nights.csv", index=False)
    plot_odi_vs_ahi(df, FIGURES_DIR / "baseline_odi3_vs_ahi.png")

    print(df.to_string(index=False))
    print()
    print("matriz de confusao (linha = real, coluna = regra)")
    print(f"            pred OSA  pred nao")
    print(f"  real OSA    {m['tp']:>3}       {m['fn']:>3}")
    print(f"  real nao    {m['fp']:>3}       {m['tn']:>3}")
    print()
    lo, hi = m["sensitivity_ci"]
    print(f"sensibilidade: {m['tp']}/{m['tp'] + m['fn']} = {m['sensitivity']:.2f} (IC95% {lo:.2f}-{hi:.2f})")
    lo, hi = m["specificity_ci"]
    print(f"especificidade: {m['tn']}/{m['tn'] + m['fp']} = {m['specificity']:.2f} (IC95% {lo:.2f}-{hi:.2f})")
    print(f"acuracia: {m['accuracy']:.2f} | chutando sempre a classe mais comum: {m['majority_accuracy']:.2f}")
    print()
    print("erros:", df.loc[df.true_osa != df.pred_osa, ["subject", "ahi", "odi3_per_h"]].to_dict("records"))
    print("salvo em docs/results/baseline_rule_nights.csv e docs/figures/baseline_odi3_vs_ahi.png")