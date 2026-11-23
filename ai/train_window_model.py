# modelo de ML no nivel da janela (60 s): essa janela tem apneia/hipopneia ou n?
# validacao leave-one-subject-out (treina em 24 noites, testa na que sobrou, 25 vezes)
# no nivel da noite a decisao continua sendo a regra (Riley: precisaria de 379 noites, temos 25)

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# pra conseguir importar common/ e ai/ rodando daqui
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai.load_ucddb import list_subjects, load_spo2, load_events, load_labels
from common.features import to_1hz, basic_invalid_mask, night_windows

# a queda de SpO2 vem dps do evento respiratorio, entao estica o fim do evento
LAG_S = 20
# janela eh positiva se encostar pelo menos isso num evento
MIN_OVERLAP_S = 10
# janela com menos sinal bom q isso fica de fora (portao de qualidade)
MIN_VALID_FRAC = 0.8
# corte do rotulo da noite
AHI_CUTOFF = 15

FEATURES = ["spo2_mean", "spo2_min", "spo2_std", "spo2_range", "drop_max", "drop_mean",
            "secs_below3", "secs_below4", "area_below", "fall_rate_max", "rise_rate_max"]

RESULTS_DIR = ROOT / "docs" / "results"
MODELS_DIR = ROOT / "ai" / "models"


# monta a tabela de janelas de todas as noites, com o rotulo de cada janela
def build_dataset():
    labels = load_labels().set_index("subject")
    all_rows = []
    for sid in list_subjects():
        rec = load_spo2(sid)
        s1, out_of_range = to_1hz(rec["spo2"], rec["fs"])
        invalid = basic_invalid_mask(s1, out_of_range)
        win = night_windows(s1, invalid)

        # intervalo de cada evento anotado, esticado pelo atraso da SpO2
        ev = load_events(sid)
        starts = ev["onset_s"].to_numpy()
        ends = (ev["offset_s"] + LAG_S).to_numpy()

        # quanto cada janela encosta no evento q mais encosta nela
        overlap = np.zeros(len(win))
        for a, b in zip(starts, ends):
            ov = np.minimum(win["end_s"], b) - np.maximum(win["start_s"], a)
            overlap = np.maximum(overlap, ov.clip(lower=0))

        win["label"] = (overlap >= MIN_OVERLAP_S).astype(int)
        win["subject"] = sid
        win["ahi"] = labels.loc[sid, "ahi"]
        win["valid_h"] = (~invalid).sum() / 3600
        all_rows.append(win)

    data = pd.concat(all_rows, ignore_index=True)
    # tira as janelas com sinal ruim
    return data[data["valid_frac"] >= MIN_VALID_FRAC].reset_index(drop=True)


# os dois modelos: logistica (simples, explicavel) e gradient boosting
def make_models():
    return {
        "logistic": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000),
        ),
        "gboost": HistGradientBoostingClassifier(class_weight="balanced", max_iter=200,
                                                 learning_rate=0.05, random_state=0),
    }


# janelas positivas por hora de sinal valido (score da noite)
def positive_windows_per_hour(pred, valid_h):
    return pred.sum() / valid_h


# escolhe o corte da noite (janelas positivas/h) q melhor separa AHI >= 15, so nos dados de treino
def best_night_cutoff(scores, truth):
    best_c, best_j = None, -1
    for c in np.unique(scores):
        pred = scores >= c
        sens = (pred & truth).sum() / max(truth.sum(), 1)
        spec = (~pred & ~truth).sum() / max((~truth).sum(), 1)
        if sens + spec - 1 > best_j:
            best_c, best_j = c, sens + spec - 1
    return best_c


# score da noite pra cada sujeito de treino, sem vazamento (cross-validation interna por sujeito)
def train_night_scores(model, X, y, groups, valid_h):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = clone_fit(model, X[tr], y[tr])
        oof[te] = m.predict(X[te])
    df = pd.DataFrame({"g": groups, "pred": oof, "vh": valid_h})
    return df.groupby("g").apply(lambda d: positive_windows_per_hour(d.pred.to_numpy(), d.vh.iloc[0]))


# copia o modelo e treina
def clone_fit(model, X, y):
    from sklearn.base import clone
    m = clone(model)
    m.fit(X, y)
    return m


# leave-one-subject-out: pra cada noite, treina nas outras 24 e testa nela
def run_loso(data, name, model):
    X = data[FEATURES].to_numpy()
    y = data["label"].to_numpy()
    groups = data["subject"].to_numpy()
    valid_h = data["valid_h"].to_numpy()
    ahi15 = data.groupby("subject")["ahi"].first() >= AHI_CUTOFF

    fold_rows, night_rows = [], []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        sid = groups[te][0]
        m = clone_fit(model, X[tr], y[tr])
        prob = m.predict_proba(X[te])[:, 1]
        pred = (prob >= 0.5).astype(int)

        # AUC da janela so faz sentido se a noite tiver as duas classes
        auc = roc_auc_score(y[te], prob) if len(np.unique(y[te])) == 2 else np.nan
        fold_rows.append({"model": name, "subject": sid, "n_windows": len(te),
                          "pos_rate": y[te].mean(), "auc": auc})

        # nivel da noite: corte escolhido SO com as noites de treino
        tr_scores = train_night_scores(model, X[tr], y[tr], groups[tr], valid_h[tr])
        cutoff = best_night_cutoff(tr_scores.to_numpy(), ahi15.loc[tr_scores.index].to_numpy())
        score = positive_windows_per_hour(pred, valid_h[te][0])
        night_rows.append({"model": name, "subject": sid, "ahi": data.loc[te, "ahi"].iloc[0],
                           "ml_pos_windows_per_h": round(score, 1), "night_cutoff": round(cutoff, 1),
                           "true_osa": bool(ahi15.loc[sid]), "pred_osa": bool(score >= cutoff)})

    return pd.DataFrame(fold_rows), pd.DataFrame(night_rows)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("montando as janelas das 25 noites...")
    data = build_dataset()
    print(f"{len(data)} janelas validas, {data.label.mean():.1%} positivas")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    all_folds, all_nights = [], []
    for name, model in make_models().items():
        print(f"\nLOSO com {name} (25 dobras)...")
        folds, nights = run_loso(data, name, model)
        all_folds.append(folds)
        all_nights.append(nights)

        aucs = folds["auc"].dropna()
        tp = (nights.true_osa & nights.pred_osa).sum(); fn = (nights.true_osa & ~nights.pred_osa).sum()
        tn = (~nights.true_osa & ~nights.pred_osa).sum(); fp = (~nights.true_osa & nights.pred_osa).sum()
        print(f"  AUC da janela: media {aucs.mean():.3f} +- {aucs.std():.3f} (min {aucs.min():.3f}, max {aucs.max():.3f}, {len(aucs)} noites)")
        print(f"  noite (AHI >= 15): sens {tp}/{tp + fn}, espec {tn}/{tn + fp}, acertos {tp + tn}/25")

        # modelo final treinado com as 25 noites (pra demo)
        final = clone_fit(model, data[FEATURES].to_numpy(), data["label"].to_numpy())
        joblib.dump({"model": final, "features": FEATURES}, MODELS_DIR / f"window_{name}.joblib")

    pd.concat(all_folds).to_csv(RESULTS_DIR / "window_model_folds.csv", index=False)
    pd.concat(all_nights).to_csv(RESULTS_DIR / "window_model_nights.csv", index=False)
    print("\nregra (baseline) pra comparar: sens 13/14, espec 9/11, acertos 22/25")
    print("salvo em docs/results/window_model_*.csv e ai/models/")