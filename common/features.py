# deteccao de dessaturacao e resumo da noite
# modulo pra uso no treino/avaliacao (ai/) e no edge (iot/) pra dar os mesmos numeros

import numpy as np
import pandas as pd


# parametros padrao (os mesmos do mid-term)
DROP_PCT = 3            # queda minima em pontos de SpO2
MIN_DUR_S = 10          # duracao minima da queda
BASELINE_S = 60         # janela da linha de base (pico dos ultimos 60 s)
MIN_BASELINE_S = 30     # minimo de segundos validos pra ter linha de base

# limites do filtro basico de qualidade
MIN_VALID_SPO2 = 50     # abaixo disso eh sensor solto
MAX_VALID_SPO2 = 101    # tem valor de 100.2 no dataset, entao tem folga
MAX_JUMP = 4            # salto maximo entre um segundo e o proximo


# sinal de fs Hz pra 1 Hz (media de cada segundo) e tbm devolve quais segundos tem alguma amostra fora da faixa
def to_1hz(spo2, fs):
    fs = int(fs)
    n = len(spo2) // fs
    blocks = np.asarray(spo2[: n * fs], dtype=float).reshape(n, fs)
    out_of_range = ((blocks < MIN_VALID_SPO2) | (blocks > MAX_VALID_SPO2)).any(axis=1)
    return blocks.mean(axis=1), out_of_range


# filtro basico de qualidade, por segundo (True = invalido)
# o portao de qualidade do edge (iot/quality_gate.py) pode trocar ou completar isso
def basic_invalid_mask(spo2_1hz, out_of_range):
    jump = np.r_[0, np.abs(np.diff(spo2_1hz))] > MAX_JUMP
    return out_of_range | jump


# acha as dessaturacoes num sinal de 1 Hz (segundo invalido = NaN)
# a regra é SpO2 <= pico dos 60 s anteriores - drop_pct, por pelo menos min_dur_s seguidos
def detect_desaturations(spo2_1hz, drop_pct=DROP_PCT, min_dur_s=MIN_DUR_S,
                         baseline_s=BASELINE_S, min_baseline_s=MIN_BASELINE_S):
    v = pd.Series(np.asarray(spo2_1hz, dtype=float))
    
    # linha de base: maior valor dos 60 s anteriores (sem contar o segundo atual)
    baseline = v.shift(1).rolling(baseline_s, min_periods=min_baseline_s).max()

    # segundos q estao abaixo da linha de base - drop (NaN conta como False)
    below = (v <= baseline - drop_pct).fillna(False).to_numpy()

    events = []
    start = None
    # o False no final fecha um evento q ainda estiver aberto no fim do sinal
    for i, b in enumerate(np.append(below, False)):
        if b and start is None:
            start = i
        elif not b and start is not None:
            if i - start >= min_dur_s:
                seg = v.iloc[start:i]
                events.append({
                    "start_s": start,
                    "end_s": i,
                    "duration_s": i - start,
                    "baseline": float(baseline.iloc[start]),
                    "nadir": float(seg.min()),
                    "depth": float(baseline.iloc[start] - seg.min()),
                })
            start = None

    return pd.DataFrame(events, columns=["start_s", "end_s", "duration_s", "baseline", "nadir", "depth"])


# resumo da noite: so numeros, nunca o sinal bruto (LGPD)
def night_summary(spo2_1hz, invalid):
    invalid = np.asarray(invalid, dtype=bool)

    # sinal com NaN onde eh invalido
    v = np.where(invalid, np.nan, np.asarray(spo2_1hz, dtype=float))

    recorded_h = len(v) / 3600
    valid_h = (~invalid).sum() / 3600

    ev3 = detect_desaturations(v, drop_pct=3)
    ev4 = detect_desaturations(v, drop_pct=4)

    # eventos por hora de gravacao (pro grafico do dashboard)
    n_hours = int(np.ceil(recorded_h))
    per_hour = np.bincount((ev3["start_s"] // 3600).astype(int), minlength=n_hours).tolist() if len(ev3) else [0] * n_hours

    # ODI usa horas de sinal VALIDO no denominador
    return {
        "recorded_hours": round(recorded_h, 2),
        "valid_hours": round(valid_h, 2),
        "valid_fraction": round(valid_h / recorded_h, 3) if recorded_h else 0.0,
        "n_desat_3pct": int(len(ev3)),
        "odi3_per_h": round(len(ev3) / valid_h, 1) if valid_h else None,
        "odi4_per_h": round(len(ev4) / valid_h, 1) if valid_h else None,
        "mean_spo2": round(float(np.nanmean(v)), 1),
        "nadir_spo2": round(float(np.nanmin(v)), 1),
        "t90_min": round(float((v < 90).sum()) / 60, 1),   # minutos abaixo de 90%
        "events_per_hour": per_hour,
        "criteria": f"desat >= {DROP_PCT}% vs max of previous {BASELINE_S} s, duration >= {MIN_DUR_S} s",
    }

# tamanho da janela e passo sem sobreposicao
WIN_S = 60
WIN_STEP_S = 60


# maior valor dos 60 s anteriores
def baseline_series(spo2_1hz, baseline_s=BASELINE_S, min_baseline_s=MIN_BASELINE_S):
    v = pd.Series(np.asarray(spo2_1hz, dtype=float))
    return v.shift(1).rolling(baseline_s, min_periods=min_baseline_s).max().to_numpy()


# features de UMA janela de 1 Hz (v = SpO2 com NaN onde eh invalido, base = linha de base)
def window_features(v, base):
    v = np.asarray(v, dtype=float)
    base = np.asarray(base, dtype=float)
    valid = ~np.isnan(v)

    feats = {"valid_frac": float(valid.mean())}

    # janela sem sinal: tudo NaN (o modelo trata)
    if valid.sum() < 2:
        for k in ["spo2_mean", "spo2_min", "spo2_std", "spo2_range", "drop_max", "drop_mean",
                  "secs_below3", "secs_below4", "area_below", "fall_rate_max", "rise_rate_max"]:
            feats[k] = np.nan
        return feats

    # quanto o sinal ta abaixo da linha de base em cada segundo
    drop = base - v

    # variacao em 5 s (pra ver se cai ou sobe rapido)
    diff5 = v[5:] - v[:-5]

    feats.update({
        "spo2_mean": float(np.nanmean(v)),
        "spo2_min": float(np.nanmin(v)),
        "spo2_std": float(np.nanstd(v)),
        "spo2_range": float(np.nanmax(v) - np.nanmin(v)),
        "drop_max": float(np.nanmax(drop)) if np.any(~np.isnan(drop)) else np.nan,
        "drop_mean": float(np.nanmean(drop)) if np.any(~np.isnan(drop)) else np.nan,
        "secs_below3": int(np.sum(drop >= 3)),
        "secs_below4": int(np.sum(drop >= 4)),
        "area_below": float(np.nansum(np.clip(drop, 0, None))),
        "fall_rate_max": float(np.nanmax(-diff5) / 5) if np.any(~np.isnan(diff5)) else np.nan,
        "rise_rate_max": float(np.nanmax(diff5) / 5) if np.any(~np.isnan(diff5)) else np.nan,
    })
    return feats


# corta a noite em janelas e calcula as features de cada uma
def night_windows(spo2_1hz, invalid, win_s=WIN_S, step_s=WIN_STEP_S):
    v = np.where(np.asarray(invalid, dtype=bool), np.nan, np.asarray(spo2_1hz, dtype=float))
    base = baseline_series(v)

    rows = []
    for start in range(0, len(v) - win_s + 1, step_s):
        feats = window_features(v[start:start + win_s], base[start:start + win_s])
        rows.append({"start_s": start, "end_s": start + win_s, **feats})
    return pd.DataFrame(rows)

# atalho: do sinal cru (fs Hz) direto pro resumo, com o filtro basico
def summarize_night(spo2, fs):
    s1, out_of_range = to_1hz(spo2, fs)
    invalid = basic_invalid_mask(s1, out_of_range)
    return night_summary(s1, invalid)


# teste rapido: python common/features.py
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # pra conseguir importar o ai/ rodando daqui
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ai.load_ucddb import load_spo2

    # teste 1: sinal sintetico com uma queda de 5 pontos por 20 s -> 1 evento
    fake = np.full(300, 96.0)
    fake[150:170] = 91.0
    ev = detect_desaturations(fake)
    assert len(ev) == 1, f"esperava 1 evento no sinal sintetico, veio {len(ev)}"
    assert ev.iloc[0]["duration_s"] == 20

    # teste 2: queda de 5 pontos mas so 5 s -> nenhum evento
    fake2 = np.full(300, 96.0)
    fake2[150:155] = 91.0
    assert len(detect_desaturations(fake2)) == 0, "queda curta n devia virar evento"

    # teste 3: noite 002 tem q bater com o mid-term (ODI3 = 19.9/h)
    rec = load_spo2("002")
    summ = summarize_night(rec["spo2"], rec["fs"])
    assert summ["odi3_per_h"] == 19.9, f"esperava ODI3 19.9 na 002, veio {summ['odi3_per_h']}"

    for k, val in summ.items():
        print(f"{k}: {val}")
    print("tudo certo")