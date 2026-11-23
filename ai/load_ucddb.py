# leitura dos arquivos do UCDDB, sinal de SpO2, etc usado pela IA e pelo publicador do IOT

from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib


# pasta raiz do repo
ROOT = Path(__file__).resolve().parent.parent

# pasta com arquivos do UCDDB
DATA_DIR = ROOT / "data" / "raw" / "ucddb-1.0.0" / "files"

# tipos de evento q contam como apneia/hipopneia no rotulo
LABEL_EVENT_TYPES = ["APNEA-O", "APNEA-C", "APNEA-M", "HYP-O", "HYP-C", "HYP-M"]

# colunas do _respevt.txt e a pos de cada uma na linha
EVENT_COLS = ["time", "type", "pb_cs", "duration_s", "spo2_low", "spo2_drop_pct", "snore", "arousal"]
EVENT_COLSPECS = [(0, 8), (10, 18), (18, 24), (24, 33), (33, 41), (41, 49), (49, 55), (55, 61)]

# segundos de um dia, pra virada da meia noite
SECONDS_PER_DAY = 24 * 3600

# caminho de um arquivo do sujeito q dá erro se n existir
def _subject_file(sid, suffix):
    path = DATA_DIR / f"ucddb{sid}{suffix}"

    if not path.exists():
        raise FileNotFoundError(f"ARQUIVO NÃO ENCONTRADO: {path}")
    
    return path

# ids dos sujeitos
def list_subjects():
    lines = (DATA_DIR / "RECORDS").read_text().splitlines()

    return [line.strip()[5:8] for line in lines if line.strip().endswith(".rec")]

# "hh:mm:ss" pra segundos
def clock_to_seconds(hhmmss):
    h, m, s = map(int, hhmmss.split(":"))
    
    return h * 3600 + m * 60 + s

# só o horario de inicio da gravacao sem ler o sinal
def load_start_time(sid):
    f = pyedflib.EdfReader(str(_subject_file(sid, ".rec")))

    try:
        return f.getStartdatetime().strftime("%H:%M:%S")
    finally:
        f.close()

# sinal de SpO2 (e pulso se pedir), frequencia e horario de inicio
def load_spo2(sid, with_pulse=False):
    f = pyedflib.EdfReader(str(_subject_file(sid, ".rec")))

    try:
        labels = f.getSignalLabels()

        # pega o canal pelo nome
        if "SpO2" not in labels:
            raise ValueError(f"Canal SpO2 não encontrado em ucddb{sid}")
        i_spo2 = labels.index("SpO2")

        # a data do arquivo eh falsa e só o horario vale
        rec = {
            "spo2": f.readSignal(i_spo2),
            "fs": f.getSampleFrequency(i_spo2),
            "start": f.getStartdatetime().strftime("%H:%M:%S"),
        }

        if with_pulse:
            if "Pulse" not in labels:
                raise ValueError(f"Canal Pulse não encontrado em ucddb{sid}")
            rec["pulse"] = f.readSignal(labels.index("Pulse"))

        return rec
    finally:
        f.close()


# eventos do sujeito com o tempo em segundos desde o inicio
def load_events(sid, only_label_types=True):

    # le por largura fixa pulando as 3 linhas de cabecalho
    ev = pd.read_fwf(
        _subject_file(sid, "_respevt.txt"),
        colspecs=EVENT_COLSPECS,
        names=EVENT_COLS,
        skiprows=3,
        encoding="latin-1",
        dtype=str,
    )

    # so as linhas q comecam com hh:mm:ss
    ev = ev[ev["time"].str.match(r"^\d\d:\d\d:\d\d$", na=False)].copy()

    # colunas numericas e campo vazio vira NaN
    for col in ["duration_s", "spo2_low", "spo2_drop_pct"]:
        ev[col] = pd.to_numeric(ev[col])

    # "+" vira True e "-" vira False
    for col in ["snore", "arousal"]:
        ev[col] = ev[col] == "+"

    # mod 24h resolve os eventos dps da meia noite
    t0 = clock_to_seconds(load_start_time(sid))
    ev["onset_s"] = (ev["time"].map(clock_to_seconds) - t0) % SECONDS_PER_DAY
    ev["offset_s"] = ev["onset_s"] + ev["duration_s"]

    # apenas apneias e hipopneias
    if only_label_types:
        ev = ev[ev["type"].isin(LABEL_EVENT_TYPES)]

    ev.insert(0, "subject", sid)
    return ev.reset_index(drop=True)


# AHI e rotulo (AHI >= 15) de cada sujeito
def load_labels():
    details = pd.read_excel(DATA_DIR / "SubjectDetails.xls")

    labels = pd.DataFrame({
        "subject": details["Study Number"].str[-3:],  # "UCDDB002" -> "002"
        "ahi": details["PSG AHI"],
        "gender": details["Gender"],
        "age": details["Age"],
        "bmi": details["BMI"],
    })
    labels["osa_ahi15"] = labels["ahi"] >= 15
    return labels


# teste rapido: python ai/load_ucddb.py
if __name__ == "__main__":
    subjects = list_subjects()
    assert len(subjects) == 25, f"esperava 25 sujeitos, veio {len(subjects)}"

    rec = load_spo2("002", with_pulse=True)
    assert len(rec["spo2"]) == 179760, f"esperava 179760 amostras, veio {len(rec['spo2'])}"
    assert rec["fs"] == 8.0, f"esperava 8 Hz, veio {rec['fs']}"
    assert rec["start"] == "00:11:04", f"esperava inicio 00:11:04, veio {rec['start']}"
    assert len(rec["pulse"]) == len(rec["spo2"])

    all_events = load_events("002", only_label_types=False)
    assert len(all_events) == 124, f"esperava 124 eventos, veio {len(all_events)}"

    label_events = load_events("002")
    assert label_events["onset_s"].between(0, len(rec["spo2"]) / rec["fs"]).all(), "evento fora da gravacao"

    labels = load_labels()
    assert labels["osa_ahi15"].sum() == 14, f"esperava 14 com AHI >= 15, veio {labels['osa_ahi15'].sum()}"
    assert set(labels["subject"]) == set(subjects)

    print("sujeitos:", len(subjects))
    print("noite 002:", len(rec["spo2"]), "amostras a", rec["fs"], "Hz, inicio", rec["start"])
    print("eventos 002:", len(all_events), "no total,", len(label_events), "apneias/hipopneias")
    print("AHI >= 15:", labels["osa_ahi15"].sum(), "de", len(labels))
    print("tudo certo")