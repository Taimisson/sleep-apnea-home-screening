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
EVENT_COLS = []
EVENT_COLSPECS = []


def list_subjects():
    pass

def clock_to_seconds(): 
    pass

def load_spo2(sid, with_pulse=False):
    pass

def load_events(sid, only_label_types=True):
    pass


def load_labels():
    pass


if __name__ == "__main__":
    pass