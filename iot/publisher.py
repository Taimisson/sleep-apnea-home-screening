# publicador do replay: toca uma noite do UCDDB no ritmo do oximetro e manda por MQTT
# so o sensor eh simulado, o resto da pipeline eh real (contrato 8.1)

import argparse
import json
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import paho.mqtt.client as mqtt

# pra conseguir importar ai/ rodando daqui
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ai.load_ucddb import load_spo2

# data da demo (dia em que a gravacao comeca) e fuso de brasilia
DEMO_DATE = "2026-12-07"
TZ = timezone(timedelta(hours=-3))

BROKER_PORT = 1883
TIMEOUT_S = 5


# horario original da medida: data da demo + inicio da gravacao + seq segundos
# o timedelta ja resolve a virada da meia noite
def sample_ts(start, seq, demo_date=DEMO_DATE):
    t0 = datetime.strptime(f"{demo_date} {start}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ)
    return (t0 + timedelta(seconds=seq)).isoformat()


# primeiro e ultimo seq do trecho (seq continua contando desde o inicio da gravacao)
def seq_range(n_sec, start_min=0, duration_min=None):
    first = round(start_min * 60)
    last = n_sec - 1 if duration_min is None else min(n_sec, first + round(duration_min * 60)) - 1
    if first < 0 or first > last:
        raise ValueError(f"trecho fora da gravacao (inicio {start_min} min, gravacao de {n_sec / 60:.0f} min)")
    return first, last


# vetor numpy pra lista do json: NaN vira null e nada eh arredondado
def to_list(chunk):
    return [None if np.isnan(x) else float(x) for x in chunk]


# mensagem de 1 segundo de sinal (formato 8.1)
def build_message(rec, sid, seq):
    fs = int(rec["fs"])
    a, b = seq * fs, (seq + 1) * fs
    return {
        "device_id": f"oximeter-ucddb{sid}",
        "patient_id": f"ucddb{sid}",
        "seq": seq,
        "ts": sample_ts(rec["start"], seq),
        "fs_hz": fs,
        "spo2": to_list(rec["spo2"][a:b]),
        "hr": to_list(rec["pulse"][a:b]),
    }


# segundos pra "hh:mm" (log)
def fmt_hhmm(seconds):
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}"


# conecta e espera o broker aceitar (connack) antes de publicar
def connect(broker, client_id):
    connected = threading.Event()

    def on_connect(client, userdata, flags, reason, props):
        if not reason.is_failure:
            connected.set()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    client.on_connect = on_connect
    client.connect(broker, BROKER_PORT)
    client.loop_start()
    if not connected.wait(TIMEOUT_S):
        client.loop_stop()
        raise ConnectionError(f"broker {broker}:{BROKER_PORT} nao aceitou a conexao")
    return client


# toca o trecho: 1 mensagem por segundo de sinal, acelerado por speed
def replay(sid, speed, start_min, duration_min, broker):
    rec = load_spo2(sid, with_pulse=True)
    fs = int(rec["fs"])
    n_sec = len(rec["spo2"]) // fs  # mesmo corte do to_1hz (segundo incompleto no fim fica de fora)
    first, last = seq_range(n_sec, start_min, duration_min)

    patient_id = f"ucddb{sid}"
    raw_topic = f"home/{patient_id}/oximeter/raw"
    control_topic = f"home/{patient_id}/oximeter/control"
    total = fmt_hhmm(n_sec)
    n_msgs = last - first + 1
    print(f"tocando {patient_id}: seq {first} a {last} ({n_msgs} s de sinal) a {speed:g}x, previsto {n_msgs / speed:.0f} s", flush=True)

    client = connect(broker, f"publisher-{patient_id}")
    failed = 0
    try:
        t0 = time.monotonic()
        for i, seq in enumerate(range(first, last + 1)):
            # horario alvo deste envio: dorme so o que falta, entao o atraso nao acumula
            delay = t0 + i / speed - time.monotonic()
            if delay > 0:
                time.sleep(delay)

            payload = json.dumps(build_message(rec, sid, seq), allow_nan=False)
            # qos 0: se perder, o edge marca o segundo como faltando
            if client.publish(raw_topic, payload, qos=0).rc != mqtt.MQTT_ERR_SUCCESS:
                failed += 1

            if seq % 60 == 0:
                print(f"enviado {fmt_hhmm(seq)} de {total} (seq {seq})", flush=True)

        # fim da noite com qos 1 e espera o broker confirmar
        end = {"event": "end_of_night", "last_seq": last}
        info = client.publish(control_topic, json.dumps(end), qos=1)
        info.wait_for_publish(TIMEOUT_S)
        if not info.is_published():
            raise RuntimeError("broker nao confirmou o end_of_night")
        print(f"fim: {n_msgs} mensagens em {time.monotonic() - t0:.1f} s, {failed} falhas de envio, end_of_night (last_seq {last})", flush=True)
    finally:
        client.disconnect()
        client.loop_stop()


# teste rapido sem broker: python iot/publisher.py --self-test
def self_test():
    # exemplo do contrato 8.1: noite 002 comeca 00:11:04, seq 1532 -> 00:36:36
    assert sample_ts("00:11:04", 1532) == "2026-12-07T00:36:36-03:00"
    # virada da meia noite
    assert sample_ts("23:59:50", 15) == "2026-12-08T00:00:05-03:00"

    # recorte: noite inteira, trecho do meio, trecho que passa do fim, trecho fora
    assert seq_range(22470) == (0, 22469)
    assert seq_range(22470, 60, 2) == (3600, 3719)
    assert seq_range(22470, 370, 30) == (22200, 22469)
    try:
        seq_range(22470, 400)
        raise AssertionError("trecho fora da gravacao devia dar erro")
    except ValueError:
        pass

    # mensagem com sinal sintetico: campos na ordem do contrato e NaN vira null
    fake = {"spo2": np.r_[np.full(15, 95.5), np.nan], "pulse": np.full(16, 62.0), "fs": 8.0, "start": "00:11:04"}
    msg = build_message(fake, "002", 1)
    assert list(msg) == ["device_id", "patient_id", "seq", "ts", "fs_hz", "spo2", "hr"]
    assert msg["spo2"][-1] is None and msg["spo2"][0] == 95.5 and len(msg["hr"]) == 8
    json.dumps(msg, allow_nan=False)

    # formato do log
    assert fmt_hhmm(4320) == "01:12" and fmt_hhmm(22470) == "06:14"

    # noite real 002: primeiro segundo, sem arredondar
    rec = load_spo2("002", with_pulse=True)
    m = build_message(rec, "002", 0)
    assert m["ts"] == "2026-12-07T00:11:04-03:00"
    assert m["spo2"] == rec["spo2"][:8].tolist() and m["hr"] == rec["pulse"][:8].tolist()

    print("tudo certo")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="replay de uma noite do UCDDB via MQTT (oximetro simulado)")
    p.add_argument("--subject", default="002", help="id do UCDDB, ex.: 002")
    p.add_argument("--speed", type=float, default=120, help="aceleracao (120 = 1 h de sono em 30 s)")
    p.add_argument("--start-min", type=float, default=0, help="minuto da gravacao onde o trecho comeca")
    p.add_argument("--duration-min", type=float, default=None, help="minutos a tocar (padrao: ate o fim)")
    p.add_argument("--broker", default="localhost")
    p.add_argument("--self-test", action="store_true", help="roda o teste rapido sem broker")
    args = p.parse_args()

    if args.self_test:
        self_test()
    elif args.speed <= 0:
        p.error("--speed tem que ser maior que 0")
    else:
        replay(args.subject, args.speed, args.start_min, args.duration_min, args.broker)