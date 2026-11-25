# servico de decisao recebe o resumo da noite e sugere repetir / encaminhar / alta
# é só uma sugestao então quem decide é o enfermeiro (CFM 2.454/2026 e LGPD art. 20)

import argparse
import json

# limites da regra (pre-especificada, sem ajuste nos dados)
MIN_VALID_HOURS = 4.0   # menos q isso de sinal bom -> repetir a noite
ODI3_CUTOFF = 15.0      # ODI3 >= 15/h -> encaminhar pra polissonografia

# versao da regra q vai junto na resposta pro FHIR
RULE_VERSION = "rule-odi3-15-v1"

# topicos mqtt (contrato 8.2 e 8.3)
SUMMARY_TOPIC = "home/+/oximeter/summary"
DECISION_TOPIC = "home/{patient_id}/oximeter/decision"
BROKER_PORT = 1883


# decide a partir do resumo da noite (dict no formato do night_summary)
def decide(summary):
    valid_h = summary.get("valid_hours") or 0.0
    odi3 = summary.get("odi3_per_h")

    # 1. pouco sinal bom: nunca encaminha automatico e pede pra repetir
    if valid_h < MIN_VALID_HOURS or odi3 is None:
        action = "repeat"
        reason = f"only {valid_h:.1f} h of valid signal (minimum {MIN_VALID_HOURS:.0f} h)"

    # 2. muitas dessaturacoes: sugere encaminhar
    elif odi3 >= ODI3_CUTOFF:
        action = "refer"
        reason = f"ODI3 {odi3:.1f}/h >= {ODI3_CUTOFF:.0f}/h with {valid_h:.1f} h of valid signal"

    # 3. senao: alta da triagem
    else:
        action = "discharge"
        reason = f"ODI3 {odi3:.1f}/h < {ODI3_CUTOFF:.0f}/h with {valid_h:.1f} h of valid signal"

    return {
        "patient_id": summary.get("patient_id"),
        "suggested_action": action,
        "reason": reason,
        "rule_odi3_per_h": odi3,
        "valid_hours": valid_h,
        "model_version": RULE_VERSION,
    }


# monta a mensagem de decisao: saida do decide + o resumo inteiro (contrato 8.3)
def build_decision_message(summary):
    out = decide(summary)
    out["summary"] = summary
    return out


# servico mqtt: escuta os resumos, decide e publica a decisao
def serve(broker="localhost"):
    import paho.mqtt.client as mqtt  # so precisa do paho quando roda o servico

    # inscreve toda vez q conecta (se o broker cair e voltar, inscreve de novo)
    def on_connect(client, userdata, flags, reason, props):
        if reason.is_failure:
            print(f"broker recusou a conexao: {reason}", flush=True)
            return
        client.subscribe(SUMMARY_TOPIC, qos=1)
        print(f"conectado em {broker}:{BROKER_PORT}, esperando resumos em {SUMMARY_TOPIC}", flush=True)

    # chegou um resumo: decide e publica
    def on_message(client, userdata, msg):
        try:
            summary = json.loads(msg.payload)
        except json.JSONDecodeError:
            print(f"resumo invalido em {msg.topic}, ignorado", flush=True)
            return

        # se o resumo n tiver patient_id, pega do topico (home/<id>/oximeter/summary)
        summary.setdefault("patient_id", msg.topic.split("/")[1])
        out = build_decision_message(summary)
        topic = DECISION_TOPIC.format(patient_id=out["patient_id"])
        client.publish(topic, json.dumps(out), qos=1)
        print(f"{out['patient_id']}: {out['suggested_action']} ({out['reason']}) -> {topic}", flush=True)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="decision-service")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker, BROKER_PORT)
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("parando o servico", flush=True)
    finally:
        client.disconnect()


# teste rapido sem broker
def self_test():
    # um resumo de mentira pra cada saida
    casos = [
        ({"patient_id": "t1", "valid_hours": 7.2, "odi3_per_h": 19.8}, "refer"),
        ({"patient_id": "t2", "valid_hours": 6.5, "odi3_per_h": 4.1}, "discharge"),
        ({"patient_id": "t3", "valid_hours": 2.5, "odi3_per_h": 30.0}, "repeat"),   # muito sinal ruim
        ({"patient_id": "t4", "valid_hours": 7.0, "odi3_per_h": 15.0}, "refer"),    # bem no corte
        ({"patient_id": "t5", "valid_hours": 0.0, "odi3_per_h": None}, "repeat"),   # sem sinal nenhum
    ]
    for summ, esperado in casos:
        out = decide(summ)
        assert out["suggested_action"] == esperado, f"{summ} deu {out['suggested_action']}, esperava {esperado}"
        print(json.dumps(out))

    # a mensagem do mqtt leva o resumo junto
    msg = build_decision_message(casos[0][0])
    assert msg["summary"] == casos[0][0] and msg["suggested_action"] == "refer"

    print("tudo certo")


# python ai/decision_service.py          -> teste rapido
# python ai/decision_service.py --serve  -> servico mqtt
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="servico de decisao (repeat / refer / discharge)")
    p.add_argument("--serve", action="store_true", help="roda o servico mqtt")
    p.add_argument("--broker", default="localhost")
    args = p.parse_args()

    if args.serve:
        serve(args.broker)
    else:
        self_test()
