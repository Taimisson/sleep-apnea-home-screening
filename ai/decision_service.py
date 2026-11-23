# servico de decisao recebe o resumo da noite e sugere repetir / encaminhar / alta
# é só uma sugestao então quem decide é o enfermeiro (CFM 2.454/2026 e LGPD art. 20)

import json

# limites da regra (pre-especificada, sem ajuste nos dados)
MIN_VALID_HOURS = 4.0   # menos q isso de sinal bom -> repetir a noite
ODI3_CUTOFF = 15.0      # ODI3 >= 15/h -> encaminhar pra polissonografia

# versao da regra q vai junto na resposta pro FHIR
RULE_VERSION = "rule-odi3-15-v1"


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


# teste rapido python ai/decision_service.py
if __name__ == "__main__":
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

    print("tudo certo")