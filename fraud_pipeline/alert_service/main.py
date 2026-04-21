"""
Alert Service — alert_service/main.py
=======================================
Consumes fraud alerts from Kafka: fraud-alerts
Displays structured alerts with LLM agent recommendations.

In production: card block API, SMS gateway, SIEM integration.
"""

import json
import logging
from datetime import datetime
from kafka import KafkaConsumer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC           = "fraud-alerts"

# Visual config
SEVERITY_COLORS = {
    "HIGH":   "🔴",
    "MEDIUM": "🟡",
    "LOW":    "🟢",
}

ACTION_LABELS = {
    "BLOCK_TRANSACTION":     "🚫 BLOCK TRANSACTION",
    "STEP_UP_AUTH":          "📱 STEP-UP AUTH (OTP)",
    "FLAG_FOR_REVIEW":       "🔍 FLAG FOR REVIEW",
    "ALLOW_WITH_MONITORING": "👁️  ALLOW WITH MONITORING",
}

ACTION_CONSEQUENCES = {
    "BLOCK_TRANSACTION":     "Card declined — transaction does not proceed",
    "STEP_UP_AUTH":          "OTP sent to cardholder's registered mobile number",
    "FLAG_FOR_REVIEW":       "Transaction held — sent to fraud analyst queue",
    "ALLOW_WITH_MONITORING": "Transaction approved — card added to watchlist",
}


def format_alert(alert: dict) -> str:
    """
    Full alert display with LLM recommendation block.
    """
    severity    = alert.get("severity", "MEDIUM")
    sev_icon    = SEVERITY_COLORS.get(severity, "⚪")
    rec         = alert.get("recommendation", {})
    action      = rec.get("recommended_action", "FLAG_FOR_REVIEW")
    confidence  = rec.get("confidence", "MEDIUM")
    source      = "Groq LLM" if rec.get("_source") == "groq_llm" else "Rule-based"

    ts = datetime.fromtimestamp(alert.get("timestamp", 0)).strftime("%Y-%m-%d %H:%M:%S")

    # Signals block
    signals_block = ""
    for i, signal in enumerate(alert.get("fired_signals", []), 1):
        # Split signal into type and description
        parts = signal.split(":", 1)
        sig_type = parts[0].strip()
        sig_desc = parts[1].strip() if len(parts) > 1 else signal
        signals_block += f"\n  [{i}] {sig_type}\n      {sig_desc}"

    lines = [
        "",
        "═" * 65,
        f"  {sev_icon}  FRAUD ALERT — {severity} SEVERITY",
        "═" * 65,
        f"  Transaction ID : {alert.get('transaction_id', 'UNKNOWN')}",
        f"  Card ID        : {alert.get('card_id', 'UNKNOWN')}",
        f"  Amount         : ₹{float(alert.get('amount', 0)):,.2f}",
        f"  Merchant       : {alert.get('merchant', 'UNKNOWN')}",
        f"  City           : {alert.get('city', 'UNKNOWN')}",
        f"  Timestamp      : {ts}",
        f"  Total Signals  : {alert.get('signal_count', 0)}",
        "─" * 65,
        "  FRAUD SIGNALS:",
        signals_block,
    ]

    # ML section
    if alert.get("ml_available"):
        ml_score = alert.get("ml_score", 0)
        lines += [
            "─" * 65,
            f"  IEEE-CIS ML Model: {ml_score:.1f}% fraud probability",
            f"  (Trained on real IEEE-CIS transaction fraud data)",
        ]

    # LLM recommendation block
    lines += [
        "─" * 65,
        f"  🤖 LLM AGENT RECOMMENDATION  [{source}]",
        "─" * 65,
        f"  Fraud Type  : {rec.get('fraud_type', 'Unknown')}",
        f"  Confidence  : {SEVERITY_COLORS.get(confidence, '⚪')} {confidence}",
        "",
        f"  Reasoning:",
        f"  {rec.get('reasoning', '')}",
        "",
        f"  ► ACTION: {ACTION_LABELS.get(action, action)}",
        f"    → {ACTION_CONSEQUENCES.get(action, '')}",
        f"    Why: {rec.get('action_rationale', '')}",
        "═" * 65,
        "",
    ]

    return "\n".join(lines)
from kafka import KafkaConsumer, TopicPartition

def main():
    consumer = KafkaConsumer(
    bootstrap_servers=KAFKA_BOOTSTRAP,
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
)

    tp = TopicPartition(TOPIC, 0)
    consumer.assign([tp])
    consumer.seek_to_end(tp)
    log.info(f"✅ Alert service listening on: {TOPIC}")
    log.info("Waiting for fraud alerts...")

    for message in consumer:
        try:
            alert = message.value
            print(format_alert(alert))
        except Exception as e:
            log.error(f"Error processing alert: {e}", exc_info=True)


if __name__ == "__main__":
    main()
