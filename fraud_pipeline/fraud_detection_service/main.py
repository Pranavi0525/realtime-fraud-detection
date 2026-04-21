"""
Fraud Detection Service — main.py
===================================
Consumes raw transactions from Kafka, runs 4 rule checks + IEEE ML model,
then passes ALL signals to the LLM agent for a reasoned recommendation.

Architecture:
  Kafka: raw-transactions
    → Rule engine (probe, velocity, location velocity, anomaly)
    → IEEE-CIS XGBoost ML scorer (trained on real transaction fraud data)
    → LLM Agent (Groq — synthesizes signals into recommendation)
    → Kafka: fraud-alerts
"""

import json
import time
import math
import redis
import logging
from datetime import datetime
from kafka import KafkaConsumer, KafkaProducer

# Local modules
from ieee_scorer import get_scorer
from llm_agent import get_recommendation, format_recommendation

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP = "localhost:9092"
INPUT_TOPIC     = "raw-transactions"
OUTPUT_TOPIC    = "fraud-alerts"
REDIS_HOST      = "localhost"
REDIS_PORT      = 6379

# Rule thresholds
PROBE_AMOUNT_THRESHOLD  = 10        # ₹ — transactions <= this are probes
PROBE_HIT_THRESHOLD     = 10_000   # ₹ — if probe seen, flag large hits
PROBE_WINDOW_SECS       = 300      # 5 minutes
VELOCITY_MAX_TXNS       = 5        # max transactions in velocity window
VELOCITY_WINDOW_SECS    = 60       # 1 minute
LOCATION_WINDOW_SECS    = 300      # 5 minutes
MAX_TRAVEL_SPEED_KMH    = 900      # Flag if implied speed > this

# City coordinates for impossible travel
CITY_COORDS = {
    "Mumbai":    (19.0760, 72.8777),
    "Delhi":     (28.7041, 77.1025),
    "Bangalore": (12.9716, 77.5946),
    "Chennai":   (13.0827, 80.2707),
    "Kolkata":   (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867),
    "Pune":      (18.5204, 73.8567),
    "Ahmedabad": (23.0225, 72.5714),
    "Jaipur":    (26.9124, 75.7873),
    "Surat":     (21.1702, 72.8311),
}


# ── Infrastructure connections ────────────────────────────────────────────────
def connect_redis():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    r.ping()
    log.info("✅ Redis connected")
    return r


def connect_kafka_consumer():
    consumer = KafkaConsumer(
        INPUT_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="latest",
        group_id="fraud-detection-service",
    )
    log.info(f"✅ Kafka consumer connected → {INPUT_TOPIC}")
    return consumer


def connect_kafka_producer():
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    log.info(f"✅ Kafka producer connected → {OUTPUT_TOPIC}")
    return producer


# ── Haversine distance ────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


# ── Rule checks ───────────────────────────────────────────────────────────────
def check_probe_attack(r: redis.Redis, txn: dict) -> tuple[bool, str]:
    """
    Detects card testing: small probe (≤₹10) followed by large hit (≥₹10K)
    within a 5-minute window.
    """
    card_id = txn["card_id"]
    amount  = float(txn["amount"])
    key     = f"probe:{card_id}"

    if amount <= PROBE_AMOUNT_THRESHOLD:
        r.setex(key, PROBE_WINDOW_SECS, str(amount))
        log.debug(f"Probe stored for {card_id}: ₹{amount}")
        return False, ""

    if amount >= PROBE_HIT_THRESHOLD:
        probe_val = r.get(key)
        if probe_val:
            signal = (
                f"PROBE_ATTACK: Card {card_id} — ₹{float(probe_val):.0f} probe detected "
                f"before ₹{amount:,.0f} transaction within {PROBE_WINDOW_SECS//60} minutes"
            )
            return True, signal

    return False, ""


def check_velocity(r: redis.Redis, txn: dict) -> tuple[bool, str]:
    """
    Detects rapid card usage: > 5 transactions in 60 seconds.
    """
    card_id = txn["card_id"]
    key     = f"velocity:{card_id}"

    count = r.incr(key)
    if count == 1:
        r.expire(key, VELOCITY_WINDOW_SECS)

    if count > VELOCITY_MAX_TXNS:
        signal = (
            f"VELOCITY_ABUSE: Card {card_id} — {count} transactions in "
            f"{VELOCITY_WINDOW_SECS}s window (max: {VELOCITY_MAX_TXNS})"
        )
        return True, signal

    return False, ""


def check_impossible_travel(r: redis.Redis, txn: dict) -> tuple[bool, str]:
    """
    Detects impossible location changes based on Haversine distance and time.
    """
    card_id   = txn["card_id"]
    city      = txn.get("city", "")
    timestamp = float(txn.get("timestamp", time.time()))
    key       = f"location:{card_id}"

    if city not in CITY_COORDS:
        return False, ""

    current_coords = CITY_COORDS[city]
    prev_data = r.get(key)

    # Store current location
    r.setex(key, LOCATION_WINDOW_SECS, json.dumps({
        "city": city,
        "lat": current_coords[0],
        "lon": current_coords[1],
        "ts": timestamp,
    }))

    if not prev_data:
        return False, ""

    prev = json.loads(prev_data)
    if prev["city"] == city:
        return False, ""

    distance_km = haversine_km(prev["lat"], prev["lon"], current_coords[0], current_coords[1])
    elapsed_hrs = max((timestamp - prev["ts"]) / 3600, 0.001)
    speed_kmh   = distance_km / elapsed_hrs

    if speed_kmh > MAX_TRAVEL_SPEED_KMH:
        signal = (
            f"IMPOSSIBLE_TRAVEL: Card {card_id} — {prev['city']} → {city} "
            f"({distance_km:.0f} km in {elapsed_hrs*60:.1f} mins = {speed_kmh:.0f} km/h)"
        )
        return True, signal

    return False, ""


# ── Context enrichment for LLM + IEEE ────────────────────────────────────────
def enrich_transaction(r: redis.Redis, txn: dict) -> dict:
    """
    Add context signals to the transaction dict that the IEEE scorer
    and LLM agent can use.
    """
    card_id   = txn["card_id"]
    timestamp = float(txn.get("timestamp", time.time()))
    dt        = datetime.fromtimestamp(timestamp)

    # Transaction count in last hour (from velocity key — approximate)
    vel_count = r.get(f"velocity:{card_id}")

    # Minutes since last transaction
    last_ts_key = f"last_ts:{card_id}"
    last_ts = r.get(last_ts_key)
    mins_since = (timestamp - float(last_ts)) / 60 if last_ts else -1
    r.setex(last_ts_key, 3600, str(timestamp))

    # Distance from last location
    last_loc = r.get(f"location:{card_id}")
    city = txn.get("city", "")
    distance_km = 0
    if last_loc and city in CITY_COORDS:
        prev = json.loads(last_loc)
        distance_km = haversine_km(
            prev["lat"], prev["lon"],
            CITY_COORDS[city][0], CITY_COORDS[city][1]
        )

    return {
        **txn,
        "txn_count_1h":        int(vel_count or 1),
        "mins_since_last_txn": round(mins_since, 1),
        "distance_km":         round(distance_km, 1),
        "hour_of_day":         dt.hour,
        "day_of_week":         dt.weekday(),
        "transaction_dt":      int(timestamp % 86400),  # Seconds within day
    }


# ── Main processing loop ──────────────────────────────────────────────────────
def process_transaction(txn_raw: dict, r: redis.Redis, scorer, producer: KafkaProducer):
    """
    Full fraud detection pipeline for a single transaction.
    1. Run rule checks
    2. Score with IEEE-CIS ML model
    3. LLM agent synthesizes recommendation
    4. Publish alert if any signal fired
    """
    txn_id  = txn_raw.get("transaction_id", "UNKNOWN")
    card_id = txn_raw.get("card_id", "UNKNOWN")
    amount  = float(txn_raw.get("amount", 0))

    # Enrich with context
    txn = enrich_transaction(r, txn_raw)

    # ── Rule checks ───────────────────────────────────────────────────────────
    fired_signals = []

    probe_fired, probe_signal = check_probe_attack(r, txn)
    if probe_fired:
        fired_signals.append(probe_signal)

    vel_fired, vel_signal = check_velocity(r, txn)
    if vel_fired:
        fired_signals.append(vel_signal)

    travel_fired, travel_signal = check_impossible_travel(r, txn)
    if travel_fired:
        fired_signals.append(travel_signal)

    # ── IEEE-CIS ML scoring ───────────────────────────────────────────────────
    ml_result = scorer.score(txn)

    if ml_result["is_fraud"]:
        ml_signal = (
            f"IEEE_ML_FRAUD_SCORE: {ml_result['fraud_probability']:.1f}% "
            f"(threshold: {ml_result['threshold']:.1f}%) | "
            + " | ".join(ml_result.get("top_signals", []))
        )
        fired_signals.append(ml_signal)

    # ── Skip if clean ─────────────────────────────────────────────────────────
    if not fired_signals:
        log.info(f"✅ CLEAN  {txn_id} | {card_id} | ₹{amount:,.0f}")
        return

    # ── LLM Agent recommendation ──────────────────────────────────────────────
    log.info(f"🚨 FRAUD SIGNALS on {txn_id} — calling LLM agent...")
    recommendation = get_recommendation(txn, fired_signals, ml_result)

    # ── Build alert payload ───────────────────────────────────────────────────
    severity = {
        "HIGH":   "HIGH",
        "MEDIUM": "MEDIUM",
        "LOW":    "LOW",
    }.get(recommendation.get("confidence", "MEDIUM"), "MEDIUM")

    alert = {
        "transaction_id":    txn_id,
        "card_id":           card_id,
        "amount":            amount,
        "merchant":          txn.get("merchant", ""),
        "city":              txn.get("city", ""),
        "timestamp":         txn.get("timestamp", time.time()),
        "fired_signals":     fired_signals,
        "signal_count":      len(fired_signals),
        "severity":          severity,
        "ml_score":          ml_result.get("fraud_probability", 0),
        "ml_available":      ml_result.get("available", False),
        "recommendation":    recommendation,
    }

    # Publish to Kafka
    producer.send(OUTPUT_TOPIC, value=alert)
    producer.flush()

    # Print formatted recommendation to terminal
    print(format_recommendation(recommendation))
    log.warning(f"🚨 ALERT published | {txn_id} | {len(fired_signals)} signals | "
                f"Action: {recommendation.get('recommended_action')}")


def main():
    log.info("Starting Fraud Detection Service")
    log.info("=" * 60)

    r        = connect_redis()
    scorer   = get_scorer()
    consumer = connect_kafka_consumer()
    producer = connect_kafka_producer()

    if scorer.is_available():
        log.info(f"🧠 IEEE-CIS model active | AUC-ROC: {scorer.metadata.get('auc_roc', '?')}")
    else:
        log.warning("⚠️  IEEE model not loaded — ML scoring disabled")
        log.warning("   Run: python scripts/train_ieee_model.py")

    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        log.info("🤖 LLM Agent: Groq API active (dynamic reasoning)")
    else:
        log.info("🤖 LLM Agent: Rule-based fallback (set GROQ_API_KEY for LLM reasoning)")

    log.info("=" * 60)
    log.info(f"Listening on Kafka topic: {INPUT_TOPIC}")

    for message in consumer:
        try:
            process_transaction(message.value, r, scorer, producer)
        except Exception as e:
            log.error(f"Error processing transaction: {e}", exc_info=True)


import os
if __name__ == "__main__":
    main()
