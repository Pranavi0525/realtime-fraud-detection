"""
Transaction Simulator — transaction_service/main.py
=====================================================
Simulates card transactions and publishes to Kafka: raw-transactions

Three simulator modes:
  1. Normal transactions    (background noise)
  2. Attack scenarios       (probe, velocity, impossible travel)
  3. IEEE-CIS replay        (if dataset available — uses real transaction amounts/patterns)

The simulator now adds transaction_dt and behavioral context
that the IEEE scorer and LLM agent can use.
"""

import json
import time
import random
import hashlib
import logging
from datetime import datetime
from kafka import KafkaProducer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC           = "raw-transactions"

# Cards and their typical behavior profiles
CARD_PROFILES = {
    "CARD1111": {"city": "Mumbai",    "typical_amount": (200, 5000),  "email": "gmail.com"},
    "CARD2222": {"city": "Delhi",     "typical_amount": (100, 3000),  "email": "yahoo.com"},
    "CARD3333": {"city": "Bangalore", "typical_amount": (500, 8000),  "email": "hotmail.com"},
    "CARD4444": {"city": "Chennai",   "typical_amount": (150, 2000),  "email": "gmail.com"},
    "CARD5555": {"city": "Mumbai",    "typical_amount": (200, 4000),  "email": "outlook.com"},  # Attack card
    "CARD6666": {"city": "Delhi",     "typical_amount": (300, 6000),  "email": "yahoo.com"},    # Attack card
    "CARD7777": {"city": "Kolkata",   "typical_amount": (100, 1500),  "email": "gmail.com"},    # Attack card
}

MERCHANTS = [
    "BigBasket", "Amazon.in", "Flipkart", "Swiggy", "Zomato",
    "MakeMyTrip", "BookMyShow", "Myntra", "Nykaa", "PhonePe",
    "OnlineStore", "ElectronicsHub", "FashionBazaar",
]


def make_txn_id():
    return "TXN" + hashlib.md5(str(random.random()).encode()).hexdigest()[:6].upper()


def connect_producer():
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    log.info(f"✅ Kafka producer connected → {TOPIC}")
    return producer


def publish(producer: KafkaProducer, txn: dict):
    producer.send(TOPIC, value=txn)
    producer.flush()
    log.info(f"📤 TXN {txn['transaction_id']} | {txn['card_id']} | ₹{txn['amount']:,.0f} | {txn['city']}")


# ── Normal transaction ────────────────────────────────────────────────────────
def normal_transaction(producer, card_id=None):
    card_id = card_id or random.choice(list(CARD_PROFILES.keys()))
    profile = CARD_PROFILES[card_id]
    lo, hi  = profile["typical_amount"]

    txn = {
        "transaction_id": make_txn_id(),
        "card_id":        card_id,
        "amount":         round(random.uniform(lo, hi), 2),
        "merchant":       random.choice(MERCHANTS),
        "city":           profile["city"],
        "timestamp":      time.time(),
        "email_domain":   profile["email"],
        "product_cd":     random.choice(["W", "H", "C"]),
    }
    publish(producer, txn)
    return txn


# ── Attack scenarios ──────────────────────────────────────────────────────────
def attack_probe(producer):
    """Card testing: small probe → large hit within 3 minutes."""
    card_id = "CARD5555"
    log.warning(f"🎯 Simulating probe attack on {card_id}")

    # Probe transaction
    publish(producer, {
        "transaction_id": make_txn_id(),
        "card_id":        card_id,
        "amount":         1.00,
        "merchant":       "OnlineStore",
        "city":           "Mumbai",
        "timestamp":      time.time(),
        "email_domain":   "gmail.com",
        "product_cd":     "W",
    })

    time.sleep(3)  # 3 seconds later (within 5-min window)

    # Large hit
    publish(producer, {
        "transaction_id": make_txn_id(),
        "card_id":        card_id,
        "amount":         50000.00,
        "merchant":       "OnlineStore",
        "city":           "Mumbai",
        "timestamp":      time.time(),
        "email_domain":   "gmail.com",
        "product_cd":     "W",
    })


def attack_velocity(producer):
    """Velocity abuse: 8 transactions in 20 seconds."""
    card_id = "CARD6666"
    log.warning(f"🎯 Simulating velocity attack on {card_id}")

    for i in range(8):
        publish(producer, {
            "transaction_id": make_txn_id(),
            "card_id":        card_id,
            "amount":         round(random.uniform(5000, 15000), 2),
            "merchant":       random.choice(MERCHANTS),
            "city":           "Delhi",
            "timestamp":      time.time(),
            "email_domain":   "yahoo.com",
            "product_cd":     "W",
        })
        time.sleep(2)


def attack_impossible_travel(producer):
    """Impossible travel: Kolkata then Bangalore in 3 seconds."""
    card_id = "CARD7777"
    log.warning(f"🎯 Simulating impossible travel on {card_id}")

    publish(producer, {
        "transaction_id": make_txn_id(),
        "card_id":        card_id,
        "amount":         2500.00,
        "merchant":       "BigBasket",
        "city":           "Kolkata",
        "timestamp":      time.time(),
        "email_domain":   "gmail.com",
        "product_cd":     "W",
    })

    time.sleep(3)

    publish(producer, {
        "transaction_id": make_txn_id(),
        "card_id":        card_id,
        "amount":         8000.00,
        "merchant":       "ElectronicsHub",
        "city":           "Bangalore",
        "timestamp":      time.time(),
        "email_domain":   "gmail.com",
        "product_cd":     "W",
    })


# ── IEEE replay (if dataset available) ───────────────────────────────────────
def ieee_replay(producer, n=20):
    """
    Replay real IEEE-CIS transactions through the pipeline.
    Only runs if data/ieee/train_transaction.csv is present.
    """
    from pathlib import Path
    ieee_path = Path("data/ieee/train_transaction.csv")
    if not ieee_path.exists():
        log.warning("IEEE dataset not found — skipping replay mode")
        return

    import pandas as pd
    log.info("Loading IEEE-CIS transactions for replay...")
    df = pd.read_csv(ieee_path, nrows=1000)

    # Sample some fraud and non-fraud
    fraud = df[df["isFraud"] == 1].sample(min(n//2, len(df[df["isFraud"]==1])))
    clean = df[df["isFraud"] == 0].sample(n - len(fraud))
    sample = pd.concat([fraud, clean]).sample(frac=1)

    cards = list(CARD_PROFILES.keys())
    cities = list(CARD_PROFILES[c]["city"] for c in cards)

    for _, row in sample.iterrows():
        card_id = random.choice(cards)
        txn = {
            "transaction_id": make_txn_id(),
            "card_id":        card_id,
            "amount":         float(row["TransactionAmt"]),
            "merchant":       random.choice(MERCHANTS),
            "city":           random.choice(cities),
            "timestamp":      time.time(),
            "email_domain":   str(row.get("P_emaildomain", "gmail.com")),
            "product_cd":     str(row.get("ProductCD", "W")),
            "transaction_dt": int(row.get("TransactionDT", 86400)),
            "_ieee_is_fraud": bool(row["isFraud"]),  # Ground truth for validation
        }
        label = "🔴 FRAUD" if row["isFraud"] else "🟢 LEGIT"
        log.info(f"IEEE replay | {label} | ₹{txn['amount']:,.2f}")
        publish(producer, txn)
        time.sleep(1)


# ── Main simulation loop ──────────────────────────────────────────────────────
def main():
    producer = connect_producer()
    log.info("Transaction Simulator started")
    log.info("Sequence: 5 normal → probe attack → 5 normal → velocity → 5 normal → travel")
    log.info("=" * 60)

    # Warm up — normal transactions across all cards
    log.info("Phase 1: Normal transactions (warmup)")
    for _ in range(5):
        normal_transaction(producer)
        time.sleep(1)

    time.sleep(2)

    # Attack 1: Probe
    log.info("\nPhase 2: Probe attack")
    attack_probe(producer)
    time.sleep(5)

    # Normal background
    for _ in range(5):
        normal_transaction(producer)
        time.sleep(1)

    # Attack 2: Velocity
    log.info("\nPhase 3: Velocity attack")
    attack_velocity(producer)
    time.sleep(5)

    # Normal background
    for _ in range(5):
        normal_transaction(producer)
        time.sleep(1)

    # Attack 3: Impossible travel
    log.info("\nPhase 4: Impossible travel attack")
    attack_impossible_travel(producer)
    time.sleep(5)

    # Optional: IEEE replay
    ieee_replay(producer, n=10)

    log.info("\n✅ Simulation complete.")


if __name__ == "__main__":
    main()
