# Real-Time Fraud Detection Pipeline
### Apache Kafka + Redis + XGBoost (IEEE-CIS) + LLM Agent | Python

---

## Overview

Most card transaction processors rely on **rule-based fraud detection** — hardcoded thresholds that flag obvious patterns. The problem: sophisticated fraudsters learn the rules and stay just below every threshold.

This project builds an **intelligence layer on top of rules** — combining real-time streaming with an IEEE-CIS trained ML model and an LLM reasoning agent that explains *why* a transaction is suspicious and *what action* to take.

---

## Architecture

\```
[Transaction Simulator]
  ├── Normal transactions (behavioral profiles per card)
  ├── Probe attack      (₹1 test → ₹50,000 hit)
  ├── Velocity attack   (8 txns in 20 seconds)
  ├── Impossible travel (Kolkata → Bangalore in 3 seconds)
  └── IEEE-CIS replay   (real transactions, if dataset available)
            │
            ▼
    Kafka: raw-transactions
            │
            ▼
  [Fraud Detection Service]
    ├── Rule 1: Probe Detection      (Redis TTL pattern)
    ├── Rule 2: Velocity Check       (Redis counter + TTL)
    ├── Rule 3: Location Velocity    (Haversine distance + speed)
    ├── Rule 4: IEEE-CIS XGBoost     (trained on 590K real transactions)
    └── LLM Agent                   (Groq — synthesizes signals, recommends action)
            │
            ▼ (if any signal fired)
    Kafka: fraud-alerts
            │
            ▼
     [Alert Service]
       Structured alert with recommendation
       (production: card block API, SMS, SIEM)
\```

---

## Sample Alert Output

\```
═════════════════════════════════════════════════════════════════
  🔴  FRAUD ALERT — HIGH SEVERITY
═════════════════════════════════════════════════════════════════
  Transaction ID : TXN847291
  Card ID        : CARD5555
  Amount         : ₹50,000.00
  Merchant       : OnlineStore
  City           : Mumbai
  Total Signals  : 2

  [1] PROBE_ATTACK
      Card CARD5555 — ₹1 probe detected before ₹50,000 transaction within 5 minutes
  [2] IEEE_ML_FRAUD_SCORE
      71.2% (threshold: 55.0%) | High transaction amount | Round small amount probe

─────────────────────────────────────────────────────────────────
  IEEE-CIS ML Model: 71.2% fraud probability
─────────────────────────────────────────────────────────────────
  🤖 LLM AGENT RECOMMENDATION  [Groq LLM]
─────────────────────────────────────────────────────────────────
  Fraud Type  : Card Testing Attack
  Confidence  : 🔴 HIGH

  Reasoning:
  Card CARD5555 placed a ₹1 probe 3 minutes before a ₹50,000
  hit at the same merchant. The IEEE ML model assigns 71.2%
  fraud probability. This is a textbook stolen card testing
  pattern — block immediately.

  ► ACTION: 🚫 BLOCK TRANSACTION
    → Card declined — transaction does not proceed
    Why: Two converging signals with HIGH confidence
═════════════════════════════════════════════════════════════════
\```

---

## Fraud Detection Rules

| Rule | Signal | Redis Key | TTL |
|------|--------|-----------|-----|
| Probe Detection | ≤₹10 transaction followed by ≥₹10,000 within 5 mins | `probe:{card_id}` | 300s |
| Velocity Abuse | >5 transactions within 60 seconds | `velocity:{card_id}` | 60s |
| Impossible Travel | Implied travel speed >900 km/h between cities | `location:{card_id}` | 300s |
| IEEE-CIS ML | XGBoost score ≥ optimal threshold (trained on real data) | — | — |

---

## ML Model

**Dataset:** IEEE-CIS Fraud Detection — 590,540 real card transactions (Kaggle)  
**Model:** XGBoost with `scale_pos_weight` for class imbalance (3.5% fraud rate)

| Metric | Value |
|--------|-------|
| AUC-ROC (test) | ~0.90 |
| PR-AUC | ~0.72 |
| Features | Transaction amount, timing, card attributes, C1-C14, D1-D15 |

---

## LLM Agent

The LLM agent synthesizes all signals into a structured recommendation like a human fraud analyst. Two modes — pipeline never breaks:

- **Groq API active** → Dynamic LLM reasoning, unique explanation per transaction
- **No API key** → Smart rule-based fallback with same structured output

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Message broker | Apache Kafka | Fault-tolerant, scalable, decoupled |
| Fast state store | Redis | In-memory, microsecond reads, native TTL |
| ML model | XGBoost (IEEE-CIS) | Trained on real transaction fraud data |
| LLM reasoning | Groq (Llama 3.1) | Dynamic fraud analyst reasoning |
| Containerisation | Docker Compose | Reproducible local environment |
| Language | Python | DS/ML ecosystem |

---

## Setup & Run

### Prerequisites
- Docker Desktop running
- Python 3.10+
- (Optional) Groq API key — free at https://console.groq.com
- (Optional) IEEE-CIS dataset — https://www.kaggle.com/competitions/ieee-fraud-detection/data

### Step 1 — Start Kafka + Redis
\```bash
cd fraud_pipeline
docker-compose up -d
\```

### Step 2 — Install dependencies
\```bash
cd fraud_detection_service && pip install -r requirements.txt
cd alert_service && pip install -r requirements.txt
cd transaction_service && pip install -r requirements.txt
\```

### Step 3 — Set Groq API key (optional)
\```bash
$env:GROQ_API_KEY = "your_key_here"   # Windows PowerShell
export GROQ_API_KEY="your_key_here"   # Linux/Mac
\```

### Step 4 — Run (3 terminals simultaneously)
\```bash
# Terminal 1
cd fraud_detection_service && python main.py

# Terminal 2
cd alert_service && python main.py

# Terminal 3
cd transaction_service && python main.py
\```

### Step 5 — Train IEEE-CIS model (optional)
\```bash
mkdir -p data/ieee
# Place train_transaction.csv from Kaggle into data/ieee/
pip install xgboost scikit-learn pandas numpy joblib
python scripts/train_ieee_model.py
\```

---

## Key Design Decisions

**Why Kafka over a simple API?**  
A synchronous API bottlenecks at high volume. Kafka decouples producers from consumers — transactions queue without data loss, and events can be replayed for audit trails.

**Why Redis over PostgreSQL for rules?**  
Velocity and location checks need sub-millisecond reads. Redis TTL auto-expires state windows with no manual cleanup.

**Why LLM on top of rules + ML?**  
Rules say "flag". ML says "score". The LLM says "this is a card testing attack — block it and here's why." That explanation is readable by analysts and compliance teams. Rules and ML alone cannot explain themselves.

**Why recommend an action, not just an alert?**  
The action — block, OTP, review queue — is what actually prevents fraud. Detection without recommendation is incomplete.
