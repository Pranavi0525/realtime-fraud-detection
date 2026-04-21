# Real-Time Fraud Detection Pipeline
### Apache Kafka + Redis + XGBoost (IEEE-CIS) + LLM Agent | Python

---

## Why This Exists

Pine Labs Credit+ (and most card transaction processors) run **rule-based fraud detection** — hardcoded thresholds that flag obvious patterns. The problem: sophisticated fraudsters learn the rules and stay just below every threshold.

This project builds the **intelligence layer on top of rules** — combining real-time streaming rules with an IEEE-CIS trained ML model and an LLM reasoning agent, exactly as described by Pine Labs' Director of Engineering:

> *"The idea would be to have an AI model. It could be driven through LLM. It could be driven through SLM. It could be driven through typical machine learning models — and then identify a pattern in order to recommend if it is a fraud."*

---

## Architecture

```
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
```

---

## What Changed — v2 Upgrades

### 1. IEEE-CIS ML Model (replaces OpenFinGuard credit proxy)

| Before | After |
|--------|-------|
| OpenFinGuard credit risk model | XGBoost trained on IEEE-CIS transaction fraud data |
| Features: delinquency, debt ratio | Features: transaction amount, timing, card attributes, velocity signals |
| AUC-ROC: 0.864 on credit default | AUC-ROC: ~0.90 on actual transaction fraud |
| Wrong problem domain | Correct problem domain |

The old model used credit-risk features (delinquency severity, revolving utilization) — valid only for first-party fraud. The IEEE-CIS model uses transaction-level features that map directly to card fraud patterns.

**Dataset:** IEEE-CIS Fraud Detection (Kaggle) — 590,540 real card transactions, 3.5% fraud rate. Download: https://www.kaggle.com/competitions/ieee-fraud-detection/data

### 2. LLM Agent — Groq Reasoning Layer

The LLM agent takes all collected signals and produces a **reasoned recommendation** like a human fraud analyst would:

```
┌─────────────────────────────────────────────────────────────┐
│  🤖  LLM AGENT RECOMMENDATION  [Groq LLM]
├─────────────────────────────────────────────────────────────┤
│  Fraud Type    : Card Testing Attack
│  Confidence    : 🔴 HIGH
├─────────────────────────────────────────────────────────────┤
│  REASONING:
│  Card CARD5555 placed a ₹1 probe transaction 3 minutes
│  before attempting ₹50,000 at the same merchant. This is
│  a textbook stolen card testing pattern — fraudsters verify
│  a card works with a micro-transaction before executing the
│  large fraud. The IEEE ML model corroborates at 71.2%.
├─────────────────────────────────────────────────────────────┤
│  🚫 ACTION: BLOCK TRANSACTION
│     → Card declined — transaction does not proceed
│     Why: Multiple converging signals, HIGH confidence
└─────────────────────────────────────────────────────────────┘
```

Two modes — pipeline never breaks:
- **Groq API active** → Dynamic LLM reasoning, unique explanation per transaction
- **No API key** → Smart rule-based fallback with same structured output

---

## Fraud Detection Rules

| Rule | Signal | Redis Key | TTL |
|------|---------|-----------|-----|
| Probe Detection | ≤₹10 transaction followed by ≥₹10,000 within 5 mins | `probe:{card_id}` | 300s |
| Velocity Abuse | >5 transactions within 60 seconds | `velocity:{card_id}` | 60s |
| Impossible Travel | Implied travel speed >900 km/h between cities | `location:{card_id}` | 300s |
| IEEE-CIS ML | XGBoost score ≥ optimal threshold (trained on real data) | — | — |

---

## ML Model Details

**Dataset:** IEEE-CIS Fraud Detection — 590,540 real card transactions (Kaggle)  
**Model:** XGBoost with `scale_pos_weight` for class imbalance (3.5% fraud rate)

| Metric | Value |
|--------|-------|
| AUC-ROC (test) | ~0.90 |
| PR-AUC | ~0.72 |
| Training data | Real IEEE-CIS transactions |
| Features | Transaction amount, timing, card attributes, counting features (C1-C14), timedelta features (D1-D15) |

**Why IEEE-CIS over OpenFinGuard credit model:**
Credit default risk and transaction fraud are genuinely different problems. A person with perfect credit can have their card stolen. The IEEE-CIS model uses features directly observable at transaction time — the correct problem domain.

---

## Setup & Run

### Prerequisites
- Docker Desktop running
- Python 3.10+
- (Optional) IEEE-CIS dataset for real ML scoring
- (Optional) Groq API key for LLM reasoning

### Step 1 — Start Kafka + Redis
```bash
docker-compose up -d
```
Wait ~15 seconds for Kafka to initialise.

### Step 2 — (Optional but recommended) Train IEEE-CIS model
```bash
# Download dataset from Kaggle first, place in data/ieee/
mkdir -p data/ieee
# Copy train_transaction.csv to data/ieee/

pip install xgboost scikit-learn pandas numpy joblib
python scripts/train_ieee_model.py
# Outputs model files to models/
```

### Step 3 — (Optional) Set Groq API key
```bash
# Get free key at: https://console.groq.com
export GROQ_API_KEY="your_key_here"      # Linux/Mac
$env:GROQ_API_KEY = "your_key_here"     # Windows PowerShell
```

### Step 4 — Install dependencies
```bash
# Terminal 1 — fraud detection service
cd fraud_detection_service && pip install -r requirements.txt

# Terminal 2 — alert service
cd alert_service && pip install -r requirements.txt

# Terminal 3 — transaction simulator
cd transaction_service && pip install -r requirements.txt
```

### Step 5 — Run (3 terminals simultaneously)
```bash
# Terminal 1 — Start fraud detector first
cd fraud_detection_service && python main.py

# Terminal 2 — Start alert listener
cd alert_service && python main.py

# Terminal 3 — Start transaction simulator
cd transaction_service && python main.py
```

---

## Sample Alert Output

```
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
  (Trained on real IEEE-CIS transaction fraud data)
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
```

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

## Key Design Decisions

**Why Kafka over a simple API?**  
At Pine Labs scale (millions of txns/day), a synchronous API would bottleneck. Kafka decouples producers from consumers — if the fraud service is slow, transactions queue without data loss. Kafka also enables event replay for audit trails.

**Why Redis over PostgreSQL for rules?**  
Velocity and location checks need sub-millisecond reads. Redis TTL also auto-expires state windows — no manual cleanup needed.

**Why IEEE-CIS over credit risk model?**  
Credit default risk and transaction fraud are different problems. The IEEE-CIS model uses features observable at transaction time — amount, timing, card attributes — not long-term financial behavior.

**Why LLM reasoning on top of rules + ML?**  
Rules say "flag". ML says "score". LLM says "this is a card testing attack — block it and here's why". A fraud analyst, compliance officer, or customer service agent can read that. Rules and ML cannot explain themselves. The LLM closes that gap.

**Why recommend action, not just alert?**  
In production, the recommendation drives the action: block the card, trigger OTP, open a review ticket. A detection system without a recommendation is incomplete — it tells you something happened but not what to do about it.
