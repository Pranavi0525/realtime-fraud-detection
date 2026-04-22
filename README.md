# Real-Time Fraud Detection Pipeline

**Apache Kafka · Redis · XGBoost (IEEE-CIS) · LLM Agent · Streamlit**

🔗 **Live Demo:** https://realtime-fraud-detection-zdkqrvion9b5rikzxdmaqk.streamlit.app/

---

## What This Does

Card transaction fraud detection that goes beyond hardcoded rules. Most production systems flag obvious patterns — this project adds an **intelligence layer on top**: an ML model trained on real transaction fraud data, combined with an LLM agent that explains *why* a transaction is suspicious and recommends *what action* to take.

Three layers run on every transaction:
- **Rule engine** — catches probe attacks, velocity abuse, and impossible travel in under a millisecond
- **XGBoost ML model** — trained on 590,540 real IEEE-CIS card transactions, catches subtle fraud that rules miss
- **LLM reasoning agent** — synthesizes all signals into a structured human-readable recommendation

---

## Architecture

```
[Transaction Simulator]
  ├── Normal transactions (behavioral profiles per card)
  ├── Probe attack      (₹1 test → ₹50,000 hit)
  ├── Velocity attack   (8 txns in 20 seconds)
  └── Impossible travel (Kolkata → Bangalore in 3 mins)
            │
            ▼
    Kafka: raw-transactions
            │
            ▼
  [Fraud Detection Service]
    ├── Rule 1: Probe Detection      (Redis TTL pattern)
    ├── Rule 2: Velocity Check       (Redis counter + TTL)
    ├── Rule 3: Location Velocity    (Haversine + speed threshold)
    ├── Rule 4: IEEE-CIS XGBoost     (590K real transactions)
    └── LLM Agent                   (Groq — synthesizes signals)
            │
            ▼ (if any signal fired)
    Kafka: fraud-alerts
            │
            ▼
     [Alert Service]
       Structured alert with recommended action
```

---

## Sample Output

```
═══════════════════════════════════════════════════════════
  🔴  FRAUD ALERT — HIGH SEVERITY
═══════════════════════════════════════════════════════════
  Transaction ID : TXN847291
  Card ID        : CARD5555
  Amount         : ₹50,000.00
  Merchant       : OnlineStore
  City           : Mumbai
  Total Signals  : 2

  [1] PROBE_ATTACK
      ₹1 probe detected 3 mins before ₹50,000 transaction

  [2] IEEE_ML_FRAUD_SCORE
      71.2% fraud probability (threshold: 55.0%)

───────────────────────────────────────────────────────────
  🤖 LLM AGENT RECOMMENDATION
───────────────────────────────────────────────────────────
  Fraud Type  : Card Testing Attack
  Confidence  : HIGH

  Reasoning   : Classic stolen card pattern — small probe
                followed immediately by large transaction
                at the same merchant. ML corroborates.

  ► ACTION: 🚫 BLOCK TRANSACTION
═══════════════════════════════════════════════════════════
```

---

## Fraud Rules

| Rule | Trigger | State Store | TTL |
|------|---------|-------------|-----|
| Probe Detection | ≤₹10 test → ≥₹10,000 hit within 5 mins | Redis | 300s |
| Velocity Abuse | >5 transactions within 60 seconds | Redis | 60s |
| Impossible Travel | Implied speed >900 km/h between cities | Redis | 300s |
| IEEE-CIS ML | XGBoost score ≥ optimal threshold | — | — |

---

## ML Model

**Dataset:** IEEE-CIS Fraud Detection — 590,540 real labeled card transactions  
**Algorithm:** XGBoost with `scale_pos_weight` to handle 3.5% fraud class imbalance

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.9494 |
| PR-AUC | 0.692 |
| Optimal Threshold | 0.81 |
| Training samples | ~413,000 |

**Key features:** Transaction amount (log-transformed), hour of day, card attributes, counting features (C1–C14 transaction history), timedelta features (D1–D15 days since last transaction), Vesta-engineered features (V12–V100).

**Missing value strategy:** All nulls filled with `-999` sentinel. XGBoost learns optimal branch direction for missing values during training — replicated identically at inference time.

**Class imbalance:** `scale_pos_weight = 28` (ratio of legit to fraud transactions) — prevents the model from ignoring the minority fraud class.

---

## LLM Agent

Runs after rules and ML complete. Takes all fired signals as input and produces a structured recommendation with fraud type, confidence level, reasoning, and a specific action.

Two modes — pipeline never breaks:
- **Groq API active** → Live LLM reasoning, unique explanation per transaction
- **No API key** → Rule-based fallback with identical output structure

Actions: `BLOCK_TRANSACTION` · `STEP_UP_AUTH` · `FLAG_FOR_REVIEW` · `ALLOW_WITH_MONITORING`

---

## Tech Stack

| Component | Technology | Reason |
|-----------|-----------|--------|
| Message broker | Apache Kafka | Decoupled, fault-tolerant, replayable |
| State store | Redis | Sub-millisecond reads, native TTL |
| ML model | XGBoost | Handles missing values, class imbalance, fast inference |
| LLM | Groq (LLaMA 3.1 8B) | Fast inference, free tier available |
| Demo UI | Streamlit | No infrastructure needed to explore |
| Containers | Docker Compose | Reproducible local environment |

---

## Local Setup

### Prerequisites
- Docker Desktop
- Python 3.10+
- Groq API key (optional) — free at https://console.groq.com

### Run the full pipeline

```bash
# 1. Start Kafka + Redis
cd fraud_pipeline
docker-compose up -d

# 2. Install dependencies
cd fraud_detection_service && pip install -r requirements.txt
cd ../alert_service && pip install -r requirements.txt
cd ../transaction_service && pip install -r requirements.txt

# 3. Set Groq key (optional)
export GROQ_API_KEY="your_key_here"        # Linux/Mac
$env:GROQ_API_KEY = "your_key_here"        # Windows PowerShell

# 4. Run (3 terminals)
cd fraud_detection_service && python main.py
cd alert_service && python main.py
cd transaction_service && python main.py
```

### Run the Streamlit demo (no Docker needed)

```bash
cd streamlit_app
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### Train the IEEE-CIS model

```bash
# Download train_transaction.csv from:
# https://www.kaggle.com/competitions/ieee-fraud-detection/data
# Place in data/ieee/

python scripts/train_ieee_model.py
# Outputs: models/ieee_fraud_model.joblib + metadata
```

---

## Design Decisions

**Kafka over a direct API** — synchronous calls bottleneck under high transaction volume. Kafka queues without data loss and enables event replay for audit.

**Redis over PostgreSQL for rules** — velocity and location checks need sub-millisecond state reads. Redis TTL auto-expires state windows with no cleanup code.

**-999 sentinel over mean imputation** — IEEE-CIS features are missing by design (Vesta only computes certain features for certain transaction types). Sentinel preserves the signal that a feature is absent; mean imputation would destroy it.

**LLM on top of rules + ML** — rules produce a flag, ML produces a score, the LLM produces an explanation. Fraud analysts and compliance teams need readable reasoning, not just a probability.

---

## Project Structure

```
fraud_pipeline_v2/
├── fraud_pipeline/
│   ├── fraud_detection_service/
│   │   ├── main.py              ← Kafka consumer, orchestrates all layers
│   │   ├── ieee_scorer.py       ← IEEE-CIS XGBoost inference
│   │   └── llm_agent.py         ← Groq LLM reasoning agent
│   ├── alert_service/
│   │   └── main.py              ← Consumes fraud-alerts topic
│   ├── transaction_service/
│   │   └── main.py              ← Simulates card transactions
│   ├── scripts/
│   │   └── train_ieee_model.py  ← Model training pipeline
│   └── docker-compose.yml
├── streamlit_app/
│   ├── app.py                   ← Interactive demo
│   ├── requirements.txt
│   └── models/                  ← Trained model artifacts
└── README.md
```

---

## Live Demo

Try the four pre-built scenarios — no setup required:

**https://realtime-fraud-detection-zdkqrvion9b5rikzxdmaqk.streamlit.app/**

| Scenario | What it triggers |
|----------|-----------------|
| 🔴 Probe Attack | Probe rule + ML score |
| 🟠 Velocity Abuse | Velocity rule |
| 🟡 Impossible Travel | Location velocity rule |
| 🟢 Clean Transaction | No signals — allow with monitoring |