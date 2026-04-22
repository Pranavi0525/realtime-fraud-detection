"""
Real-Time Fraud Detection — Streamlit Demo
==========================================
Interactive demo of the fraud detection pipeline.
No Kafka or Redis required — runs the detection logic directly.

Deployable on Streamlit Cloud.
"""

import json
import math
import time
import http.client
import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg: #0a0e1a;
    --surface: #111827;
    --border: #1f2937;
    --accent: #3b82f6;
    --danger: #ef4444;
    --warning: #f59e0b;
    --success: #10b981;
    --text: #f1f5f9;
    --muted: #6b7280;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}

.main-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #60a5fa;
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
}

.subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: var(--muted);
    margin-bottom: 2rem;
    letter-spacing: 0.05em;
}

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}

.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.3rem;
}

.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent);
}

.signal-card {
    background: #1a0f0f;
    border: 1px solid #7f1d1d;
    border-left: 3px solid var(--danger);
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
}

.signal-type {
    color: #fca5a5;
    font-weight: 600;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.signal-detail {
    color: #d1d5db;
    margin-top: 0.2rem;
    font-size: 0.78rem;
}

.clean-card {
    background: #0a1f14;
    border: 1px solid #065f46;
    border-left: 3px solid var(--success);
    border-radius: 6px;
    padding: 1rem 1.2rem;
    font-family: 'JetBrains Mono', monospace;
}

.recommendation-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem;
    margin-top: 1rem;
}

.rec-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.6rem;
    margin-bottom: 1rem;
}

.action-block {
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin-top: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    font-weight: 600;
}

.action-block.block {
    background: #1a0505;
    border: 1px solid var(--danger);
    color: #fca5a5;
}

.action-block.otp {
    background: #1a1205;
    border: 1px solid var(--warning);
    color: #fcd34d;
}

.action-block.review {
    background: #0d1a2e;
    border: 1px solid var(--accent);
    color: #93c5fd;
}

.action-block.monitor {
    background: #0a1f14;
    border: 1px solid var(--success);
    color: #6ee7b7;
}

.scenario-btn {
    width: 100%;
    text-align: left;
}

.stButton > button {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    border-radius: 6px !important;
    padding: 0.5rem 1rem !important;
    width: 100%;
    text-align: left !important;
}

.stButton > button:hover {
    border-color: var(--accent) !important;
    color: #60a5fa !important;
}

.stSlider > div > div > div {
    background: var(--accent) !important;
}

.score-bar-bg {
    background: var(--border);
    border-radius: 4px;
    height: 8px;
    width: 100%;
    margin-top: 0.3rem;
}

.score-bar-fill {
    height: 8px;
    border-radius: 4px;
    transition: width 0.5s ease;
}

[data-testid="stSelectbox"] > div > div {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Model loading ─────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent / "models"

@st.cache_resource
def load_model():
    model_path = MODELS_DIR / "ieee_fraud_model.joblib"
    feature_path = MODELS_DIR / "ieee_feature_names.json"
    meta_path = MODELS_DIR / "ieee_model_metadata.json"

    if not model_path.exists():
        return None, None, None

    model = joblib.load(model_path)
    with open(feature_path) as f:
        feature_names = json.load(f)
    with open(meta_path) as f:
        metadata = json.load(f)
    return model, feature_names, metadata

model, feature_names, model_meta = load_model()
THRESHOLD = model_meta["optimal_threshold"] if model_meta else 0.81

# ── City coordinates ──────────────────────────────────────────────────────────
CITY_COORDS = {
    "Mumbai":    (19.0760, 72.8777),
    "Delhi":     (28.7041, 77.1025),
    "Bangalore": (12.9716, 77.5946),
    "Chennai":   (13.0827, 80.2707),
    "Kolkata":   (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867),
    "Pune":      (18.5204, 73.8567),
}

def haversine_km(c1, c2):
    lat1, lon1 = map(math.radians, CITY_COORDS[c1])
    lat2, lon2 = map(math.radians, CITY_COORDS[c2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 6371 * 2 * math.asin(math.sqrt(a))

# ── Fraud rule checks ─────────────────────────────────────────────────────────
def check_probe(amount, had_probe):
    if had_probe and amount >= 10000:
        return True, f"₹1 probe transaction detected before ₹{amount:,.0f} hit — card testing pattern"
    return False, ""

def check_velocity(txn_count):
    if txn_count > 5:
        return True, f"{txn_count} transactions in 60 seconds — velocity abuse"
    return False, ""

def check_location(city, prev_city, mins_gap):
    if not prev_city or prev_city == city or prev_city == "None":
        return False, ""
    dist = haversine_km(city, prev_city)
    hrs = max(mins_gap / 60, 0.001)
    speed = dist / hrs
    if speed > 900:
        return True, f"{prev_city} → {city} ({dist:.0f} km in {mins_gap} mins = {speed:,.0f} km/h — physically impossible)"
    return False, ""

# ── ML scoring ────────────────────────────────────────────────────────────────
def score_transaction(amount, txn_count, hour, is_weekend, prev_city, city, mins_gap):
    if model is None:
        return None

    dist = haversine_km(city, prev_city) if prev_city and prev_city != "None" and prev_city != city else 0

    row = {f: -999 for f in feature_names}
    row.update({
        "TransactionAmt": amount,
        "TransactionAmt_log": np.log1p(amount),
        "TransactionDT": hour * 3600,
        "C1": txn_count, "C2": txn_count, "C6": txn_count,
        "C11": txn_count, "C9": 1, "C14": 1,
        "D1": mins_gap if mins_gap > 0 else -999,
        "D4": mins_gap if mins_gap > 0 else -999,
        "dist1": dist if dist > 0 else -999,
        "hour_of_day": hour,
        "day_of_week": 5 if is_weekend else 2,
        "is_weekend": int(is_weekend),
        "amt_is_round": int(amount % 10 == 0),
        "is_high_value": int(amount > 500),
        "addr2": 87,
        "card3": 150, "card5": 226,
        "M1": 1, "M2": 1, "M3": 1, "M5": 1, "M6": 1,
    })

    X = pd.DataFrame([row])[feature_names].fillna(-999)
    prob = float(model.predict_proba(X)[0, 1])
    return prob

# ── LLM agent ─────────────────────────────────────────────────────────────────
def get_llm_recommendation(txn_summary, signals, ml_score, api_key):
    if not api_key:
        return None

    system = """You are an expert fraud analyst at a card payment processing company.
Analyze the transaction and fraud signals. Respond ONLY with valid JSON, no markdown:
{
  "fraud_type": "short label",
  "reasoning": "2-3 specific sentences about THIS transaction",
  "recommended_action": "BLOCK_TRANSACTION or STEP_UP_AUTH or FLAG_FOR_REVIEW or ALLOW_WITH_MONITORING",
  "action_rationale": "one sentence why",
  "confidence": "HIGH or MEDIUM or LOW"
}"""

    signals_text = "\n".join(f"- {s}" for s in signals) if signals else "- None"
    ml_text = f"{ml_score*100:.1f}% fraud probability (threshold: {THRESHOLD*100:.1f}%)" if ml_score else "Model unavailable"

    user = f"""Transaction: {txn_summary}
Fraud Signals:
{signals_text}
ML Score: {ml_text}
Provide recommendation as JSON."""

    try:
        payload = json.dumps({
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": 300,
            "temperature": 0.2,
        })
        conn = http.client.HTTPSConnection("api.groq.com", timeout=8)
        conn.request("POST", "/openai/v1/chat/completions", payload, {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
        resp = conn.getresponse()
        data = json.loads(resp.read().decode())
        raw = data["choices"][0]["message"]["content"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except:
        return None

def fallback_recommendation(signals, ml_score):
    n = len(signals)
    has_probe = any("probe" in s.lower() or "test" in s.lower() for s in signals)
    has_travel = any("impossible" in s.lower() for s in signals)
    has_velocity = any("velocity" in s.lower() for s in signals)
    ml_fired = ml_score and ml_score >= THRESHOLD

    if n >= 2 or (n >= 1 and ml_fired):
        action, confidence = "BLOCK_TRANSACTION", "HIGH"
    elif has_probe or has_travel:
        action, confidence = "BLOCK_TRANSACTION", "HIGH"
    elif has_velocity or ml_fired:
        action, confidence = "STEP_UP_AUTH", "MEDIUM"
    elif ml_score and ml_score > 0.5:
        action, confidence = "FLAG_FOR_REVIEW", "MEDIUM"
    else:
        action, confidence = "ALLOW_WITH_MONITORING", "LOW"

    fraud_type = "Card Testing Attack" if has_probe else \
                 "Impossible Travel" if has_travel else \
                 "Velocity Abuse" if has_velocity else \
                 "ML Risk Signal" if ml_fired else "Low Risk"

    reasoning = f"{'Multiple' if n >= 2 else 'Single'} fraud signal{'s' if n != 1 else ''} detected. " + \
                (f"ML model assigns {ml_score*100:.1f}% fraud probability. " if ml_score else "") + \
                "Recommendation based on signal severity and combination."

    return {
        "fraud_type": fraud_type,
        "reasoning": reasoning,
        "recommended_action": action,
        "action_rationale": f"Based on {n} fraud signal(s) with {confidence} confidence.",
        "confidence": confidence,
        "_source": "fallback"
    }

# ── Action display ─────────────────────────────────────────────────────────────
ACTION_CONFIG = {
    "BLOCK_TRANSACTION":     ("🚫", "BLOCK TRANSACTION", "block", "Transaction declined at POS terminal"),
    "STEP_UP_AUTH":          ("📱", "STEP-UP AUTHENTICATION", "otp", "OTP sent to cardholder's registered mobile"),
    "FLAG_FOR_REVIEW":       ("🔍", "FLAG FOR MANUAL REVIEW", "review", "Sent to fraud analyst queue"),
    "ALLOW_WITH_MONITORING": ("👁️", "ALLOW WITH MONITORING", "monitor", "Approved — card added to watchlist"),
}

CONFIDENCE_COLOR = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#10b981"}

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🛡️ FRAUD DETECTION DEMO</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">IEEE-CIS XGBoost · Rule Engine · LLM Reasoning Agent</div>', unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    groq_key = st.text_input("Groq API Key (optional)", type="password",
                              help="Get free key at console.groq.com for live LLM reasoning")

    st.markdown("---")
    st.markdown("### 🎯 Quick Scenarios")
    st.caption("Load a pre-built attack scenario")

    if st.button("🔴 Probe Attack — Card Testing"):
        st.session_state.scenario = "probe"
    if st.button("🟠 Velocity Abuse — Rapid Fire"):
        st.session_state.scenario = "velocity"
    if st.button("🟡 Impossible Travel — Teleport"):
        st.session_state.scenario = "travel"
    if st.button("🟢 Clean Transaction"):
        st.session_state.scenario = "clean"

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    if model_meta:
        st.markdown(f"**AUC-ROC:** `{model_meta['auc_roc']}`")
        st.markdown(f"**PR-AUC:** `{model_meta['pr_auc']}`")
        st.markdown(f"**Threshold:** `{model_meta['optimal_threshold']}`")
        st.markdown(f"**Dataset:** IEEE-CIS (590K txns)")
    else:
        st.warning("Model not found. Place models/ folder next to app.py")

    st.markdown("---")
    st.caption("Built on Apache Kafka + Redis + XGBoost + Groq LLaMA 3.1")

# ── Scenario defaults ─────────────────────────────────────────────────────────
scenario = st.session_state.get("scenario", "clean")

defaults = {
    "probe":    dict(amount=50000, txn_count=2, hour=14, weekend=False, had_probe=True,  prev_city="Mumbai",    city="Mumbai",    mins=15),
    "velocity": dict(amount=8000,  txn_count=9, hour=23, weekend=True,  had_probe=False, prev_city="Delhi",     city="Delhi",     mins=5),
    "travel":   dict(amount=12000, txn_count=2, hour=10, weekend=False, had_probe=False, prev_city="Kolkata",   city="Bangalore", mins=3),
    "clean":    dict(amount=1200,  txn_count=1, hour=11, weekend=False, had_probe=False, prev_city="Chennai",   city="Chennai",   mins=120),
}

d = defaults.get(scenario, defaults["clean"])

# ── Main input form ────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📋 Transaction Details")

    amount = st.number_input("Transaction Amount (₹)", min_value=1.0, max_value=500000.0,
                              value=float(d["amount"]), step=100.0, format="%.2f")

    city = st.selectbox("Current City", list(CITY_COORDS.keys()),
                         index=list(CITY_COORDS.keys()).index(d["city"]))

    merchant = st.text_input("Merchant", value="OnlineStore")

    col1a, col1b = st.columns(2)
    with col1a:
        hour = st.slider("Hour of Day", 0, 23, d["hour"])
    with col1b:
        is_weekend = st.checkbox("Weekend", d["weekend"])

with col2:
    st.markdown("### 🔍 Transaction Context")

    txn_count = st.slider("Transactions in last 60 seconds", 1, 15, d["txn_count"],
                           help="How many transactions has this card made recently?")

    had_probe = st.checkbox("Probe transaction detected (₹1 test in last 5 mins)", d["had_probe"],
                             help="Was there a tiny test transaction before this one?")

    prev_city_options = ["None"] + list(CITY_COORDS.keys())
    prev_city_default = d["prev_city"] if d["prev_city"] in prev_city_options else "None"
    prev_city = st.selectbox("Previous transaction city", prev_city_options,
                              index=prev_city_options.index(prev_city_default))

    mins_gap = st.slider("Minutes since last transaction", 1, 300, d["mins"],
                          help="Time elapsed since the card's previous transaction")

# ── Analyze button ─────────────────────────────────────────────────────────────
st.markdown("---")
analyze = st.button("⚡ ANALYZE TRANSACTION", use_container_width=True)

if analyze:
    with st.spinner("Running fraud detection pipeline..."):
        time.sleep(0.3)

        # Run all checks
        fired_signals = []

        probe_fired, probe_msg = check_probe(amount, had_probe)
        if probe_fired:
            fired_signals.append(("PROBE_ATTACK", probe_msg))

        vel_fired, vel_msg = check_velocity(txn_count)
        if vel_fired:
            fired_signals.append(("VELOCITY_ABUSE", vel_msg))

        if prev_city and prev_city != "None":
            travel_fired, travel_msg = check_location(city, prev_city, mins_gap)
            if travel_fired:
                fired_signals.append(("IMPOSSIBLE_TRAVEL", travel_msg))

        ml_score = score_transaction(amount, txn_count, hour, is_weekend,
                                      prev_city if prev_city != "None" else None,
                                      city, mins_gap)
        ml_fired = ml_score is not None and ml_score >= THRESHOLD

        # Get recommendation
        signal_texts = [msg for _, msg in fired_signals]
        if ml_fired:
            signal_texts.append(f"IEEE-CIS ML: {ml_score*100:.1f}% fraud probability")

        txn_summary = f"₹{amount:,.0f} at {merchant} in {city} | Hour: {hour}:00 | {txn_count} recent txns"

        if signal_texts or ml_fired:
            rec = get_llm_recommendation(txn_summary, signal_texts, ml_score, groq_key)
            if not rec:
                rec = fallback_recommendation(signal_texts, ml_score)
        else:
            rec = {"recommended_action": "ALLOW_WITH_MONITORING", "confidence": "LOW",
                   "fraud_type": "Clean", "reasoning": "No fraud signals detected.",
                   "action_rationale": "Transaction appears legitimate.", "_source": "fallback"}

    # ── Results ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Detection Results")

    r1, r2, r3, r4 = st.columns(4)

    total_flags = len(fired_signals) + (1 if ml_fired else 0)
    severity = "HIGH" if rec["confidence"] == "HIGH" else \
               "MEDIUM" if rec["confidence"] == "MEDIUM" else "LOW"
    sev_color = CONFIDENCE_COLOR.get(rec["confidence"], "#6b7280")

    with r1:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Total Signals</div>
        <div class="metric-value" style="color:{'#ef4444' if total_flags > 0 else '#10b981'}">{total_flags}</div>
        </div>""", unsafe_allow_html=True)

    with r2:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">ML Score</div>
        <div class="metric-value" style="color:{'#ef4444' if ml_fired else '#10b981'}">
        {f"{ml_score*100:.1f}%" if ml_score is not None else "N/A"}</div>
        </div>""", unsafe_allow_html=True)

    with r3:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Severity</div>
        <div class="metric-value" style="color:{sev_color}">{severity}</div>
        </div>""", unsafe_allow_html=True)

    with r4:
        action_short = rec["recommended_action"].replace("_", " ")
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Action</div>
        <div class="metric-value" style="font-size:0.9rem;color:{sev_color}">{action_short}</div>
        </div>""", unsafe_allow_html=True)

    # Signals
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("#### 🚨 Fraud Signals")
        if not fired_signals and not ml_fired:
            st.markdown('<div class="clean-card">✅ No rule violations detected</div>', unsafe_allow_html=True)
        else:
            for sig_type, sig_msg in fired_signals:
                st.markdown(f"""<div class="signal-card">
                <div class="signal-type">⚠ {sig_type}</div>
                <div class="signal-detail">{sig_msg}</div>
                </div>""", unsafe_allow_html=True)

            if ml_fired and ml_score is not None:
                bar_width = min(int(ml_score * 100), 100)
                st.markdown(f"""<div class="signal-card">
                <div class="signal-type">🧠 IEEE-CIS ML MODEL</div>
                <div class="signal-detail">{ml_score*100:.1f}% fraud probability (threshold: {THRESHOLD*100:.1f}%)</div>
                <div class="score-bar-bg"><div class="score-bar-fill" style="width:{bar_width}%;background:#ef4444;"></div></div>
                </div>""", unsafe_allow_html=True)

    with right:
        st.markdown("#### 🤖 LLM Agent Recommendation")
        action = rec.get("recommended_action", "FLAG_FOR_REVIEW")
        icon, label, css_class, consequence = ACTION_CONFIG.get(action, ("⚠️", action, "review", ""))
        confidence = rec.get("confidence", "MEDIUM")
        conf_color = CONFIDENCE_COLOR.get(confidence, "#6b7280")
        source_label = "Groq LLaMA 3.1" if rec.get("_source") != "fallback" else "Rule-based fallback"

        st.markdown(f"""<div class="recommendation-card">
        <div class="rec-header">🤖 LLM AGENT  ·  {source_label}</div>
        <div style="margin-bottom:0.6rem">
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#9ca3af;">FRAUD TYPE</span><br>
            <span style="font-family:'JetBrains Mono',monospace;font-weight:600;color:#f1f5f9;">{rec.get('fraud_type','Unknown')}</span>
        </div>
        <div style="margin-bottom:0.6rem">
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#9ca3af;">CONFIDENCE</span><br>
            <span style="font-family:'JetBrains Mono',monospace;font-weight:700;color:{conf_color};">● {confidence}</span>
        </div>
        <div style="margin-bottom:1rem;font-size:0.85rem;color:#d1d5db;line-height:1.5;">
            {rec.get('reasoning','')}
        </div>
        <div class="action-block {css_class}">
            {icon} {label}<br>
            <span style="font-weight:400;font-size:0.8rem;opacity:0.8;">→ {consequence}</span><br>
            <span style="font-weight:400;font-size:0.75rem;opacity:0.6;">{rec.get('action_rationale','')}</span>
        </div>
        </div>""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("📡 In production: this pipeline runs on Apache Kafka + Redis with 3 microservices. "
           "This demo runs the detection logic directly without Kafka for interactive exploration.")
