"""
LLM Agent — Fraud Analyst Reasoning Layer
==========================================
Synthesizes all fraud signals from the rule engine and ML model
into a structured recommendation — exactly what Vaibhav described:

"The idea would be to have an AI model driven through LLM that
identifies a pattern in order to recommend if it is a fraud."

Two modes:
  1. Groq API available  → LLM reasons dynamically, explains the specific pattern
  2. No API key          → Smart rule-based fallback, still produces structured output

Set your key:
  export GROQ_API_KEY="your_key_here"
  (Free at: https://console.groq.com)
"""

import os
import json
import time
import http.client
from typing import Optional


# ── Groq configuration ────────────────────────────────────────────────────────
GROQ_API_URL  = "api.groq.com"
GROQ_MODEL    = "llama-3.1-8b-instant"   # Fast, free, sufficient for this task
MAX_TOKENS    = 400
TIMEOUT_SECS  = 8


SYSTEM_PROMPT = """You are an AI fraud analyst at a card payment processing company.

You will receive a structured report of a transaction and the fraud signals detected.
Your job is to reason through the signals like a senior fraud analyst would — 
calmly, specifically, and with a clear recommendation.

Respond ONLY with a valid JSON object. No preamble, no explanation outside the JSON.

Required format:
{
  "fraud_type": "one short label, e.g. Card Testing Attack / Velocity Abuse / Impossible Travel / Composite Risk",
  "reasoning": "2-3 sentences. Be specific to THIS transaction's signals. Reference the amount, card, pattern.",
  "recommended_action": "one of: BLOCK_TRANSACTION / STEP_UP_AUTH / FLAG_FOR_REVIEW / ALLOW_WITH_MONITORING",
  "action_rationale": "one sentence explaining why this action",
  "confidence": "HIGH / MEDIUM / LOW"
}

Confidence guide:
  HIGH   — 2+ strong signals or 1 rule + ML score both fired
  MEDIUM — 1 strong signal or borderline ML score
  LOW    — only ML score, no rules fired"""


def _call_groq(prompt: str, api_key: str) -> Optional[dict]:
    """Make a synchronous HTTPS call to Groq API using stdlib only."""
    payload = json.dumps({
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.3,   # Low temp = consistent, structured output
    })

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        conn = http.client.HTTPSConnection(GROQ_API_URL, timeout=TIMEOUT_SECS)
        conn.request("POST", "/openai/v1/chat/completions", payload, headers)
        response = conn.getresponse()
        body = response.read().decode("utf-8")
        conn.close()

        if response.status != 200:
            return None

        data = json.loads(body)
        content = data["choices"][0]["message"]["content"].strip()

        # Strip markdown fences if present
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)

    except Exception as e:
        return None


def _rule_based_recommendation(signals: list[str], ml_score: float, amount: float) -> dict:
    """
    Fallback: deterministic recommendation when no API key is set.
    Still produces structured output — pipeline never breaks.
    """
    signal_text = " | ".join(signals) if signals else "ML score only"
    n_rules = len([s for s in signals if "ML" not in s])

    # Determine fraud type
    if any("probe" in s.lower() or "test" in s.lower() for s in signals):
        fraud_type = "Card Testing Attack"
        reasoning = (
            f"A small probe transaction was detected before a ₹{amount:,.0f} hit. "
            f"This is a classic card testing pattern — fraudsters verify a stolen card "
            f"with a small charge before executing a large transaction."
        )
    elif any("velocity" in s.lower() for s in signals):
        fraud_type = "Velocity Abuse"
        reasoning = (
            f"Abnormal transaction frequency detected on this card. "
            f"Rapid successive transactions are a strong indicator of card compromise — "
            f"fraudsters often spend quickly before the card is blocked."
        )
    elif any("travel" in s.lower() or "location" in s.lower() for s in signals):
        fraud_type = "Impossible Travel"
        reasoning = (
            f"Transaction location is physically inconsistent with prior activity. "
            f"The implied travel speed exceeds what is physically possible, "
            f"suggesting either card cloning or CNP fraud from a different geography."
        )
    elif ml_score > 70:
        fraud_type = "High-Risk Profile"
        reasoning = (
            f"No individual rule fired but the ML model assigned a {ml_score:.1f}% fraud probability. "
            f"The transaction amount of ₹{amount:,.0f} combined with card behavioral features "
            f"matches patterns associated with fraudulent transactions in training data."
        )
    else:
        fraud_type = "Composite Risk"
        reasoning = (
            f"Multiple risk signals detected: {signal_text}. "
            f"No single signal is conclusive but the combination warrants intervention."
        )

    # Determine action
    if n_rules >= 2 or (n_rules >= 1 and ml_score > 65):
        action = "BLOCK_TRANSACTION"
        rationale = "Multiple converging signals indicate high confidence of fraud."
        confidence = "HIGH"
    elif n_rules == 1 or ml_score > 65:
        action = "STEP_UP_AUTH"
        rationale = "One strong signal detected — OTP verification before proceeding."
        confidence = "MEDIUM"
    elif ml_score > 50:
        action = "FLAG_FOR_REVIEW"
        rationale = "ML score elevated — send to fraud analyst queue for manual review."
        confidence = "MEDIUM"
    else:
        action = "ALLOW_WITH_MONITORING"
        rationale = "Signals are weak — approve but add card to monitoring watchlist."
        confidence = "LOW"

    return {
        "fraud_type": fraud_type,
        "reasoning": reasoning,
        "recommended_action": action,
        "action_rationale": rationale,
        "confidence": confidence,
        "_source": "rule_based_fallback",
    }


def build_prompt(transaction: dict, fired_signals: list[str], ml_result: dict) -> str:
    """Build the analyst prompt from all collected signals."""
    ml_available = ml_result.get("available", False)
    ml_score     = ml_result.get("fraud_probability", 0.0)
    ml_signals   = ml_result.get("top_signals", [])
    ml_threshold = ml_result.get("threshold", 50.0)

    rules_block = "\n".join(f"  - {s}" for s in fired_signals) if fired_signals else "  (none)"
    ml_block = (
        f"  Score: {ml_score:.1f}% (threshold: {ml_threshold:.1f}%)\n"
        f"  Fired: {'YES — above threshold' if ml_result.get('is_fraud') else 'NO — below threshold'}\n"
        f"  Top signals: {', '.join(ml_signals)}"
        if ml_available else "  Model not loaded."
    )

    return f"""TRANSACTION DETAILS
===================
Card ID       : {transaction.get('card_id', 'UNKNOWN')}
Amount        : ₹{float(transaction.get('amount', 0)):,.2f}
Merchant      : {transaction.get('merchant', 'UNKNOWN')}
City          : {transaction.get('city', 'UNKNOWN')}
Transaction ID: {transaction.get('transaction_id', 'UNKNOWN')}

RULE ENGINE OUTPUT
==================
Signals fired:
{rules_block}

ML MODEL OUTPUT (IEEE-CIS XGBoost — trained on real transaction fraud data)
============================================================================
{ml_block}

Analyze these signals and provide your recommendation as JSON."""


def get_recommendation(
    transaction: dict,
    fired_signals: list[str],
    ml_result: dict,
) -> dict:
    """
    Main entry point — called from fraud_detection_service/main.py.

    Returns a recommendation dict with keys:
      fraud_type, reasoning, recommended_action, action_rationale, confidence, _source
    """
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    amount  = float(transaction.get("amount", 0))
    ml_score = ml_result.get("fraud_probability", 0.0)

    if api_key:
        prompt = build_prompt(transaction, fired_signals, ml_result)
        result = _call_groq(prompt, api_key)
        if result:
            result["_source"] = "groq_llm"
            return result
        # If Groq call failed, fall through to fallback silently

    return _rule_based_recommendation(fired_signals, ml_score, amount)


# ── Action display helpers ────────────────────────────────────────────────────
ACTION_DISPLAY = {
    "BLOCK_TRANSACTION":     ("🚫", "BLOCK TRANSACTION",     "Transaction declined at POS"),
    "STEP_UP_AUTH":          ("📱", "STEP-UP AUTHENTICATION", "OTP sent to cardholder's registered mobile"),
    "FLAG_FOR_REVIEW":       ("🔍", "FLAG FOR MANUAL REVIEW", "Sent to fraud analyst queue"),
    "ALLOW_WITH_MONITORING": ("👁️",  "ALLOW WITH MONITORING", "Approved — card added to watchlist"),
}

CONFIDENCE_COLOR = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}


def format_recommendation(rec: dict) -> str:
    """Format recommendation for terminal output."""
    action = rec.get("recommended_action", "FLAG_FOR_REVIEW")
    icon, label, consequence = ACTION_DISPLAY.get(action, ("⚠️", action, ""))
    confidence = rec.get("confidence", "MEDIUM")
    source = "Groq LLM" if rec.get("_source") == "groq_llm" else "Rule-based fallback"

    return f"""
┌─────────────────────────────────────────────────────────────┐
│  🤖  LLM AGENT RECOMMENDATION  [{source}]
├─────────────────────────────────────────────────────────────┤
│  Fraud Type    : {rec.get('fraud_type', 'Unknown')}
│  Confidence    : {CONFIDENCE_COLOR.get(confidence, '⚪')} {confidence}
├─────────────────────────────────────────────────────────────┤
│  REASONING:
│  {rec.get('reasoning', '')}
├─────────────────────────────────────────────────────────────┤
│  {icon} ACTION: {label}
│     → {consequence}
│     Why: {rec.get('action_rationale', '')}
└─────────────────────────────────────────────────────────────┘"""


if __name__ == "__main__":
    # Quick smoke test — no API key needed
    test_txn = {
        "card_id": "CARD5555",
        "amount": 50000,
        "merchant": "OnlineStore",
        "city": "Mumbai",
        "transaction_id": "TXN847291",
    }
    test_signals = ["PROBE_ATTACK: ₹1 probe detected 3 minutes ago"]
    test_ml = {
        "available": True,
        "fraud_probability": 71.2,
        "is_fraud": True,
        "threshold": 55.0,
        "top_signals": ["High transaction amount", "Card behavioral pattern"],
    }

    rec = get_recommendation(test_txn, test_signals, test_ml)
    print(format_recommendation(rec))
