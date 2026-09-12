import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from .agent import load_agent, normalize

ROOT = Path(__file__).resolve().parents[1]

def reply_proxy(row):
    text = normalize(row["reply"])
    score = 0
    if any(k in text for k in ["sorry","apolog"]): score += 1
    if any(k in text for k in ["please","send","check","use","share"]): score += 1
    if "privately" in text or "official" in text: score += 1
    if not any(k in text for k in ["guarantee","will definitely","definitely arrive","refund you"]): score += 1
    return score / 4

def main():
    agent, train, human = load_agent()
    human = human.dropna(subset=["human_intent"]).copy()
    y = human["human_intent"].astype(str)
    texts = human["customer_text"].fillna("").astype(str)

    # Trivial majority baseline
    majority = y.value_counts().idxmax()
    pred_majority = np.array([majority]*len(y))

    # Simple TF-IDF + logistic regression baseline
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2500, class_weight="balanced"))
    ])
    pipe.fit(train["customer_text"].fillna("").astype(str), train["final_intent"].astype(str))
    pred_simple = pipe.predict(texts)

    # Agent
    outputs = [agent.run(t) for t in texts]
    pred_agent = np.array([o["intent"] for o in outputs])

    rows = []
    for name, pred in [
        ("Majority baseline", pred_majority),
        ("TF-IDF + LogisticRegression", pred_simple),
        ("Evidence-grounded agent", pred_agent)
    ]:
        rows.append({
            "model": name,
            "accuracy": round(accuracy_score(y, pred), 4),
            "macro_f1": round(f1_score(y, pred, average="macro", zero_division=0), 4),
            "weighted_f1": round(f1_score(y, pred, average="weighted", zero_division=0), 4),
        })
    metrics = pd.DataFrame(rows)
    print("\n=== HEADLINE RESULTS ===")
    print(metrics.to_string(index=False))

    # Escalation analysis
    decisions = pd.DataFrame(outputs)
    # Treat difficult/transactional classes as human-needed for a conservative safety proxy.
    high_risk = y.isin({
        "account_access_or_security","amazon_pay_or_cashback","payment_or_gift_card",
        "delivered_but_missing","package_not_received","damaged_or_defective_product",
        "replacement_or_exchange","return_request_or_policy","order_cancellation"
    })
    pred_escalate = decisions["decision"].eq("escalate")
    escalation_recall = (pred_escalate & high_risk).sum() / max(1, high_risk.sum())
    escalation_rate = pred_escalate.mean()
    print("\n=== ESCALATION PROXY ===")
    print(f"Escalation rate: {escalation_rate:.3f}")
    print(f"High-risk escalation recall: {escalation_recall:.3f}")

    # Confusion matrix for agent
    labels = sorted(set(y) | set(pred_agent))
    cm = confusion_matrix(y, pred_agent, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(ROOT/"reports/confusion_matrix.csv")

    # Failure examples
    failure_mask = y.to_numpy() != pred_agent
    failures = pd.DataFrame({
        "customer_text": texts.to_numpy(),
        "gold_intent": y.to_numpy(),
        "predicted_intent": pred_agent,
        "confidence": decisions["confidence"],
        "reason": decisions["reason"]
    })[failure_mask].copy()
    failures.to_csv(ROOT/"reports/failure_examples.csv", index=False)

    # Proxy reply score
    rp = pd.DataFrame([{
        "customer_text": o["input"],
        "intent": o["intent"],
        "decision": o["decision"],
        "reply": o["reply"],
        "reply_proxy_score": reply_proxy(o)
    } for o in outputs])
    print("\n=== REPLY SAFETY/QUALITY PROXY ===")
    print(f"Mean proxy score: {rp.reply_proxy_score.mean():.3f} / 1.0")
    rp.to_csv(ROOT/"reports/reply_outputs.csv", index=False)

    report = f"""# Evaluation Report

## Headline
- Human evaluation examples: **{len(y)}**
- Majority accuracy: **{accuracy_score(y, pred_majority):.3f}**
- TF-IDF + LogisticRegression macro-F1: **{f1_score(y,pred_simple,average='macro',zero_division=0):.3f}**
- Evidence-grounded agent macro-F1: **{f1_score(y,pred_agent,average='macro',zero_division=0):.3f}**
- Agent weighted-F1: **{f1_score(y,pred_agent,average='weighted',zero_division=0):.3f}**
- Escalation rate: **{escalation_rate:.3f}**
- High-risk escalation recall proxy: **{escalation_recall:.3f}**

## Why this metric
Macro-F1 prevents the frequent delivery/support classes from completely hiding failures on small classes.

## Failure analysis
See `failure_examples.csv` and `confusion_matrix.csv`. The recurring boundaries are delivery-date vs missing-package, tracking vs complaint, login vs security, and product issue vs complaint.

## Reply quality
The offline score is a safety proxy, not a substitute for human review. The optional LLM judge is intended to measure groundedness, helpfulness, tone, actionability and unsupported promises.

## Mandatory caveat
The headline score is based on only 100 human-labelled examples and the development set is semantically close to the evaluation distribution. It demonstrates prototype quality, not production reliability.
"""
    (ROOT/"reports/evaluation_report.md").write_text(report)
    print("\nSaved reports/")

if __name__ == "__main__":
    main()
