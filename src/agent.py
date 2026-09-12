import json, re, html
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = json.loads((ROOT/"src/taxonomy.json").read_text())

def normalize(text: str) -> str:
    text = html.unescape(str(text or ""))
    text = re.sub(r"https?://\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

class AmazonSupportAgent:
    def __init__(self, train_df):
        df = train_df.copy()
        df["text"] = df["customer_text"].map(normalize)
        df["label"] = df["final_intent"]
        self.train = df.reset_index(drop=True)

        self.vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3,5), min_df=1, sublinear_tf=True
        )
        X = self.vec.fit_transform(self.train["text"])
        self.clf = LogisticRegression(max_iter=2500, class_weight="balanced")
        self.clf.fit(X, self.train["label"])

        self.centroid_vec = TfidfVectorizer(
            analyzer="word", ngram_range=(1,2), min_df=1, sublinear_tf=True
        )
        self.centroid_X = self.centroid_vec.fit_transform(self.train["text"])

    def classify(self, text):
        x = self.vec.transform([normalize(text)])
        probs = self.clf.predict_proba(x)[0]
        order = np.argsort(probs)[::-1]
        pred = self.clf.classes_[order[0]]
        top_prob = float(probs[order[0]])
        margin = float(probs[order[0]] - probs[order[1]]) if len(order) > 1 else top_prob
        return pred, top_prob, margin

    def retrieve(self, text, k=3):
        q = self.centroid_vec.transform([normalize(text)])
        sims = cosine_similarity(q, self.centroid_X)[0]
        idx = np.argsort(sims)[::-1][:k]
        out=[]
        for i in idx:
            out.append({
                "customer_text": self.train.iloc[i]["customer_text"],
                "intent": self.train.iloc[i]["final_intent"],
                "similarity": round(float(sims[i]), 4)
            })
        return out

    def decide(self, text, intent, prob, margin, evidence):
        t = normalize(text)
        high_risk = any(k in t for k in [
            "hacked","fraud","fraudulent","stolen","unauthorized","suspicious",
            "credit card","bank","scam","password changed","account compromised"
        ])
        repeated = any(k in t for k in [
            "already contacted","again","again and again","multiple","10 calls",
            "5-6 times","weeks","days now","still no response","no response"
        ])
        # Retrieval similarity is an important confidence signal for short/noisy tweets.
        # A strong historical match can override a weak probabilistic margin.
        top_similarity = evidence[0]["similarity"] if evidence else 0.0
        ambiguity = ((prob < 0.12 and top_similarity < 0.28)
                      or (margin < 0.04 and top_similarity < 0.30)
                      or top_similarity < 0.12)

        if high_risk:
            return "escalate", "security/payment risk requires human review"
        if intent in {"customer_support_or_escalation","customer_service_complaint"} and repeated:
            return "escalate", "repeated unresolved support interaction"
        if ambiguity:
            return "escalate", "low classifier confidence or weak historical evidence"
        if intent in {"delivered_but_missing","package_not_received","damaged_or_defective_product",
                      "replacement_or_exchange","return_request_or_policy","amazon_pay_or_cashback",
                      "payment_or_gift_card","order_cancellation"}:
            return "escalate", "account/order-specific action cannot be safely completed without system access"
        return "auto-handle", "high-confidence informational response with no unsupported account action"

    def draft_reply(self, text, intent, evidence):
        p = TAXONOMY.get(intent, {"playbook":"Acknowledge the issue and route the customer to the official support flow without making unsupported promises."})["playbook"]
        if intent == "non_actionable":
            reply = "Thanks for reaching out — glad we could help. If you still need support, please let us know."
        elif intent == "customer_service_complaint":
            reply = "I'm sorry about the experience you've had. We'd like to review what happened and help get this moving. Please send the relevant order/case details privately so the team can investigate."
        elif intent == "customer_support_or_escalation":
            reply = "I'm sorry you've had to follow up repeatedly. We don't want you to keep repeating the same information. Please send the case/order details privately so this can be reviewed and escalated as needed."
        elif intent == "account_access_or_security":
            reply = "I'm sorry you're dealing with this. For your security, please don't share passwords, OTPs or full payment details here. Please use Amazon's official account/support flow, and send the case details privately if you need further help."
        elif intent in {"delivered_but_missing","package_not_received"}:
            reply = "I'm sorry your package hasn't reached you as expected. Please check the delivery location and any safe-place/neighbor details first; if it's still missing, please send the order or tracking details privately so the delivery can be investigated."
        elif intent == "delivery_delay_or_date":
            reply = "I'm sorry the delivery date has changed or the package is late. I don't want to give you an unverified ETA. Please send the order details privately so the current status can be checked."
        elif intent == "delivery_attempt_or_driver":
            reply = "I'm sorry about the delivery attempt. Please send the order details privately so the delivery event and any driver notes can be reviewed."
        elif intent == "tracking_or_carrier":
            reply = "I can understand the frustration when tracking doesn't provide a clear update. Please send the tracking or order number privately so the latest carrier information can be checked."
        elif intent in {"damaged_or_defective_product","replacement_or_exchange","return_request_or_policy"}:
            reply = "I'm sorry there was a problem with the item. Please use the order's return/replacement options, or send the order details privately if you need help checking the available next step."
        elif intent == "amazon_pay_or_cashback":
            reply = "I'm sorry the cashback or Amazon Pay amount hasn't appeared as expected. Please send the relevant transaction/order details privately so the payment can be checked. Please don't post full payment information here."
        elif intent == "payment_or_gift_card":
            reply = "Sorry you're having trouble with the payment or gift card. Please use the official payment/support flow, and don't share full card or gift-card details publicly."
        elif intent == "order_cancellation":
            reply = "I can help point you in the right direction. Cancellation depends on the current order status, so please check the cancellation option on the order or send the order details privately for review."
        elif intent == "order_status_or_details":
            reply = "Please send the order details privately and we can help point you to the right status/support flow."
        elif intent == "preorder_or_release":
            reply = "I understand the concern about the preorder or release. Please send the product/order details privately so the release status can be checked rather than guessing at the date."
        elif intent == "pricing_discount_or_availability":
            reply = "I understand you're checking the price, discount or availability. Offers can vary, so please share the product details privately if you need help checking the specific case."
        elif intent == "product_or_app_support":
            reply = "Sorry you're running into trouble with the product or app. Please share the device/app details and what you're seeing privately so the appropriate troubleshooting steps can be checked."
        elif intent == "contact_or_customer_details":
            reply = "Happy to help. Please use Amazon's official customer-support/contact route rather than posting personal details here. If you've already contacted us, send the case reference privately."
        elif intent == "shipping_speed_or_fee":
            reply = "I'm sorry the shipping speed or fee wasn't as expected. Please send the order details privately so the applicable delivery option and charges can be checked."
        else:
            reply = "Thanks for reaching out. Please send the relevant order or case details privately so the team can review the issue."

        evidence_line = ""
        if evidence:
            evidence_line = f" Evidence used: {evidence[0]['intent']} historical example (similarity {evidence[0]['similarity']})."
        return reply, p, evidence_line

    def run(self, text):
        intent, prob, margin = self.classify(text)
        evidence = self.retrieve(text)
        action, reason = self.decide(text, intent, prob, margin, evidence)
        reply, playbook, evidence_line = self.draft_reply(text, intent, evidence)
        return {
            "input": text,
            "intent": intent,
            "confidence": round(prob, 4),
            "margin": round(margin, 4),
            "decision": action,
            "reason": reason,
            "reply": reply,
            "historical_evidence": evidence,
            "playbook": playbook,
            "evidence_note": evidence_line
        }

def load_agent():
    root = Path(__file__).resolve().parents[1]
    dev = pd.read_csv(root/"data/amazon_gold_200_human_validation_ready.csv")
    hist_path = root/"data/historical_examples.csv"
    hist = pd.read_csv(hist_path) if hist_path.exists() else pd.DataFrame(columns=["customer_text","final_intent"])
    human = pd.read_csv(root/"data/amazon_intent_validation_100.csv")
    # Remove exact overlap between all development/reference examples and the human evaluation set.
    eval_texts = set(human["customer_text"].fillna("").map(normalize))
    dev = dev[["customer_text","final_intent"]].copy()
    hist = hist[["customer_text","final_intent"]].copy()
    train = pd.concat([hist, dev], ignore_index=True)
    train = train[~train["customer_text"].fillna("").map(normalize).isin(eval_texts)]
    train = train.drop_duplicates(subset=["customer_text"]).reset_index(drop=True)
    return AmazonSupportAgent(train), train, human
