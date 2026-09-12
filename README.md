# Amazon AI Support Agent — Hiver SDE Intern Take-Home

A reproducible, evidence-first customer-support agent built around one brand: **Amazon**.

The system does three things:

1. **Intent classification** into a compact taxonomy inferred from the supplied Amazon support data.
2. **Evidence-grounded reply drafting** using nearest historical examples + an intent-specific resolution playbook.
3. **Auto-handle vs. human escalation** with an explicit reason and conservative confidence rules.

> Important: the supplied 200-row file is a model-reviewed gold-development set, while the separate 100-row file contains human-validated labels. The 100-row human set is kept as the headline evaluation set. Exact duplicate customer messages are excluded from training where possible.

## 1. Quickstart (<15 min)

### Requirements
- Python 3.10+
- `pip install -r requirements.txt`

### Run the offline evaluation
```bash
python -m src.evaluate
```

This prints:
- majority baseline
- TF-IDF + logistic regression baseline
- retrieval/routing agent results
- confusion matrix
- escalation metrics
- reply-quality heuristic scores

### Try the agent
```bash
python -m src.cli "My package says delivered but I never received it"
```

### Optional LLM judge
Set an API key:
```bash
export OPENAI_API_KEY="..."
```

Then:
```bash
python -m src.llm_judge --limit 30
```

The judge produces structured scores for groundedness, helpfulness, tone, actionability and unnecessary promises.

## 2. Data used

- `data/amazon_gold_200_human_validation_ready.csv`
  - 200 Amazon examples with a normalized `final_intent`.
  - Used as development/reference data.
- `data/amazon_intent_validation_100.csv`
  - 100 examples with `human_intent`.
  - Used as the primary human evaluation set.

The original assignment's large Kaggle Twitter dataset is intentionally **not bundled**. The repo is designed so a larger source file can be supplied later without changing the evaluation harness.

## 3. Intent taxonomy

The taxonomy is intentionally small enough to be operational:

- `account_access_or_security`
- `amazon_pay_or_cashback`
- `contact_or_customer_details`
- `customer_service_complaint`
- `customer_support_or_escalation`
- `damaged_or_defective_product`
- `delivered_but_missing`
- `delivery_attempt_or_driver`
- `delivery_delay_or_date`
- `non_actionable`
- `order_cancellation`
- `order_status_or_details`
- `package_not_received`
- `payment_or_gift_card`
- `preorder_or_release`
- `pricing_discount_or_availability`
- `product_or_app_support`
- `replacement_or_exchange`
- `return_request_or_policy`
- `shipping_speed_or_fee`
- `tracking_or_carrier`

These labels are operational rather than pretending to be the original dataset's ground truth. Several source intents collapse into a more useful support action.

## 4. Architecture

```text
incoming tweet
     |
     v
text normalization
     |
     +-----------------------------+
     |                             |
     v                             v
TF-IDF classifier            nearest-example retrieval
     |                             |
     +--------------+--------------+
                    |
                    v
             confidence / ambiguity
                    |
          +---------+---------+
          |                   |
       auto-handle         escalate
          |                   |
          v                   v
 resolution playbook     human reason
          |
          v
 evidence-grounded reply draft
```

### Why retrieval + playbook?

The dataset contains customer complaints and outcomes/intent annotations, but not a clean action-policy database. I therefore avoid inventing historical Amazon policy. The reply generator is grounded in:
- similar historical customer examples;
- the observed intent;
- a conservative response playbook.

It never invents order status, refund amounts, delivery dates or policy guarantees.

## 5. Evaluation design

### Baselines
**Trivial baseline:** always predict the majority human intent.

**Simple baseline:** TF-IDF word/character features + logistic regression.

**Agent:** TF-IDF classifier + nearest-neighbour evidence + conservative escalation policy.

The headline metric is **macro-F1**, not accuracy, because the 100-example human set is highly imbalanced.

### Reply evaluation
The offline harness calculates:
- groundedness proxy
- actionability
- tone
- unsupported-promise penalty
- retrieval evidence availability

The optional LLM judge scores each reply on a 1–5 rubric.

### Human agreement for the LLM judge
`data/judge_calibration.csv` is a deliberately small calibration sheet. A human can score a sample of generated replies using the same 1–5 rubric; `src/llm_judge.py` computes:
- Pearson/Spearman correlation
- exact/adjacent agreement
- mean absolute error

This is included because a judge score without judge validation is not trustworthy.

## 6. What is misleading about my headline number?

The headline macro-F1 is useful but not equivalent to production readiness.

1. **Only 100 human-labelled examples** are available, so confidence intervals are wide.
2. The set is heavily concentrated in a handful of delivery/support intents.
3. Twitter messages are noisy, short and context-dependent; a single tweet can be ambiguous.
4. Some development examples are semantically close to evaluation examples even when exact duplicates are removed.
5. The reply system is evaluated mainly for safe, useful drafting—not for actual resolution because the dataset does not contain transactional state.
6. Escalation is deliberately conservative; a high auto-handle rate would not automatically be better.
7. LLM-judge scores are not human truth unless the calibration file shows agreement.

The correct interpretation is therefore: **the system is a promising prototype with evidence of intent-routing quality, not a production-ready autonomous support agent.**

## 7. Top failure modes to inspect

The report generated by `src.evaluate` highlights:
1. delivery date vs. package not received
2. tracking/carrier vs. customer-service complaint
3. account login vs. account security
4. product issue hidden inside a complaint
5. vague “help / thanks / already did that” messages

Each is a taxonomy-boundary problem rather than simply a model-capacity problem.

## 8. Decision log

See `reports/decision_log.md`.

## 9. Sources

- Kaggle Customer Support on Twitter: `thoughtvector/customer-support-on-twitter`
- Human-labelled development/evaluation files supplied during the project.
- OpenAI API is optional for the judge only; the core classifier runs locally.

No source code or policy text is copied from Amazon.
