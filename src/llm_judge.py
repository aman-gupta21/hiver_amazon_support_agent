import os, json, re, argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RUBRIC = """
Score the support reply from 1-5 on each dimension:
1. groundedness: does it avoid inventing account/order facts and stay consistent with provided evidence?
2. helpfulness: does it meaningfully move the customer toward a safe next step?
3. tone: empathetic, concise, professional, not defensive.
4. actionability: gives a clear next action appropriate to the situation.
5. promise_safety: avoids unsupported guarantees about refunds, dates, eligibility or outcomes.

Return JSON only:
{"groundedness":1-5,"helpfulness":1-5,"tone":1-5,"actionability":1-5,"promise_safety":1-5,"overall":1-5,"reason":"short reason"}
"""

def judge_with_openai(items):
    from openai import OpenAI
    client = OpenAI()
    out=[]
    for item in items:
        prompt = RUBRIC + "\n\nCustomer:\n" + item["customer_text"] + \
                 "\n\nPredicted intent:\n" + item["intent"] + \
                 "\n\nReply:\n" + item["reply"]
        resp = client.responses.create(
            model=os.getenv("OPENAI_JUDGE_MODEL","gpt-5-mini"),
            input=prompt
        )
        text = resp.output_text
        m = re.search(r"\{.*\}", text, re.S)
        out.append(json.loads(m.group(0) if m else text))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()
    path = ROOT/"reports/reply_outputs.csv"
    if not path.exists():
        raise SystemExit("Run `python -m src.evaluate` first.")
    df = pd.read_csv(path).head(args.limit)
    items = df.to_dict("records")
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. The offline evaluation is still complete.")
        print("Create reports/judge_calibration.csv using the rubric in this file, then rerun with an API key.")
        return
    scores = judge_with_openai(items)
    result = pd.concat([df.reset_index(drop=True), pd.DataFrame(scores)], axis=1)
    result.to_csv(ROOT/"reports/llm_judge_results.csv", index=False)
    print(result[["overall","groundedness","helpfulness","tone","actionability","promise_safety"]].mean(numeric_only=True))

if __name__ == "__main__":
    main()
