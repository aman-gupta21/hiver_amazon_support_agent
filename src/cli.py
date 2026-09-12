import sys, json
from .agent import load_agent

def main():
    text = " ".join(sys.argv[1:]).strip()
    if not text:
        print('Usage: python -m src.cli "your customer message"')
        return
    agent, _, _ = load_agent()
    print(json.dumps(agent.run(text), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
