from src.agent import load_agent

def test_agent_smoke():
    agent, _, _ = load_agent()
    out = agent.run("My package says delivered but I never received it")
    assert out["intent"]
    assert out["reply"]
    assert out["decision"] in {"auto-handle","escalate"}

def test_backend_adapter_smoke():
    import server
    out = server.make_agent_response("My package says delivered but I never received it")
    assert out["intent"]
    assert out["reply"]
    assert out["decision"] in {"auto-handle", "escalate"}
