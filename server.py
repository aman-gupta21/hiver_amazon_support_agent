import json
import os
import sys
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

from src.agent import load_agent

ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"
PORT = int(os.environ.get("PORT", "8000"))

agent, _, _ = load_agent()


def make_agent_response(text: str):
    """Return the same structured reply payload used by the Python agent."""
    return agent.run(text)


class AgentHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=str(FRONTEND_DIR), **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def log_message(self, fmt, *args):
        sys.stdout.write("[server] %s\n" % (fmt % args))

    def do_OPTIONS(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/agent":
            self.send_json({"ok": True, "allowed": ["POST"]}, status=200)
            return
        self.send_json({"ok": True}, status=200)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self.send_json({"status": "ok", "message": "Amazon support agent backend is running."})
            return

        if parsed.path == "/api/evaluate":
            self.send_json({"status": "offline-evaluation", "route": "python -m src.evaluate"})
            return

        if parsed.path in {"/", "/index.html"}:
            self.path = "/index.html"
            return super().do_GET()

        if parsed.path.endswith(".js") or parsed.path.endswith(".css"):
            return super().do_GET()

        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/agent":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8") if length else ""
                payload = json.loads(body) if body.strip() else {}
                message = payload.get("message") or payload.get("text") or payload.get("customer_message") or ""
                if not isinstance(message, str):
                    raise ValueError("message must be a string")
                if not message.strip():
                    raise ValueError("message field is required")
                result = make_agent_response(message)
                self.send_json({"ok": True, "result": result})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, status=400)
            return

        self.send_json({"ok": False, "error": "not found"}, status=404)

    def send_json(self, payload, status=200):
        body = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)


def run_server(host="127.0.0.1", port=PORT):
    server = ThreadingHTTPServer((host, port), AgentHandler)
    print(f"Amazon Support Agent frontend is running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server")
        server.server_close()


if __name__ == "__main__":
    run_server()
