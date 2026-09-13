import json
import os
import sys
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse
from io import BytesIO

from src.agent import load_agent

ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"
PORT = int(os.environ.get("PORT", "8000"))

agent, _, _ = load_agent()


def make_agent_response(text: str):
    """Return the same structured reply payload used by the Python agent."""
    return agent.run(text)


def _json_payload(payload, status=200):
    """Return a WSGI-style response tuple for compatibility with Vercel-style exports."""
    body = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
    return status, body


def _text_payload(text, status=200):
    body = text.encode("utf-8") if isinstance(text, str) else text
    return status, body


def app(environ, start_response):
    """WSGI-compatible top-level callable expected by Vercel-style Python deploys.

    This preserves the same API surface as the local `server.py` routes while
    providing a standards-compatible export for `app`.
    """
    method = environ.get("REQUEST_METHOD", "GET").upper()
    path = environ.get("PATH_INFO", "/")

    if method == "OPTIONS":
        if path == "/api/agent":
            status, body = _json_payload({"ok": True, "allowed": ["POST"]}, 200)
        else:
            status, body = _json_payload({"ok": True}, 200)
        headers = [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(body))),
            ("Access-Control-Allow-Origin", "*"),
            ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
            ("Access-Control-Allow-Headers", "Content-Type"),
        ]
        start_response(f"{status} OK", headers)
        return [body]

    if method == "GET" and path == "/api/health":
        status, body = _json_payload({"status": "ok", "message": "Amazon support agent backend is running."}, 200)
        headers = [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(body))),
            ("Access-Control-Allow-Origin", "*"),
        ]
        start_response(f"{status} OK", headers)
        return [body]

    if method == "GET" and path == "/api/evaluate":
        status, body = _json_payload({"status": "offline-evaluation", "route": "python -m src.evaluate"}, 200)
        headers = [("Content-Type", "application/json; charset=utf-8"), ("Content-Length", str(len(body)))]
        start_response(f"{status} OK", headers)
        return [body]

    if method == "POST" and path == "/api/agent":
        try:
            length = int(environ.get("CONTENT_LENGTH") or 0)
            raw = environ.get("wsgi.input").read(length) if length else b""
            payload = json.loads(raw.decode("utf-8")) if raw.strip() else {}
            message = payload.get("message") or payload.get("text") or payload.get("customer_message") or ""
            if not isinstance(message, str):
                raise ValueError("message must be a string")
            if not message.strip():
                raise ValueError("message field is required")
            result = make_agent_response(message)
            status, body = _json_payload({"ok": True, "result": result}, 200)
        except Exception as exc:
            status, body = _json_payload({"ok": False, "error": str(exc)}, 400)
        headers = [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(body))),
            ("Access-Control-Allow-Origin", "*"),
        ]
        start_response(f"{status} OK", headers)
        return [body]

    # If deployed under a serverless entry point, default to an empty JSON not-found
    # response rather than crashing during import-time route inspection.
    status, body = _json_payload({"ok": False, "error": "not found"}, 404)
    headers = [
        ("Content-Type", "application/json; charset=utf-8"),
        ("Content-Length", str(len(body))),
        ("Access-Control-Allow-Origin", "*"),
    ]
    start_response(f"{status} OK", headers)
    return [body]


# Keep the same public name expected by platform tooling and local references.
application = app


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
