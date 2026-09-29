"""Serveur local sans dépendance web ni CDN. Lecture/analyse uniquement."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
import numpy as np
import pandas as pd

STATIC = Path(__file__).parent / "static"


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def create_server(service, port=8765):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return  # Les exceptions sont retournées explicitement en JSON à l'interface.

        def send(self, status, payload, content_type="application/json; charset=utf-8"):
            if isinstance(payload, (dict, list)):
                payload = json.dumps(clean(payload), ensure_ascii=False, allow_nan=False).encode()
            elif isinstance(payload, str):
                payload = payload.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(payload)

        def valid_host(self):
            return self.headers.get("Host") in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}")

        def params(self):
            query = parse_qs(urlsplit(self.path).query)
            symbol = query.get("symbol", [service.config.symbols[0]])[0]
            timeframe = int(query.get("timeframe", [5])[0])
            minute = query.get("minute", [None])[0]
            leverage = query.get("leverage", [None])[0]
            if timeframe not in (5, 10, 15):
                raise ValueError("Unité de temps invalide.")
            return symbol, timeframe, int(minute) if minute else None, float(leverage) if leverage else None

        def do_GET(self):
            if not self.valid_host():
                return self.send(403, {"error": "Accès local uniquement."})
            path = urlsplit(self.path).path
            try:
                if path == "/api/config":
                    return self.send(200, dict(mode=service.mode, config=service.config.to_dict(), version="8.0"))
                if path.startswith("/api/"):
                    symbol, timeframe, minute, leverage = self.params()
                    if path == "/api/scan":
                        return self.send(200, dict(items=service.scan(timeframe, minute)))
                    if path == "/api/analysis":
                        return self.send(200, service.analyze(symbol, timeframe, minute, leverage))
                    if path == "/api/model":
                        return self.send(200, service.model_estimate(symbol, timeframe, minute))
                    return self.send(404, {"error": "Route inconnue."})
                names = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
                if path not in names:
                    return self.send(404, "Page inconnue", "text/plain; charset=utf-8")
                file = STATIC / names[path]
                return self.send(200, file.read_bytes(), (mimetypes.guess_type(file.name)[0] or "text/plain") + "; charset=utf-8")
            except (ValueError, FileNotFoundError, RuntimeError) as exc:
                self.send(400, {"error": str(exc)})
            except Exception as exc:
                self.send(500, {"error": f"{type(exc).__name__}: {exc}"})

        def do_POST(self):
            origin = self.headers.get("Origin")
            allowed_origins = (None, f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}")
            if not self.valid_host() or origin not in allowed_origins:
                return self.send(403, {"error": "Origine locale requise."})
            try:
                if urlsplit(self.path).path != "/api/chat":
                    return self.send(404, {"error": "Route inconnue."})
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 8192:
                    raise ValueError("Message vide ou trop long.")
                body = json.loads(self.rfile.read(length))
                context = service.analyze(str(body["symbol"]), int(body.get("timeframe", 5)),
                                          body.get("minute"), body.get("leverage"))
                answer = service.assistant.answer(str(body.get("question", ""))[:2000], context)
                self.send(200, dict(answer=answer, provider="règles locales", orders_enabled=False))
            except (ValueError, KeyError, RuntimeError) as exc:
                self.send(400, {"error": str(exc)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(service, port=8765):
    server = create_server(service, port)
    print(f"Trading AI V8 — {service.mode}\nOuvrir http://127.0.0.1:{server.server_port}\nCtrl+C pour arrêter.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
