"""tools/bench.py against a simulated server (no GPU or models)."""
import json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import bench

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert b["temperature"] == 0 and self.headers.get("Authorization") == "Bearer k"
        time.sleep(0.02)
        r = json.dumps({"usage": {"prompt_tokens": 20, "completion_tokens": 10},
                        "stats": {"time_to_first_token": 0.01, "tokens_per_second": 200}}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(r))); self.end_headers(); self.wfile.write(r)

s = HTTPServer(("127.0.0.1", 18999), H); threading.Thread(target=s.serve_forever, daemon=True).start()
fallas = []
if bench.main(["--url", "http://127.0.0.1:18999", "--model", "m", "--runs", "3", "--key", "k"]) != 0: fallas.append("bench ok")
if bench.main(["--url", "http://127.0.0.1:1", "--model", "m", "--runs", "1", "--timeout", "2"]) != 1: fallas.append("bench error")
print("bench: %d fallas" % len(fallas)); sys.exit(1 if fallas else 0)
