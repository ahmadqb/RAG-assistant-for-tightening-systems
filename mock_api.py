"""
Fake manufacturing API for testing the data agent.
Serves the 3 endpoints that data_agent.py calls, on port 8000.

Simulated values: they match docs/machine_configuration.txt and incident #2024-118.
MCH-01 / MCH-02: angle head 3W050 (max 50 Nm), target 40 Nm
MCH-03: angle head 3W090 (max 90 Nm), target 85 Nm
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs



MACHINES = {
    "MCH-01": [
        {"timestamp": "2024-10-20T08:00:00", "torque_nm": 40.1, "status": "running"},
        {"timestamp": "2024-10-20T08:05:00", "torque_nm": 39.8, "status": "running"},
        {"timestamp": "2024-10-20T08:10:00", "torque_nm": 52.4, "status": "error"},
    ],
    "MCH-02": [
        {"timestamp": "2024-10-20T08:00:00", "torque_nm": 40.0, "status": "running"},
        {"timestamp": "2024-10-20T08:05:00", "torque_nm": 39.9, "status": "running"},
    ],
    "MCH-03": [
        {"timestamp": "2024-10-20T08:00:00", "torque_nm": 78.6, "status": "error"},
        {"timestamp": "2024-10-20T08:05:00", "torque_nm": 79.4, "status": "error"},
    ],
}

ERRORS = [
    {"machine_id": "MCH-01", "error_code": "E-460", "count": 1},
    {"machine_id": "MCH-03", "error_code": "E-451", "count": 2},
]


class Handler(BaseHTTPRequestHandler):
    def _send(self, payload, code=200):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        params = parse_qs(url.query)

        if url.path == "/api/all_machines":
            self._send(sorted(MACHINES.keys()))
        elif url.path == "/api/machine_status":
            machine_id = params.get("machine_id", [""])[0]
            self._send(MACHINES.get(machine_id, []))
        elif url.path == "/api/errors":
            self._send(ERRORS)
        else:
            self._send({"error": "not found"}, 404)


if __name__ == "__main__":
    print("Mock API running on http://localhost:8000 (Ctrl+C to stop)")
    HTTPServer(("0.0.0.0", 8000), Handler).serve_forever()