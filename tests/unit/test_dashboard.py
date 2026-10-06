import httpx
import pytest
from fastapi.testclient import TestClient

from meshtrain.dashboard import create_app
from meshtrain.networking.control import ControlClient


def test_dashboard_reads_live_metrics_and_keeps_token_on_server():
    token = "private-test-cluster-token"
    worker = {"worker_id": "gpu", "status": "BUSY", "backend": "cuda"}
    job = {"job_id": "training-1", "status": "RUNNING", "losses": [[0, 2.1], [1, 1.8]],
           "last_metrics": {"0": {"step": 1, "step_s": 0.1, "comm_s": 0.02}}}
    requests = []

    def upstream(request):
        requests.append(request)
        assert request.headers["X-MeshTrain-Token"] == token
        return httpx.Response(200, json={"workers": [worker]} if request.url.path == "/cluster/status"
                              else {"jobs": [job]})

    control = ControlClient("127.0.0.1:8080", token, transport=httpx.MockTransport(upstream))
    with TestClient(create_app(control), base_url="http://127.0.0.1") as browser:
        page = browser.get("/")
        assert page.status_code == 200 and "Loss curve" in page.text
        assert token not in page.text
        snapshot = browser.get("/api/snapshot")
        assert snapshot.json() == {"coordinator": "http://127.0.0.1:8080", "workers": [worker], "jobs": [job]}
        assert token not in snapshot.text
        assert [(r.method, r.url.path) for r in requests] == [("GET", "/cluster/status"), ("GET", "/jobs")]
        assert browser.post("/api/snapshot").status_code == 405
        assert browser.get("/api/snapshot", headers={"Host": "untrusted.example"}).status_code == 400


@pytest.mark.parametrize("unreachable", [False, True])
def test_coordinator_auth_and_connection_errors_are_reported(unreachable):
    def upstream(request):
        if unreachable:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(401, json={"detail": "invalid or missing cluster token"})

    control = ControlClient("127.0.0.1:8080", "tok", transport=httpx.MockTransport(upstream))
    with TestClient(create_app(control), base_url="http://127.0.0.1") as browser:
        response = browser.get("/api/snapshot")
        assert response.status_code == 502
        assert ("unreachable" if unreachable else "invalid or missing") in response.json()["detail"]
