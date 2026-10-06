"""Local, read-only browser dashboard for an existing MeshTrain coordinator."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from meshtrain.networking.control import ControlClient, ControlError


def create_app(client: ControlClient) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        yield
        client.http.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])

    @app.get("/")
    def index():
        return FileResponse(Path(__file__).with_name("dashboard.html"), headers={"Cache-Control": "no-store"})

    @app.get("/api/snapshot")
    def snapshot():
        try:
            status = client.status()
            return {"coordinator": client.base, "workers": status["workers"], "jobs": client.jobs()["jobs"]}
        except ControlError as exc:
            raise HTTPException(502, str(exc)) from exc

    return app


def run_dashboard(client: ControlClient, port: int = 8081) -> None:
    import uvicorn

    print(f"MeshTrain dashboard: http://127.0.0.1:{port}  (Ctrl-C to close)", flush=True)
    uvicorn.run(create_app(client), host="127.0.0.1", port=port, log_level="warning")
