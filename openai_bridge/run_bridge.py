from __future__ import annotations

from pathlib import Path

import uvicorn

from .config import BridgeConfig
from .logging_setup import configure_bridge_logging, resolve_logs_dir


def main() -> None:
    cfg = BridgeConfig.from_env()
    repo_root = Path(__file__).resolve().parent.parent
    logs_dir = resolve_logs_dir(env_var="BRIDGE_LOGS_DIR", repo_root=repo_root)
    configure_bridge_logging(logs_dir=logs_dir, service_name="bridge")
    uvicorn.run(
        "openai_bridge.server:app",
        host=cfg.host,
        port=cfg.port,
        reload=False,
        log_level="info",
        log_config=None,
    )


if __name__ == "__main__":
    main()
