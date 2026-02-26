from __future__ import annotations

import uvicorn

from .config import BridgeConfig


def main() -> None:
    cfg = BridgeConfig.from_env()
    uvicorn.run(
        "openai_bridge.server:app",
        host=cfg.host,
        port=cfg.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
