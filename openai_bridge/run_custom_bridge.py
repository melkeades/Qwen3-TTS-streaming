from __future__ import annotations

import uvicorn

from .custom_config import CustomBridgeConfig


def main() -> None:
    cfg = CustomBridgeConfig.from_env()
    uvicorn.run(
        "openai_bridge.custom_server:app",
        host=cfg.host,
        port=cfg.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()

