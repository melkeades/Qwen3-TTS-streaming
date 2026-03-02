from __future__ import annotations

import argparse
import os

import uvicorn

from .custom_config import CustomBridgeConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CustomVoice OpenAI bridge server")
    parser.add_argument(
        "--empty",
        action="store_true",
        help="Start with no preloaded model (select/load later via API or client).",
    )
    args = parser.parse_args()

    if args.empty:
        os.environ["CUSTOM_BRIDGE_START_EMPTY"] = "1"

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
