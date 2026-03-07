from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from .custom_config import CustomBridgeConfig
from .logging_setup import configure_bridge_logging, resolve_logs_dir


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
    repo_root = Path(__file__).resolve().parent.parent
    logs_dir = resolve_logs_dir(env_var="CUSTOM_BRIDGE_LOGS_DIR", repo_root=repo_root)
    configure_bridge_logging(logs_dir=logs_dir, service_name="custom_bridge")
    uvicorn.run(
        "openai_bridge.custom_server:app",
        host=cfg.host,
        port=cfg.port,
        reload=False,
        log_level="info",
        log_config=None,
    )


if __name__ == "__main__":
    main()
