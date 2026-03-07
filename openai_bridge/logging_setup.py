from __future__ import annotations

import logging
import logging.config
import os
from pathlib import Path


def resolve_logs_dir(*, env_var: str, repo_root: Path) -> Path:
    raw = os.getenv(env_var, "custom_bridge/logs")
    path = Path(raw)
    if not path.is_absolute():
        path = repo_root / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_bridge_logging(*, logs_dir: Path, service_name: str) -> None:
    log_file = logs_dir / f"{service_name}.log"

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s %(levelname)s [%(name)s] %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "access": {
                "format": (
                    "%(asctime)s %(levelname)s [%(name)s] "
                    "%(client_addr)s - \"%(request_line)s\" %(status_code)s"
                ),
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.FileHandler",
                "level": "INFO",
                "formatter": "standard",
                "filename": str(log_file),
                "encoding": "utf-8",
            },
            "access_file": {
                "class": "logging.FileHandler",
                "level": "INFO",
                "formatter": "access",
                "filename": str(log_file),
                "encoding": "utf-8",
            },
        },
        "root": {"level": "INFO", "handlers": ["console", "file"]},
        "loggers": {
            "uvicorn": {"level": "INFO", "handlers": ["console", "file"], "propagate": False},
            "uvicorn.error": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["console", "access_file"],
                "propagate": False,
            },
        },
    }
    logging.config.dictConfig(config)
    logging.getLogger(__name__).info("Logging initialized: %s", log_file)
