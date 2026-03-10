from __future__ import annotations

import logging

from openai_bridge.logging_setup import configure_bridge_logging


def test_uvicorn_access_logging_formats_without_stderr(tmp_path, capsys) -> None:
    previous_raise_exceptions = logging.raiseExceptions
    logging.raiseExceptions = True
    try:
        configure_bridge_logging(logs_dir=tmp_path, service_name="test_bridge")

        logger = logging.getLogger("uvicorn.access")
        logger.info(
            '%s - "%s %s HTTP/%s" %d',
            "127.0.0.1:9999",
            "GET",
            "/favicon.ico",
            "1.1",
            204,
        )

        for handler in logger.handlers:
            handler.flush()
    finally:
        logging.raiseExceptions = previous_raise_exceptions

    captured = capsys.readouterr()
    assert captured.err == ""

    log_text = (tmp_path / "test_bridge.log").read_text(encoding="utf-8")
    assert '127.0.0.1:9999 - "GET /favicon.ico HTTP/1.1" 204 No Content' in log_text
