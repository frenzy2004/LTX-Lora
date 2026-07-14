from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable


def safe_console_text(value: Any, *, encoding: str | None = None) -> str:
    target_encoding = encoding or sys.stdout.encoding or "utf-8"
    return str(value).encode(target_encoding, errors="backslashreplace").decode(target_encoding)


def require_key() -> None:
    if not os.getenv("FAL_KEY"):
        raise RuntimeError("FAL_KEY is not set in the process environment")


def upload(path: Path) -> str:
    require_key()
    import fal_client

    return fal_client.upload_file(str(path))


def submit(
    endpoint: str,
    arguments: dict[str, Any],
    on_update: Callable[[Any], None] | None = None,
    on_enqueue: Callable[[str], None] | None = None,
) -> Any:
    require_key()
    import fal_client

    handler = fal_client.submit(endpoint, arguments=arguments)
    if on_enqueue:
        on_enqueue(handler.request_id)
    if on_update:
        for event in handler.iter_events(with_logs=True):
            on_update(event)
    return handler.get()
