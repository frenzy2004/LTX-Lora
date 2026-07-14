from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable


def require_key() -> None:
    if not os.getenv("FAL_KEY"):
        raise RuntimeError("FAL_KEY is not set in the process environment")


def upload(path: Path) -> str:
    require_key()
    import fal_client

    return fal_client.upload_file(str(path))


def submit(endpoint: str, arguments: dict[str, Any], on_update: Callable[[Any], None] | None = None) -> Any:
    require_key()
    import fal_client

    handler = fal_client.submit(endpoint, arguments=arguments)
    if on_update:
        for event in handler.iter_events(with_logs=True):
            on_update(event)
    return handler.get()
