from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable

from .authorization import A2V_ENDPOINT


A2V_QUEUE_URL = "https://queue.fal.run/fal-ai/ltx23-trainer-v2/a2v"


def safe_console_text(value: Any, *, encoding: str | None = None) -> str:
    target_encoding = encoding or sys.stdout.encoding or "utf-8"
    return str(value).encode(target_encoding, errors="backslashreplace").decode(target_encoding)


def resolve_key() -> str:
    key = os.getenv("FAL_KEY")
    if not isinstance(key, str) or not key:
        raise RuntimeError("FAL_KEY is not set in the process environment")
    return key


def require_key() -> str:
    """Backward-compatible name for credential resolution without logging it."""

    return resolve_key()


def upload(path: Path) -> str:
    require_key()
    import fal_client

    return fal_client.upload_file(str(path))


def upload_staged_file(path: Path, key: str) -> str:
    """Upload only a pre-validated private staged path using an already-resolved key."""

    if not isinstance(key, str) or not key:
        raise RuntimeError("FAL_KEY is not set in the process environment")
    if not isinstance(path, Path) or not path.is_file():
        raise ValueError("staged upload path is unavailable")
    import fal_client

    url = fal_client.SyncClient(key=key).upload_file(str(path))
    if not isinstance(url, str) or not url.startswith("https://"):
        raise RuntimeError("provider upload did not return a secure URL")
    return url


def submit(
    endpoint: str,
    arguments: dict[str, Any],
    on_update: Callable[[Any], None] | None = None,
    on_enqueue: Callable[[str], None] | None = None,
) -> Any:
    if endpoint.startswith("fal-ai/ltx23-trainer-v2/"):
        raise RuntimeError("direct LTX trainer submit is disabled; use immutable A2V execution")
    require_key()
    import fal_client

    handler = fal_client.submit(endpoint, arguments=arguments)
    if on_enqueue:
        on_enqueue(handler.request_id)
    if on_update:
        for event in handler.iter_events(with_logs=True):
            on_update(event)
    return handler.get()


def submit_a2v_once(
    endpoint: str,
    body: bytes,
    key: str,
    *,
    transport: Any | None = None,
) -> dict[str, Any]:
    """Issue exactly one non-redirecting raw queue POST for the sealed A2V body."""

    if endpoint != A2V_QUEUE_URL:
        raise ValueError("A2V queue endpoint is fixed")
    if not isinstance(body, bytes) or not body:
        raise ValueError("A2V request body is required")
    if not isinstance(key, str) or not key:
        raise RuntimeError("FAL_KEY is not set in the process environment")

    import httpx

    selected_transport = transport
    if selected_transport is None:
        selected_transport = httpx.HTTPTransport(
            retries=0,
            verify=True,
            trust_env=False,
            http1=True,
            http2=False,
        )
    headers = {
        "Authorization": f"Key {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Fal-No-Retry": "1",
        "x-app-fal-disable-fallback": "true",
        "X-Fal-Store-IO": "0",
    }
    try:
        with httpx.Client(
            transport=selected_transport,
            follow_redirects=False,
            trust_env=False,
            http1=True,
            http2=False,
        ) as client:
            response = client.post(endpoint, content=body, headers=headers)
    except Exception as exc:
        raise RuntimeError("provider submission transport is ambiguous") from exc
    if response.status_code < 200 or response.status_code >= 300:
        raise RuntimeError("provider submission status is ambiguous")
    try:
        acknowledgement = response.json()
    except Exception as exc:
        raise RuntimeError("provider submission acknowledgement is malformed") from exc
    if (
        type(acknowledgement) is not dict
        or type(acknowledgement.get("request_id")) is not str
        or not acknowledgement["request_id"].strip()
    ):
        raise RuntimeError("provider submission acknowledgement is malformed")
    return {"request_id": acknowledgement["request_id"]}
