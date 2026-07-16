import sys
from types import SimpleNamespace

import pytest

from ltx_lora_pilot.fal_api import A2V_ENDPOINT, safe_console_text, submit


def test_safe_console_text_escapes_unsupported_unicode() -> None:
    rendered = safe_console_text("progress: 50% 😀", encoding="cp1252")

    rendered.encode("cp1252")
    assert "progress: 50%" in rendered


def test_submit_persists_request_id_before_streaming_events(monkeypatch) -> None:
    order = []

    class FakeHandle:
        request_id = "request-123"

        def iter_events(self, *, with_logs: bool):
            assert with_logs is True
            order.append("events")
            yield "training 😀"

        def get(self):
            order.append("result")
            return {"ok": True}

    monkeypatch.setenv("FAL_KEY", "test-only")
    monkeypatch.setitem(sys.modules, "fal_client", SimpleNamespace(submit=lambda *_args, **_kwargs: FakeHandle()))

    result = submit(
        "test-endpoint",
        {"input": True},
        on_enqueue=lambda request_id: order.append(f"id:{request_id}"),
        on_update=lambda event: order.append(f"event:{event}"),
    )

    assert result == {"ok": True}
    assert order == ["id:request-123", "events", "event:training 😀", "result"]


def test_legacy_submit_rejects_the_paid_a2v_endpoint_before_client_import(monkeypatch) -> None:
    monkeypatch.setenv("FAL_KEY", "test-only")
    called = []
    monkeypatch.setitem(
        sys.modules,
        "fal_client",
        SimpleNamespace(submit=lambda *_args, **_kwargs: called.append(True)),
    )

    with pytest.raises(RuntimeError, match="immutable A2V"):
        submit(A2V_ENDPOINT, {})

    assert called == []


def test_legacy_submit_rejects_all_ltx_trainer_endpoints_before_client_import(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FAL_KEY", "test-only")
    called = []
    monkeypatch.setitem(
        sys.modules,
        "fal_client",
        SimpleNamespace(submit=lambda *_args, **_kwargs: called.append(True)),
    )

    with pytest.raises(RuntimeError, match="immutable A2V"):
        submit("fal-ai/ltx23-trainer-v2/i2v", {})

    assert called == []
