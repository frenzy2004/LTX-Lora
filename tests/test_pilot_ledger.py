from __future__ import annotations

from contextlib import closing, contextmanager
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from decimal import Decimal
from typing import Any, Callable

import pytest

import ltx_lora_pilot.pilot_ledger as pilot_ledger
from ltx_lora_pilot.artifacts import canonical_json_bytes
from ltx_lora_pilot.pilot_ledger import (
    LedgerPreflightSnapshot,
    PilotLedger,
    Reservation,
    migrate_legacy_ledger,
)


ROOT = Path(__file__).resolve().parents[1]
MIGRATION_SCRIPT = ROOT / "scripts" / "migrate_budget_ledger.py"
PILOT_ID = "pilot_00000000000040008000000000000001"
LEDGER_ID = "ledger_00000000000040008000000000000002"
MIGRATION_ID = "migration_00000000000040008000000000000003"
FIXED_TIME = "2026-07-15T03:00:00Z"
AMOUNTS = ["1.2000", "0.1099", "0.1099", "0.3272", "0.3272", "1.4667"]
STATES = ["consumed", "consumed", "consumed", "consumed", "reserved", "consumed"]


def _typed_id(prefix: str, number: int) -> str:
    return f"{prefix}_{number:012x}40008{number:015x}"


def _source_id(number: int) -> str:
    return f"00000000-0000-4000-8000-{number:012x}"


def _bundle_id(number: int) -> str:
    return hashlib.sha256(f"synthetic-history-{number}".encode("ascii")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    try:
        content = canonical_json_bytes(value)
    except TypeError:
        content = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    path.write_bytes(content)


def _fixture_documents(
    *,
    amounts: list[Any] | None = None,
    states: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected_amounts = list(AMOUNTS if amounts is None else amounts)
    selected_states = list(STATES if states is None else states)
    source_entries = []
    manifest_entries = []
    for index, (amount, state) in enumerate(
        zip(selected_amounts, selected_states, strict=True), start=1
    ):
        source_entry_id = _source_id(index)
        source_entries.append(
            {
                "id": source_entry_id,
                "label": f"PRIVATE synthetic legacy label {index}",
                "amount_usd": amount,
                "status": state,
                "created_at": 1_700_000_000 + index,
                **({"finalized_at": 1_700_000_100 + index} if state != "reserved" else {}),
            }
        )
        manifest_entries.append(
            {
                "source_entry_id": source_entry_id,
                "reservation_id": _typed_id("reservation", index + 10),
                "bundle_id": _bundle_id(index),
                "execution_id": _typed_id("exec", index + 20),
                "amount_usd": amount,
                "state": state,
            }
        )
    source = {"cap_usd": "12.0000", "entries": source_entries}
    source_content = canonical_json_bytes(source)
    manifest = {
        "schema_version": "pilot-budget-migration-v1",
        "pilot_id": PILOT_ID,
        "ledger_id": LEDGER_ID,
        "migration_id": MIGRATION_ID,
        "cap_usd": "12.0000",
        "source_ledger_sha256": hashlib.sha256(source_content).hexdigest(),
        "created_at_utc": FIXED_TIME,
        "entries": manifest_entries,
    }
    return source, manifest


def _write_fixture(
    tmp_path: Path,
    *,
    mutate_source: Callable[[dict[str, Any]], None] | None = None,
    mutate_manifest: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Path, Path, Path]:
    source, manifest = _fixture_documents()
    if mutate_source is not None:
        mutate_source(source)
    if mutate_manifest is not None:
        mutate_manifest(manifest)
    source_path = tmp_path / "legacy-private-budget.json"
    manifest_path = tmp_path / "reviewed-migration.json"
    ledger_path = tmp_path / "pilot.sqlite3"
    _write_json(source_path, source)
    _write_json(manifest_path, manifest)
    return source_path, manifest_path, ledger_path


def _migrate_fixture(tmp_path: Path) -> tuple[PilotLedger, Path, Path, Path]:
    source_path, manifest_path, ledger_path = _write_fixture(tmp_path)
    ledger = migrate_legacy_ledger(source_path, manifest_path, ledger_path)
    return ledger, source_path, manifest_path, ledger_path


def test_preflight_snapshot_is_one_verified_read_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, _, _, _ = _migrate_fixture(tmp_path)
    bundle_id = _bundle_id(100)
    execution_id = _typed_id("exec", 100)
    expected_head = ledger.head_hash
    original_read_connection = pilot_ledger._read_connection
    observed = {"read_transactions": 0}

    @contextmanager
    def counted_read_connection(path: Path) -> Any:
        observed["read_transactions"] += 1
        with original_read_connection(path) as connection:
            yield connection

    monkeypatch.setattr(pilot_ledger, "_read_connection", counted_read_connection)

    def unexpected_component_snapshot() -> Any:
        raise AssertionError("separate ledger accessor was called")

    monkeypatch.setattr(ledger, "_snapshot", unexpected_component_snapshot)

    snapshot = ledger.preflight_snapshot(bundle_id, execution_id)

    assert observed == {"read_transactions": 1}
    assert snapshot == LedgerPreflightSnapshot(
        pilot_id=PILOT_ID,
        ledger_id=LEDGER_ID,
        bundle_id=bundle_id,
        execution_id=execution_id,
        head_sha256=expected_head,
        committed_usd="3.5409",
        remaining_usd="8.4591",
        replay_detected=False,
    )


def test_ledger_preflight_snapshot_is_a_public_export() -> None:
    assert "LedgerPreflightSnapshot" in pilot_ledger.__all__


@pytest.mark.parametrize("suffix", ["-journal", "-wal", "-shm"])
def test_preflight_snapshot_rejects_sidecar_before_opening_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    ledger, _, _, ledger_path = _migrate_fixture(tmp_path)
    sidecar = Path(f"{ledger_path}{suffix}")
    sidecar.write_bytes(b"synthetic untrusted sidecar")

    def unexpected_open(path: Path) -> Any:
        del path
        raise AssertionError("database opened before sidecar rejection")

    monkeypatch.setattr(pilot_ledger, "_read_connection", unexpected_open)

    with pytest.raises(ValueError, match="ledger sidecar exists"):
        ledger.preflight_snapshot(_bundle_id(100), _typed_id("exec", 100))

    assert sidecar.read_bytes() == b"synthetic untrusted sidecar"


@pytest.mark.parametrize("suffix", ["-journal", "-wal", "-shm"])
def test_preflight_snapshot_rechecks_sidecar_absence_before_returning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    ledger, _, _, ledger_path = _migrate_fixture(tmp_path)
    sidecar = Path(f"{ledger_path}{suffix}")
    original_verify = pilot_ledger._verify_connection

    def verify_then_create_sidecar(*args: Any, **kwargs: Any) -> Any:
        snapshot = original_verify(*args, **kwargs)
        sidecar.write_bytes(b"synthetic sidecar race")
        return snapshot

    monkeypatch.setattr(pilot_ledger, "_verify_connection", verify_then_create_sidecar)

    with pytest.raises(ValueError, match="ledger sidecar exists"):
        ledger.preflight_snapshot(_bundle_id(100), _typed_id("exec", 100))

    assert sidecar.read_bytes() == b"synthetic sidecar race"


def test_preflight_snapshot_rechecks_sidecars_after_retaining_file_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, _, _, ledger_path = _migrate_fixture(tmp_path)
    sidecar = Path(f"{ledger_path}-journal")
    original_anchor = pilot_ledger._preflight_file_anchor

    @contextmanager
    def anchor_then_create_sidecar(path: Path) -> Any:
        with original_anchor(path) as values:
            sidecar.write_bytes(b"synthetic anchor race")
            yield values

    def unexpected_open(path: Path) -> Any:
        del path
        raise AssertionError("database opened after sidecar appeared")

    monkeypatch.setattr(
        pilot_ledger,
        "_preflight_file_anchor",
        anchor_then_create_sidecar,
    )
    monkeypatch.setattr(pilot_ledger, "_read_connection", unexpected_open)

    with pytest.raises(ValueError, match="ledger sidecar exists"):
        ledger.preflight_snapshot(_bundle_id(100), _typed_id("exec", 100))

    assert sidecar.read_bytes() == b"synthetic anchor race"


def test_preflight_snapshot_rejects_database_metadata_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, _, _, ledger_path = _migrate_fixture(tmp_path)
    original_verify = pilot_ledger._verify_connection

    def verify_then_touch_database(*args: Any, **kwargs: Any) -> Any:
        snapshot = original_verify(*args, **kwargs)
        current = ledger_path.stat()
        os.utime(
            ledger_path,
            ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000_000),
        )
        return snapshot

    monkeypatch.setattr(pilot_ledger, "_verify_connection", verify_then_touch_database)

    with pytest.raises(ValueError, match="ledger changed during verification"):
        ledger.preflight_snapshot(_bundle_id(100), _typed_id("exec", 100))


def test_preflight_snapshot_rejects_same_inode_database_byte_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, _, _, ledger_path = _migrate_fixture(tmp_path)
    original_read_connection = pilot_ledger._read_connection
    original_identity = (ledger_path.stat().st_dev, ledger_path.stat().st_ino)

    @contextmanager
    def mutate_after_read_transaction(path: Path) -> Any:
        with original_read_connection(path) as connection:
            yield connection
        current = path.stat()
        with path.open("r+b") as handle:
            handle.seek(-1, os.SEEK_END)
            final_byte = handle.read(1)
            handle.seek(-1, os.SEEK_END)
            handle.write(bytes([final_byte[0] ^ 1]))
        os.utime(path, ns=(current.st_atime_ns, current.st_mtime_ns))

    monkeypatch.setattr(
        pilot_ledger,
        "_read_connection",
        mutate_after_read_transaction,
    )

    with pytest.raises(ValueError, match="ledger changed during verification"):
        ledger.preflight_snapshot(_bundle_id(100), _typed_id("exec", 100))

    assert (ledger_path.stat().st_dev, ledger_path.stat().st_ino) == original_identity


def test_preflight_snapshot_rejects_path_replacement_after_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, _, _, ledger_path = _migrate_fixture(tmp_path)
    original_read_connection = pilot_ledger._read_connection
    original_stat = Path.stat
    observed = {"replacement_visible": False}

    @contextmanager
    def replace_after_read_transaction(path: Path) -> Any:
        with original_read_connection(path) as connection:
            yield connection
        observed["replacement_visible"] = True

    def stat_with_replaced_identity(path: Path, *args: Any, **kwargs: Any) -> os.stat_result:
        result = original_stat(path, *args, **kwargs)
        if path == ledger_path and observed["replacement_visible"]:
            values = list(result)
            values[1] += 1
            return os.stat_result(values)
        return result

    monkeypatch.setattr(
        pilot_ledger,
        "_read_connection",
        replace_after_read_transaction,
    )
    monkeypatch.setattr(Path, "stat", stat_with_replaced_identity)

    with pytest.raises(ValueError, match="ledger changed during verification"):
        ledger.preflight_snapshot(_bundle_id(100), _typed_id("exec", 100))

    assert observed == {"replacement_visible": True}


@pytest.mark.parametrize("replay_field", ["bundle", "execution"])
def test_preflight_snapshot_reports_replay_without_private_row_details(
    tmp_path: Path,
    replay_field: str,
) -> None:
    ledger, _, _, _ = _migrate_fixture(tmp_path)
    reserved = ledger.reserve(
        _bundle_id(100),
        _typed_id("exec", 100),
        Decimal("0.1000"),
    )
    bundle_id = reserved.bundle_id if replay_field == "bundle" else _bundle_id(101)
    execution_id = (
        reserved.execution_id
        if replay_field == "execution"
        else _typed_id("exec", 101)
    )

    snapshot = ledger.preflight_snapshot(bundle_id, execution_id)

    assert snapshot.replay_detected is True
    assert snapshot.bundle_id == bundle_id
    assert snapshot.execution_id == execution_id
    assert not hasattr(snapshot, "reservation_id")
    assert not hasattr(snapshot, "state")


@pytest.mark.parametrize("wrong_identity", ["pilot", "ledger"])
def test_preflight_snapshot_rejects_wrong_open_identity(
    tmp_path: Path,
    wrong_identity: str,
) -> None:
    _, _, _, ledger_path = _migrate_fixture(tmp_path)
    wrong_pilot = _typed_id("pilot", 900)
    wrong_ledger = _typed_id("ledger", 901)
    ledger = PilotLedger(
        ledger_path,
        wrong_pilot if wrong_identity == "pilot" else PILOT_ID,
        wrong_ledger if wrong_identity == "ledger" else LEDGER_ID,
    )

    with pytest.raises(ValueError, match=f"{wrong_identity} identity mismatch"):
        ledger.preflight_snapshot(_bundle_id(100), _typed_id("exec", 100))


def test_preflight_snapshot_rejects_corrupt_event_chain(tmp_path: Path) -> None:
    ledger, _, _, ledger_path = _migrate_fixture(tmp_path)
    recreate_trigger = next(
        statement
        for statement in pilot_ledger.TRIGGER_STATEMENTS
        if "CREATE TRIGGER events_no_update" in statement
    )
    with closing(sqlite3.connect(ledger_path)) as connection, connection:
        connection.execute("DROP TRIGGER events_no_update")
        connection.execute(
            "UPDATE events SET event_hash = ? WHERE event_index = 1",
            ("0" * 64,),
        )
        connection.execute(recreate_trigger)

    with pytest.raises(ValueError, match="event chain mismatch"):
        ledger.preflight_snapshot(_bundle_id(100), _typed_id("exec", 100))


def test_reserve_training_requires_expected_head(tmp_path: Path) -> None:
    ledger, _, _, _ = _migrate_fixture(tmp_path)

    with pytest.raises(TypeError, match="expected_head_sha256"):
        ledger.reserve_training(  # type: ignore[call-arg]
            _bundle_id(100),
            _typed_id("exec", 100),
            Decimal("1.0000"),
        )

    assert ledger.remaining() == Decimal("8.4591")


@pytest.mark.parametrize("expected_head", ["not-a-hash", "A" * 64])
def test_reserve_training_rejects_malformed_head_before_write_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    expected_head: str,
) -> None:
    ledger, _, _, _ = _migrate_fixture(tmp_path)

    def unexpected_write_connection(path: Path) -> Any:
        del path
        raise AssertionError("write connection opened before head validation")

    monkeypatch.setattr(
        pilot_ledger,
        "_write_connection",
        unexpected_write_connection,
    )

    with pytest.raises(ValueError, match="expected ledger head must be a lowercase SHA-256"):
        ledger.reserve_training(
            _bundle_id(100),
            _typed_id("exec", 100),
            Decimal("1.0000"),
            expected_head_sha256=expected_head,
        )


def test_reserve_training_rejects_changed_ledger_head_without_rows(
    tmp_path: Path,
) -> None:
    ledger, _, _, ledger_path = _migrate_fixture(tmp_path)
    bundle_id = _bundle_id(100)
    execution_id = _typed_id("exec", 100)
    snapshot = ledger.preflight_snapshot(bundle_id, execution_id)
    ledger.reserve(
        _bundle_id(101),
        _typed_id("exec", 101),
        Decimal("0.1000"),
    )

    with pytest.raises(RuntimeError, match="ledger head changed after approval"):
        ledger.reserve_training(
            bundle_id,
            execution_id,
            Decimal("1.0000"),
            expected_head_sha256=snapshot.head_sha256,
        )

    with closing(sqlite3.connect(ledger_path)) as connection:
        reservation = connection.execute(
            "SELECT 1 FROM reservations WHERE bundle_id = ? OR execution_id = ?",
            (bundle_id, execution_id),
        ).fetchone()
    assert reservation is None


def test_reserve_training_checks_head_after_begin_and_verify_before_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, _, _, ledger_path = _migrate_fixture(tmp_path)
    bundle_id = _bundle_id(100)
    execution_id = _typed_id("exec", 100)
    snapshot = ledger.preflight_snapshot(bundle_id, execution_id)
    ledger.reserve(
        _bundle_id(101),
        _typed_id("exec", 101),
        Decimal("0.1000"),
    )
    statements: list[str] = []
    original_connect = sqlite3.connect

    def traced_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        connection = original_connect(*args, **kwargs)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(pilot_ledger.sqlite3, "connect", traced_connect)

    with pytest.raises(RuntimeError, match="ledger head changed after approval"):
        ledger.reserve_training(
            bundle_id,
            execution_id,
            Decimal("1.0000"),
            expected_head_sha256=snapshot.head_sha256,
        )

    normalized = [" ".join(statement.upper().split()) for statement in statements]
    begin_index = normalized.index("BEGIN IMMEDIATE")
    verify_index = normalized.index("PRAGMA INTEGRITY_CHECK")
    assert begin_index < verify_index
    assert not any(statement.startswith("INSERT ") for statement in normalized)
    assert all(
        not Path(f"{ledger_path}{suffix}").exists()
        for suffix in ("-journal", "-wal", "-shm")
    )


def _install_commit_then_disconnect(
    monkeypatch: pytest.MonkeyPatch,
    *,
    readback_close_raises: bool = False,
) -> dict[str, Any]:
    original_connect = sqlite3.connect
    observed: dict[str, Any] = {
        "write_connections": 0,
        "commit_applied": False,
        "write_connection_unusable": False,
        "fresh_readback_opened": False,
        "exact_event_readback": False,
        "exact_reservation_readback": False,
        "readback_close_called": False,
        "readback_close_raised": False,
    }

    class CommitThenDisconnectConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self._connection = connection

        @property
        def in_transaction(self) -> bool:
            return False

        def __getattr__(self, name: str) -> Any:
            return getattr(self._connection, name)

        def commit(self) -> None:
            self._connection.commit()
            observed["commit_applied"] = True
            self._connection.close()
            try:
                self._connection.execute("SELECT 1")
            except sqlite3.ProgrammingError:
                observed["write_connection_unusable"] = True
            else:
                raise AssertionError("write connection remained usable after close")
            raise sqlite3.OperationalError("synthetic write connection loss after commit")

    class TrackedReadbackConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self._connection = connection

        def __getattr__(self, name: str) -> Any:
            return getattr(self._connection, name)

        def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
            normalized = " ".join(sql.split()).casefold()
            if "from events where event_id = ?" in normalized:
                observed["exact_event_readback"] = True
            if "from reservations where reservation_id = ?" in normalized:
                observed["exact_reservation_readback"] = True
            return self._connection.execute(sql, *args, **kwargs)

        def close(self) -> None:
            observed["readback_close_called"] = True
            self._connection.close()
            if readback_close_raises:
                observed["readback_close_raised"] = True
                raise OSError("synthetic write-effect readback close failure")

    def connect(*args: Any, **kwargs: Any) -> Any:
        connection = original_connect(*args, **kwargs)
        target = args[0] if args else kwargs.get("database")
        if isinstance(target, str) and target.endswith("?mode=rw"):
            observed["write_connections"] += 1
            if observed["write_connections"] == 1:
                return CommitThenDisconnectConnection(connection)
        if (
            observed["commit_applied"]
            and not observed["fresh_readback_opened"]
            and isinstance(target, str)
            and target.endswith("?mode=ro")
        ):
            observed["fresh_readback_opened"] = True
            return TrackedReadbackConnection(connection)
        return connection

    monkeypatch.setattr(pilot_ledger.sqlite3, "connect", connect)
    return observed


def _last_event_row(ledger_path: Path, reservation_id: str) -> tuple[Any, ...]:
    with closing(sqlite3.connect(ledger_path)) as connection:
        row = connection.execute(
            """
            SELECT event_id, from_state, to_state, amount_usd,
                   reason_code, evidence_sha256
            FROM events
            WHERE reservation_id = ?
            ORDER BY event_index DESC
            LIMIT 1
            """,
            (reservation_id,),
        ).fetchone()
    assert row is not None
    return row


def _event_count(ledger_path: Path) -> int:
    with closing(sqlite3.connect(ledger_path)) as connection:
        row = connection.execute("SELECT COUNT(*) FROM events").fetchone()
    assert row is not None
    return row[0]


def _prepare_write_case(tmp_path: Path, operation: str) -> dict[str, Any]:
    ledger, _, _, ledger_path = _migrate_fixture(tmp_path)
    bundle_id = _bundle_id(100)
    execution_id = _typed_id("exec", 100)
    amount = Decimal("0.1000")
    evidence = hashlib.sha256(b"synthetic exact reconciliation evidence").hexdigest()
    reservation: Reservation | None = None
    if operation == "reserve":
        call: Callable[[], Any] = lambda: ledger.reserve(
            bundle_id,
            execution_id,
            amount,
        )
        expected_from_state = None
        expected_state = "reserved"
        expected_reason = "reservation_created"
        expected_evidence = None
    else:
        reservation = ledger.reserve(bundle_id, execution_id, amount)
        if operation == "submit_started":
            ledger.transition(reservation.id, "uploading")
            call = lambda: ledger.transition(reservation.id, "submit_started")
            expected_from_state = "uploading"
            expected_state = "submit_started"
            expected_reason = "state_transition"
            expected_evidence = None
        elif operation == "release":
            ledger.transition(reservation.id, "uploading")
            call = lambda: ledger.release_pre_submit(
                reservation.id,
                "synthetic neutral release",
            )
            expected_from_state = "uploading"
            expected_state = "released"
            expected_reason = "pre_submit_release"
            expected_evidence = None
        elif operation == "reconcile":
            for state in ("uploading", "submit_started", "submitted"):
                ledger.transition(reservation.id, state)
            call = lambda: ledger.reconcile(
                reservation.id,
                "consumed",
                evidence_sha256=evidence,
            )
            expected_from_state = "submitted"
            expected_state = "consumed"
            expected_reason = "provider_reconciliation"
            expected_evidence = evidence
        else:
            raise AssertionError("unknown synthetic write case")
    return {
        "ledger": ledger,
        "ledger_path": ledger_path,
        "bundle_id": bundle_id,
        "execution_id": execution_id,
        "amount": amount,
        "reservation": reservation,
        "call": call,
        "expected_from_state": expected_from_state,
        "expected_state": expected_state,
        "expected_reason": expected_reason,
        "expected_evidence": expected_evidence,
        "events_before": _event_count(ledger_path),
    }


def _rewrite_latest_event_effect(connection: sqlite3.Connection) -> None:
    _drop_immutability_triggers(connection)
    row = connection.execute(
        """
        SELECT event_index, event_id, reservation_id, from_state, to_state,
               amount_usd, previous_hash, reason_code, evidence_sha256
        FROM events ORDER BY event_index DESC LIMIT 1
        """
    ).fetchone()
    assert row is not None
    (
        event_index,
        event_id,
        reservation_id,
        from_state,
        to_state,
        amount_usd,
        previous_hash,
        reason_code,
        evidence_sha256,
    ) = row
    replacement_created_at = "2000-01-01T00:00:00Z"
    replacement_hash = pilot_ledger._event_hash(
        event_id=event_id,
        reservation_id=reservation_id,
        from_state=from_state,
        to_state=to_state,
        amount_usd=amount_usd,
        created_at_utc=replacement_created_at,
        previous_hash=previous_hash,
        reason_code=reason_code,
        evidence_sha256=evidence_sha256,
    )
    connection.execute(
        "UPDATE events SET created_at_utc = ?, event_hash = ? WHERE event_index = ?",
        (replacement_created_at, replacement_hash, event_index),
    )
    for statement in pilot_ledger.TRIGGER_STATEMENTS:
        connection.execute(statement)


def _install_write_commit_fault(
    monkeypatch: pytest.MonkeyPatch,
    ledger_path: Path,
    *,
    durable_effect: bool,
    close_after_commit: bool,
    reported_in_transaction: bool,
    readback_fault: str | None = None,
    readback_close_raises: bool = False,
    commit_exception: Exception | None = None,
) -> dict[str, Any]:
    original_connect = sqlite3.connect
    primary_commit_exception = (
        commit_exception
        if commit_exception is not None
        else sqlite3.OperationalError(
            "synthetic primary commit acknowledgement error"
        )
    )
    observed: dict[str, Any] = {
        "write_connections": 0,
        "commit_calls": 0,
        "rollback_calls": 0,
        "write_close_calls": 0,
        "write_connection_unusable": False,
        "in_transaction_reads": 0,
        "readback_attempts": 0,
        "exact_event_readback": False,
        "exact_reservation_readback": False,
        "readback_close_calls": 0,
        "readback_close_raised": False,
        "identity_swap_attempted": False,
        "sequence": [],
    }

    class FaultedWriteConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self._connection = connection

        @property
        def in_transaction(self) -> bool:
            observed["in_transaction_reads"] += 1
            return reported_in_transaction

        def __getattr__(self, name: str) -> Any:
            return getattr(self._connection, name)

        def commit(self) -> None:
            observed["commit_calls"] += 1
            if durable_effect:
                self._connection.commit()
            if close_after_commit:
                self._connection.close()
                try:
                    self._connection.execute("SELECT 1")
                except sqlite3.ProgrammingError:
                    observed["write_connection_unusable"] = True
                else:
                    raise AssertionError("write connection remained usable after close")
            observed["sequence"].append("commit_exception")
            raise primary_commit_exception

        def rollback(self) -> None:
            observed["rollback_calls"] += 1
            observed["sequence"].append("rollback")
            self._connection.rollback()

        def close(self) -> None:
            observed["write_close_calls"] += 1
            observed["sequence"].append("write_close")
            self._connection.close()

    class TrackedReadbackConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self._connection = connection

        def __getattr__(self, name: str) -> Any:
            return getattr(self._connection, name)

        def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
            normalized = " ".join(sql.split()).casefold()
            if "from events where event_id = ?" in normalized:
                observed["exact_event_readback"] = True
            if "from reservations where reservation_id = ?" in normalized:
                observed["exact_reservation_readback"] = True
            return self._connection.execute(sql, *args, **kwargs)

        def close(self) -> None:
            observed["readback_close_calls"] += 1
            self._connection.close()
            if readback_close_raises:
                observed["readback_close_raised"] = True
                raise OSError("synthetic exact readback close failure")

    def prepare_readback_fault() -> None:
        if readback_fault == "integrity":
            with closing(original_connect(ledger_path)) as tamper, tamper:
                _drop_immutability_triggers(tamper)
        elif readback_fault == "event_mismatch":
            with closing(original_connect(ledger_path)) as tamper, tamper:
                _rewrite_latest_event_effect(tamper)
        elif readback_fault == "reservation_mismatch":
            with closing(original_connect(ledger_path)) as tamper, tamper:
                _drop_immutability_triggers(tamper)
                tamper.execute(
                    """
                    UPDATE reservations SET created_at_utc = ?
                    WHERE reservation_id = (
                        SELECT reservation_id FROM reservations
                        WHERE migration_id IS NULL ORDER BY rowid DESC LIMIT 1
                    )
                    """,
                    ("2000-01-01T00:00:00Z",),
                )
                for statement in pilot_ledger.TRIGGER_STATEMENTS:
                    tamper.execute(statement)
        elif readback_fault == "identity_swap":
            observed["identity_swap_attempted"] = True
            alias = ledger_path.with_name(f"{ledger_path.name}.anchor")
            replacement = ledger_path.with_name(f"{ledger_path.name}.replacement")
            os.link(ledger_path, alias)
            replacement.write_bytes(ledger_path.read_bytes())
            try:
                os.replace(replacement, ledger_path)
            except PermissionError:
                pass

    def connect(*args: Any, **kwargs: Any) -> Any:
        target = args[0] if args else kwargs.get("database")
        if (
            isinstance(target, str)
            and target.endswith("?mode=ro")
            and observed["commit_calls"]
            and observed["readback_attempts"] == 0
        ):
            observed["readback_attempts"] += 1
            observed["sequence"].append("readback_open")
            if readback_fault == "open":
                raise sqlite3.OperationalError("synthetic exact readback open failure")
            prepare_readback_fault()
            connection = original_connect(*args, **kwargs)
            return TrackedReadbackConnection(connection)
        connection = original_connect(*args, **kwargs)
        if isinstance(target, str) and target.endswith("?mode=rw"):
            observed["write_connections"] += 1
            if observed["write_connections"] == 1:
                return FaultedWriteConnection(connection)
        return connection

    monkeypatch.setattr(pilot_ledger.sqlite3, "connect", connect)
    return observed


def _assert_write_case_effect(case: dict[str, Any], result: Any) -> Reservation:
    reservation = result if case["reservation"] is None else case["reservation"]
    assert isinstance(reservation, Reservation)
    assert _event_count(case["ledger_path"]) == case["events_before"] + 1
    assert case["ledger"].state(reservation.id) == case["expected_state"]
    event = _last_event_row(case["ledger_path"], reservation.id)
    assert event[0].startswith("event_")
    assert event[1:] == (
        case["expected_from_state"],
        case["expected_state"],
        "0.1000",
        case["expected_reason"],
        case["expected_evidence"],
    )
    return reservation


def _drop_immutability_triggers(connection: sqlite3.Connection) -> None:
    names = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'trigger' ORDER BY name"
    ).fetchall()
    for (name,) in names:
        escaped = name.replace('"', '""')
        connection.execute(f'DROP TRIGGER "{escaped}"')


def _reservation_worker(
    ledger_path: str,
    bundle_id: str,
    execution_id: str,
    ready: Any,
    start: Any,
    results: Any,
) -> None:
    try:
        ledger = PilotLedger.open_existing(
            Path(ledger_path),
            PILOT_ID,
            expected_ledger_id=LEDGER_ID,
        )
        ready.put("ready")
        if not start.wait(20):
            results.put(("timeout",))
            return
        reservation = ledger.reserve(
            bundle_id,
            execution_id,
            Decimal("6.0000"),
        )
        results.put(("reserved", reservation.id))
    except Exception as exc:
        results.put(("rejected", type(exc).__name__, str(exc)))


def _post_snapshot_commit_worker(
    ledger_path: str,
    ready: Any,
    start: Any,
    done: Any,
    results: Any,
) -> None:
    try:
        ledger = PilotLedger.open_existing(
            Path(ledger_path),
            PILOT_ID,
            expected_ledger_id=LEDGER_ID,
        )
        ready.set()
        if not start.wait(20):
            results.put(("timeout",))
            return
        reservation = ledger.reserve(
            _bundle_id(300),
            _typed_id("exec", 300),
            Decimal("0.1000"),
        )
        results.put(("reserved", reservation.id))
    except Exception as exc:
        results.put(("rejected", type(exc).__name__, str(exc)))
    finally:
        done.set()


def _migration_worker(
    source_path: str,
    manifest_path: str,
    ledger_path: str,
    ready: Any,
    start: Any,
    results: Any,
) -> None:
    ready.put("ready")
    if not start.wait(20):
        results.put(("timeout",))
        return
    try:
        ledger = migrate_legacy_ledger(
            Path(source_path),
            Path(manifest_path),
            Path(ledger_path),
        )
        results.put(("migrated", ledger.remaining().to_eng_string()))
    except Exception as exc:
        results.put(("rejected", type(exc).__name__, str(exc)))


def _windows_hold_without_delete_share(
    temporary_path: str,
    destination_path: str,
    ready: Any,
    release: Any,
    results: Any,
) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    invalid_handle = ctypes.c_void_p(-1).value
    handles = []
    try:
        for database_path in (temporary_path, destination_path):
            handle = create_file(
                database_path,
                0x80000000,
                0x00000001 | 0x00000002,
                None,
                3,
                0x00000080,
                None,
            )
            if handle == invalid_handle:
                results.put(("error", ctypes.get_last_error()))
                ready.set()
                return
            handles.append(handle)
        results.put(("opened",))
        ready.set()
        release.wait(30)
    finally:
        for handle in handles:
            close_handle(handle)


def test_migration_reproduces_exact_conservative_total(tmp_path: Path) -> None:
    ledger, _, _, ledger_path = _migrate_fixture(tmp_path)

    assert ledger.pilot_id == PILOT_ID
    assert ledger.ledger_id == LEDGER_ID
    assert ledger.committed().as_tuple().exponent == -4
    assert ledger.committed().to_eng_string() == "3.5409"
    assert ledger.remaining().to_eng_string() == "8.4591"
    assert ledger.verify_integrity() is True

    reopened = PilotLedger.open_existing(
        ledger_path,
        PILOT_ID,
        expected_ledger_id=LEDGER_ID,
    )
    assert reopened.remaining().to_eng_string() == "8.4591"


def test_open_existing_never_creates_or_mutates_a_database(tmp_path: Path) -> None:
    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(ValueError, match="ledger database is required"):
        PilotLedger.open_existing(missing, PILOT_ID, expected_ledger_id=LEDGER_ID)
    assert not missing.exists()

    fresh = tmp_path / "fresh.sqlite3"
    sqlite3.connect(fresh).close()
    before = fresh.read_bytes()
    with pytest.raises(ValueError, match="migration manifest is required"):
        PilotLedger.open_existing(fresh, PILOT_ID, expected_ledger_id=LEDGER_ID)
    assert fresh.read_bytes() == before


def test_open_existing_requires_expected_ledger_identity(tmp_path: Path) -> None:
    _, _, _, ledger_path = _migrate_fixture(tmp_path)

    with pytest.raises(TypeError, match="expected_ledger_id"):
        PilotLedger.open_existing(ledger_path, PILOT_ID)  # type: ignore[call-arg]


def test_migration_rejects_source_ledger_hash_mismatch_atomically(tmp_path: Path) -> None:
    source_path, manifest_path, ledger_path = _write_fixture(
        tmp_path,
        mutate_source=lambda source: source["entries"][0].update(
            {"label": "PRIVATE changed after review"}
        ),
    )

    with pytest.raises(ValueError, match="source ledger hash mismatch"):
        migrate_legacy_ledger(source_path, manifest_path, ledger_path)
    assert not ledger_path.exists()


def test_migration_rejects_source_changed_between_hash_and_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path, manifest_path, ledger_path = _write_fixture(tmp_path)
    original_sha256_file = pilot_ledger.sha256_file

    def hash_then_change(path: Path) -> Any:
        digest = original_sha256_file(path)
        source = json.loads(path.read_text(encoding="utf-8"))
        source["entries"][0]["label"] = "PRIVATE changed after hashing"
        _write_json(path, source)
        return digest

    monkeypatch.setattr(pilot_ledger, "sha256_file", hash_then_change)

    with pytest.raises(ValueError, match="source ledger changed during migration"):
        migrate_legacy_ledger(source_path, manifest_path, ledger_path)
    assert not ledger_path.exists()


@pytest.mark.parametrize(
    "mutation",
    ["omitted", "added", "reordered", "amount", "state", "duplicate_source_id"],
)
def test_migration_rejects_nonexact_historical_entries(
    tmp_path: Path,
    mutation: str,
) -> None:
    def mutate(manifest: dict[str, Any]) -> None:
        entries = manifest["entries"]
        if mutation == "omitted":
            entries.pop()
        elif mutation == "added":
            entries.append(dict(entries[-1], source_entry_id=_source_id(99)))
        elif mutation == "reordered":
            entries[0], entries[1] = entries[1], entries[0]
        elif mutation == "amount":
            entries[0]["amount_usd"] = "1.2001"
        elif mutation == "state":
            entries[4]["state"] = "released"
        else:
            entries[1]["source_entry_id"] = entries[0]["source_entry_id"]

    source_path, manifest_path, ledger_path = _write_fixture(
        tmp_path,
        mutate_manifest=mutate,
    )

    with pytest.raises(ValueError):
        migrate_legacy_ledger(source_path, manifest_path, ledger_path)
    assert not ledger_path.exists()


@pytest.mark.parametrize(
    "amount",
    ["1.2", "1.20000", "01.2000", "1e0", "-1.0000", "0.0000", 1.2, True, 1],
)
def test_migration_rejects_noncanonical_money(tmp_path: Path, amount: Any) -> None:
    source_path, manifest_path, ledger_path = _write_fixture(
        tmp_path,
        mutate_manifest=lambda manifest: manifest["entries"][0].update(
            {"amount_usd": amount}
        ),
    )

    with pytest.raises(ValueError, match="canonical money"):
        migrate_legacy_ledger(source_path, manifest_path, ledger_path)
    assert not ledger_path.exists()


@pytest.mark.parametrize("location", ["manifest", "entry"])
def test_migration_manifest_rejects_unknown_fields(
    tmp_path: Path,
    location: str,
) -> None:
    def mutate(manifest: dict[str, Any]) -> None:
        target = manifest if location == "manifest" else manifest["entries"][0]
        target["private_note"] = "PRIVATE data must not be accepted"

    source_path, manifest_path, ledger_path = _write_fixture(
        tmp_path,
        mutate_manifest=mutate,
    )

    with pytest.raises(ValueError, match="schema mismatch"):
        migrate_legacy_ledger(source_path, manifest_path, ledger_path)
    assert not ledger_path.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cap_usd", "12.0001"),
        ("pilot_id", "pilot_not-a-uuid"),
        ("ledger_id", "ledger_not-a-uuid"),
        ("migration_id", "migration_not-a-uuid"),
    ],
)
def test_migration_rejects_wrong_cap_or_identity(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    source_path, manifest_path, ledger_path = _write_fixture(
        tmp_path,
        mutate_manifest=lambda manifest: manifest.update({field: value}),
    )

    with pytest.raises(ValueError):
        migrate_legacy_ledger(source_path, manifest_path, ledger_path)
    assert not ledger_path.exists()


def test_migration_is_one_shot_and_does_not_replace_existing_destination(
    tmp_path: Path,
) -> None:
    ledger, source_path, manifest_path, ledger_path = _migrate_fixture(tmp_path)
    before = ledger_path.read_bytes()

    with pytest.raises(ValueError, match="destination ledger already exists"):
        migrate_legacy_ledger(source_path, manifest_path, ledger_path)
    assert ledger_path.read_bytes() == before
    assert ledger.verify_integrity() is True


def test_two_processes_have_exactly_one_migration_activation_winner(
    tmp_path: Path,
) -> None:
    source_path, manifest_path, ledger_path = _write_fixture(tmp_path)
    context = multiprocessing.get_context("spawn")
    ready = context.Queue()
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_migration_worker,
            args=(
                str(source_path),
                str(manifest_path),
                str(ledger_path),
                ready,
                start,
                results,
            ),
        )
        for _ in range(2)
    ]
    try:
        for process in processes:
            process.start()
        assert [ready.get(timeout=30) for _ in processes] == ["ready", "ready"]
        start.set()
        observed = [results.get(timeout=30) for _ in processes]
        for process in processes:
            process.join(timeout=30)
            assert not process.is_alive()
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
            process.close()
        ready.close()
        ready.join_thread()
        results.close()
        results.join_thread()

    assert [item[0] for item in observed].count("migrated") == 1
    assert [item[0] for item in observed].count("rejected") == 1
    assert next(item for item in observed if item[0] == "migrated") == (
        "migrated",
        "8.4591",
    )
    rejected = next(item for item in observed if item[0] == "rejected")
    assert rejected[1] == "ValueError"
    reopened = PilotLedger.open_existing(
        ledger_path,
        PILOT_ID,
        expected_ledger_id=LEDGER_ID,
    )
    assert reopened.verify_integrity() is True
    assert not list(tmp_path.glob(".pilot.sqlite3.*.tmp"))


@pytest.mark.parametrize("suffix", ["-wal", "-shm"])
def test_migration_rejects_orphan_destination_sidecars_before_publication(
    tmp_path: Path,
    suffix: str,
) -> None:
    source_path, manifest_path, ledger_path = _write_fixture(tmp_path)
    sidecar = Path(f"{ledger_path}{suffix}")
    sidecar.write_bytes(b"synthetic orphan SQLite sidecar")
    before = sidecar.read_bytes()

    with pytest.raises(ValueError, match="destination ledger sidecar exists"):
        migrate_legacy_ledger(source_path, manifest_path, ledger_path)

    assert not ledger_path.exists()
    assert sidecar.read_bytes() == before


def test_migration_fails_closed_when_destination_sidecar_cannot_be_stat_checked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path, manifest_path, ledger_path = _write_fixture(tmp_path)
    denied_sidecar = Path(f"{ledger_path}-wal")
    original_lstat = Path.lstat

    def deny_sidecar_stat(path: Path) -> os.stat_result:
        if path == denied_sidecar:
            raise PermissionError("synthetic sidecar metadata denial")
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", deny_sidecar_stat)

    with pytest.raises(ValueError, match="destination ledger path cannot be verified"):
        migrate_legacy_ledger(source_path, manifest_path, ledger_path)

    assert not ledger_path.exists()


def test_migration_rejects_orphan_hot_journal_before_publication(
    tmp_path: Path,
) -> None:
    source_path, manifest_path, ledger_path = _write_fixture(tmp_path)
    crash_script = """
import os
import sqlite3
import sys

connection = sqlite3.connect(sys.argv[1])
connection.execute("PRAGMA journal_mode = DELETE")
connection.execute("PRAGMA synchronous = FULL")
connection.execute("CREATE TABLE old_ledger (value TEXT NOT NULL)")
connection.execute("INSERT INTO old_ledger VALUES ('before')")
connection.commit()
connection.execute("BEGIN IMMEDIATE")
connection.execute("UPDATE old_ledger SET value = 'after'")
os._exit(0)
"""
    result = subprocess.run(
        [sys.executable, "-c", crash_script, str(ledger_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    journal_path = Path(f"{ledger_path}-journal")
    assert journal_path.exists()
    journal_bytes = journal_path.read_bytes()
    assert journal_bytes
    ledger_path.unlink()

    with pytest.raises(ValueError, match="destination ledger sidecar exists"):
        migrate_legacy_ledger(source_path, manifest_path, ledger_path)

    assert not ledger_path.exists()
    assert journal_path.read_bytes() == journal_bytes


def test_migration_does_not_publish_a_candidate_that_fails_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path, manifest_path, ledger_path = _write_fixture(tmp_path)

    def write_invalid_database(
        path: Path,
        manifest: dict[str, Any],
        manifest_digest: str,
    ) -> None:
        del manifest, manifest_digest
        path.write_bytes(b"not a SQLite database")

    monkeypatch.setattr(
        pilot_ledger,
        "_initialize_database",
        write_invalid_database,
    )

    with pytest.raises(ValueError, match="ledger publication verification failed"):
        migrate_legacy_ledger(source_path, manifest_path, ledger_path)
    assert not ledger_path.exists()


def test_migration_requires_candidate_digest_stability_during_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path, manifest_path, ledger_path = _write_fixture(tmp_path)
    original_verify_connection = pilot_ledger._verify_connection
    mutated = False

    def verify_then_mutate_header(
        connection: sqlite3.Connection,
        *,
        expected_pilot_id: str,
        expected_ledger_id: str,
        staged: bool = False,
    ) -> Any:
        nonlocal mutated
        snapshot = original_verify_connection(
            connection,
            expected_pilot_id=expected_pilot_id,
            expected_ledger_id=expected_ledger_id,
            staged=staged,
        )
        if staged and not mutated:
            database_path = Path(
                connection.execute("PRAGMA database_list").fetchone()[2]
            )
            with database_path.open("r+b") as database_file:
                database_file.seek(68)
                database_file.write((1280591951).to_bytes(4, "big"))
                database_file.flush()
                os.fsync(database_file.fileno())
            mutated = True
        return snapshot

    monkeypatch.setattr(
        pilot_ledger,
        "_verify_connection",
        verify_then_mutate_header,
    )

    with pytest.raises(ValueError, match="ledger publication verification failed"):
        migrate_legacy_ledger(source_path, manifest_path, ledger_path)

    assert mutated is True
    assert not ledger_path.exists()


def test_migration_rejects_candidate_mutated_by_the_link_publication_hook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path, manifest_path, ledger_path = _write_fixture(tmp_path)
    original_link = os.link
    original_unlink = Path.unlink

    def mutate_then_link(source: Any, destination: Any, *args: Any, **kwargs: Any) -> None:
        original_link(source, destination, *args, **kwargs)
        with closing(sqlite3.connect(destination)) as connection, connection:
            connection.execute("PRAGMA application_id = 1280591949")

    def refuse_destination_cleanup(
        path: Path,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if path == ledger_path:
            raise PermissionError("synthetic destination cleanup lock")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(pilot_ledger.os, "link", mutate_then_link)
    monkeypatch.setattr(Path, "unlink", refuse_destination_cleanup)

    with pytest.raises(ValueError, match="ledger publication"):
        migrate_legacy_ledger(source_path, manifest_path, ledger_path)

    assert ledger_path.exists()
    assert ledger_path.stat().st_nlink == 1
    with pytest.raises(ValueError):
        PilotLedger.open_existing(
            ledger_path,
            PILOT_ID,
            expected_ledger_id=LEDGER_ID,
        )


def test_migration_rejects_sidecar_created_by_the_link_publication_hook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path, manifest_path, ledger_path = _write_fixture(tmp_path)
    sidecar = Path(f"{ledger_path}-wal")
    sidecar_bytes = b"synthetic sidecar race must be preserved"
    original_link = os.link
    original_unlink = Path.unlink

    def link_then_create_sidecar(
        source: Any,
        destination: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        original_link(source, destination, *args, **kwargs)
        sidecar.write_bytes(sidecar_bytes)

    def refuse_destination_cleanup(
        path: Path,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if path == ledger_path:
            raise PermissionError("synthetic destination cleanup lock")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(pilot_ledger.os, "link", link_then_create_sidecar)
    monkeypatch.setattr(Path, "unlink", refuse_destination_cleanup)

    with pytest.raises(ValueError, match="ledger publication failed safely"):
        migrate_legacy_ledger(source_path, manifest_path, ledger_path)

    assert ledger_path.exists()
    assert ledger_path.stat().st_nlink == 1
    assert sidecar.read_bytes() == sidecar_bytes
    sidecar.unlink()
    with pytest.raises(ValueError):
        PilotLedger.open_existing(
            ledger_path,
            PILOT_ID,
            expected_ledger_id=LEDGER_ID,
        )


def test_activation_rehashes_staged_bytes_after_acquiring_write_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path, manifest_path, ledger_path = _write_fixture(tmp_path)
    original_begin_immediate = pilot_ledger._begin_immediate
    original_unlink = Path.unlink
    mutated = False

    def mutate_then_begin(connection: sqlite3.Connection) -> None:
        nonlocal mutated
        if not mutated:
            with closing(sqlite3.connect(ledger_path)) as mutator, mutator:
                mutator.execute("PRAGMA application_id = 1280591950")
            mutated = True
        original_begin_immediate(connection)

    def refuse_destination_cleanup(
        path: Path,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if path == ledger_path:
            raise PermissionError("synthetic destination cleanup lock")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(pilot_ledger, "_begin_immediate", mutate_then_begin)
    monkeypatch.setattr(Path, "unlink", refuse_destination_cleanup)

    with pytest.raises(ValueError, match="ledger publication failed safely"):
        migrate_legacy_ledger(source_path, manifest_path, ledger_path)

    assert mutated is True
    assert ledger_path.exists()
    assert ledger_path.stat().st_nlink == 1
    with pytest.raises(ValueError):
        PilotLedger.open_existing(
            ledger_path,
            PILOT_ID,
            expected_ledger_id=LEDGER_ID,
        )


def test_activation_rechecks_sidecars_after_acquiring_write_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path, manifest_path, ledger_path = _write_fixture(tmp_path)
    sidecar = Path(f"{ledger_path}-wal")
    sidecar_bytes = b"synthetic activation sidecar race"
    original_begin_immediate = pilot_ledger._begin_immediate
    original_unlink = Path.unlink

    def begin_then_create_sidecar(connection: sqlite3.Connection) -> None:
        original_begin_immediate(connection)
        sidecar.write_bytes(sidecar_bytes)

    def refuse_destination_cleanup(
        path: Path,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if path == ledger_path:
            raise PermissionError("synthetic destination cleanup lock")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(pilot_ledger, "_begin_immediate", begin_then_create_sidecar)
    monkeypatch.setattr(Path, "unlink", refuse_destination_cleanup)

    with pytest.raises(ValueError, match="ledger publication failed safely"):
        migrate_legacy_ledger(source_path, manifest_path, ledger_path)

    assert ledger_path.exists()
    assert sidecar.read_bytes() == sidecar_bytes
    sidecar.unlink()
    with pytest.raises(ValueError):
        PilotLedger.open_existing(
            ledger_path,
            PILOT_ID,
            expected_ledger_id=LEDGER_ID,
        )


def test_activation_commit_acknowledgement_loss_returns_verified_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path, manifest_path, ledger_path = _write_fixture(tmp_path)
    original_connect = sqlite3.connect

    class CommitThenRaiseConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self._connection = connection

        def __getattr__(self, name: str) -> Any:
            return getattr(self._connection, name)

        def commit(self) -> None:
            self._connection.commit()
            raise sqlite3.OperationalError("synthetic lost commit acknowledgement")

    def connect(*args: Any, **kwargs: Any) -> Any:
        connection = original_connect(*args, **kwargs)
        target = args[0] if args else kwargs.get("database")
        if isinstance(target, str) and target.endswith("?mode=rw"):
            return CommitThenRaiseConnection(connection)
        return connection

    monkeypatch.setattr(pilot_ledger.sqlite3, "connect", connect)

    ledger = migrate_legacy_ledger(source_path, manifest_path, ledger_path)

    assert ledger.verify_integrity() is True
    assert ledger.remaining() == Decimal("8.4591")


def test_activation_commit_failure_before_effect_rolls_back_to_staged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path, manifest_path, ledger_path = _write_fixture(tmp_path)
    original_connect = sqlite3.connect
    original_unlink = Path.unlink

    class CommitBeforeEffectFailureConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self._connection = connection

        def __getattr__(self, name: str) -> Any:
            return getattr(self._connection, name)

        def commit(self) -> None:
            raise sqlite3.OperationalError("synthetic commit failure before effect")

    def connect(*args: Any, **kwargs: Any) -> Any:
        connection = original_connect(*args, **kwargs)
        target = args[0] if args else kwargs.get("database")
        if isinstance(target, str) and target.endswith("?mode=rw"):
            return CommitBeforeEffectFailureConnection(connection)
        return connection

    def refuse_destination_cleanup(
        path: Path,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if path == ledger_path:
            raise PermissionError("synthetic destination cleanup lock")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(pilot_ledger.sqlite3, "connect", connect)
    monkeypatch.setattr(Path, "unlink", refuse_destination_cleanup)

    with pytest.raises(ValueError, match="ledger publication failed safely"):
        migrate_legacy_ledger(source_path, manifest_path, ledger_path)

    with closing(sqlite3.connect(ledger_path)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (0,)
        assert connection.execute(
            "SELECT schema_version FROM pilot WHERE singleton = 1"
        ).fetchone() == (pilot_ledger.STAGED_LEDGER_SCHEMA_VERSION,)
        assert connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'trigger'"
        ).fetchone() == (0,)
    with pytest.raises(ValueError):
        PilotLedger.open_existing(
            ledger_path,
            PILOT_ID,
            expected_ledger_id=LEDGER_ID,
        )


def test_activation_reopens_to_classify_commit_after_connection_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path, manifest_path, ledger_path = _write_fixture(tmp_path)
    original_connect = sqlite3.connect

    class CommitThenDisconnectConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self._connection = connection

        @property
        def in_transaction(self) -> bool:
            return False

        def __getattr__(self, name: str) -> Any:
            return getattr(self._connection, name)

        def commit(self) -> None:
            self._connection.commit()
            self._connection.close()
            raise sqlite3.OperationalError("synthetic connection loss after commit")

    def connect(*args: Any, **kwargs: Any) -> Any:
        connection = original_connect(*args, **kwargs)
        target = args[0] if args else kwargs.get("database")
        if isinstance(target, str) and target.endswith("?mode=rw"):
            return CommitThenDisconnectConnection(connection)
        return connection

    monkeypatch.setattr(pilot_ledger.sqlite3, "connect", connect)

    ledger = migrate_legacy_ledger(source_path, manifest_path, ledger_path)

    assert ledger.verify_integrity() is True
    assert ledger.remaining() == Decimal("8.4591")


def test_verified_final_readback_close_failure_preserves_commit_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path, manifest_path, ledger_path = _write_fixture(tmp_path)
    original_connect = sqlite3.connect
    original_unlink = Path.unlink
    activation_commit_applied = False
    activation_connection_unusable = False
    final_readback_wrapped = False
    final_readback_verified = False
    final_readback_close_raised = False
    destination_cleanup_denied = False

    class CommitThenDisconnectConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self._connection = connection

        @property
        def in_transaction(self) -> bool:
            return False

        def __getattr__(self, name: str) -> Any:
            return getattr(self._connection, name)

        def commit(self) -> None:
            nonlocal activation_commit_applied, activation_connection_unusable
            self._connection.commit()
            activation_commit_applied = True
            self._connection.close()
            try:
                self._connection.execute("SELECT 1")
            except sqlite3.ProgrammingError:
                activation_connection_unusable = True
            else:
                raise AssertionError("activation connection remained usable after close")
            raise sqlite3.OperationalError("synthetic connection loss after commit")

    class CloseThenRaiseFinalReadbackConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self._connection = connection

        def __getattr__(self, name: str) -> Any:
            return getattr(self._connection, name)

        def close(self) -> None:
            nonlocal final_readback_verified, final_readback_close_raised
            assert self._connection.execute("PRAGMA user_version").fetchone() == (
                pilot_ledger.SQLITE_USER_VERSION,
            )
            assert self._connection.execute(
                "SELECT schema_version FROM pilot WHERE singleton = 1"
            ).fetchone() == (pilot_ledger.LEDGER_SCHEMA_VERSION,)
            final_readback_verified = True
            self._connection.close()
            final_readback_close_raised = True
            raise OSError("synthetic FINAL readback close failure")

    def connect(*args: Any, **kwargs: Any) -> Any:
        nonlocal final_readback_wrapped
        connection = original_connect(*args, **kwargs)
        target = args[0] if args else kwargs.get("database")
        if isinstance(target, str) and target.endswith("?mode=rw"):
            return CommitThenDisconnectConnection(connection)
        if (
            activation_commit_applied
            and not final_readback_wrapped
            and isinstance(target, str)
            and target.endswith("?mode=ro")
        ):
            final_readback_wrapped = True
            return CloseThenRaiseFinalReadbackConnection(connection)
        return connection

    def refuse_destination_cleanup(
        path: Path,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        nonlocal destination_cleanup_denied
        if path == ledger_path:
            destination_cleanup_denied = True
            raise PermissionError("synthetic destination cleanup lock")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(pilot_ledger.sqlite3, "connect", connect)
    monkeypatch.setattr(Path, "unlink", refuse_destination_cleanup)

    ledger = migrate_legacy_ledger(source_path, manifest_path, ledger_path)

    assert activation_commit_applied is True
    assert activation_connection_unusable is True
    assert final_readback_wrapped is True
    assert final_readback_verified is True
    assert final_readback_close_raised is True
    assert ledger.verify_integrity() is True
    with pytest.raises(PermissionError, match="synthetic destination cleanup lock"):
        ledger_path.unlink()
    assert destination_cleanup_denied is True
    assert ledger_path.exists()
    assert ledger.verify_integrity() is True
    assert ledger.remaining() == Decimal("8.4591")


def test_activation_close_failure_after_commit_cannot_reverse_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path, manifest_path, ledger_path = _write_fixture(tmp_path)
    original_connect = sqlite3.connect

    class CloseThenRaiseConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self._connection = connection

        def __getattr__(self, name: str) -> Any:
            return getattr(self._connection, name)

        def close(self) -> None:
            self._connection.close()
            raise OSError("synthetic close acknowledgement loss")

    def connect(*args: Any, **kwargs: Any) -> Any:
        connection = original_connect(*args, **kwargs)
        target = args[0] if args else kwargs.get("database")
        if isinstance(target, str) and target.endswith("?mode=rw"):
            return CloseThenRaiseConnection(connection)
        return connection

    monkeypatch.setattr(pilot_ledger.sqlite3, "connect", connect)

    ledger = migrate_legacy_ledger(source_path, manifest_path, ledger_path)

    assert ledger.verify_integrity() is True
    assert ledger.remaining() == Decimal("8.4591")


def test_migration_never_reports_failure_with_a_usable_published_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path, manifest_path, ledger_path = _write_fixture(tmp_path)
    original_unlink = Path.unlink

    def fail_activation(connection: sqlite3.Connection) -> None:
        connection.execute(pilot_ledger.TRIGGER_STATEMENTS[0])
        raise ValueError("synthetic activation failure")

    def refuse_destination_cleanup(
        path: Path,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if path == ledger_path:
            raise PermissionError("synthetic destination lock")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(
        pilot_ledger,
        "_create_immutability_triggers",
        fail_activation,
    )
    monkeypatch.setattr(Path, "unlink", refuse_destination_cleanup)

    with pytest.raises(ValueError, match="ledger publication failed safely"):
        migrate_legacy_ledger(source_path, manifest_path, ledger_path)
    assert ledger_path.exists()
    assert ledger_path.stat().st_nlink == 1
    with closing(sqlite3.connect(ledger_path)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (0,)
        assert connection.execute(
            "SELECT schema_version FROM pilot WHERE singleton = 1"
        ).fetchone() == (pilot_ledger.STAGED_LEDGER_SCHEMA_VERSION,)
        assert connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'trigger'"
        ).fetchone() == (0,)
    with pytest.raises(ValueError):
        PilotLedger.open_existing(
            ledger_path,
            PILOT_ID,
            expected_ledger_id=LEDGER_ID,
        )


def test_failed_publication_uses_hardlink_quarantine_when_cleanup_is_locked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path, manifest_path, ledger_path = _write_fixture(tmp_path)
    original_unlink = Path.unlink

    def refuse_link_cleanup(
        path: Path,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        is_temporary_database = (
            path.parent == tmp_path
            and path.name.startswith(".pilot.sqlite3.")
            and path.suffix == ".tmp"
        )
        if path == ledger_path or is_temporary_database:
            raise PermissionError("synthetic Windows file lock")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse_link_cleanup)

    with pytest.raises(ValueError, match="ledger publication failed safely"):
        migrate_legacy_ledger(source_path, manifest_path, ledger_path)

    quarantine_links = list(tmp_path.glob(".pilot.sqlite3.*.tmp"))
    assert ledger_path.exists()
    assert len(quarantine_links) == 1
    assert os.path.samefile(ledger_path, quarantine_links[0])
    assert ledger_path.stat().st_nlink >= 2
    with pytest.raises(ValueError, match="canonical ledger path"):
        PilotLedger.open_existing(
            ledger_path,
            PILOT_ID,
            expected_ledger_id=LEDGER_ID,
        )

    original_unlink(quarantine_links[0])
    assert ledger_path.stat().st_nlink == 1
    with pytest.raises(ValueError):
        PilotLedger.open_existing(
            ledger_path,
            PILOT_ID,
            expected_ledger_id=LEDGER_ID,
        )


@pytest.mark.skipif(os.name != "nt", reason="requires Windows share-mode semantics")
def test_windows_delete_share_lock_cannot_make_failed_destination_later_usable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path, manifest_path, ledger_path = _write_fixture(tmp_path)
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    results = context.Queue()
    original_link = os.link
    lock_process: Any = None
    quarantine_links: list[Path] = []

    def link_then_lock_temporary(
        source: Any,
        destination: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        nonlocal lock_process
        original_link(source, destination, *args, **kwargs)
        lock_process = context.Process(
            target=_windows_hold_without_delete_share,
            args=(str(source), str(destination), ready, release, results),
        )
        lock_process.start()
        assert ready.wait(30), "delete-share lock process did not become ready"
        assert results.get(timeout=30) == ("opened",)

    monkeypatch.setattr(pilot_ledger.os, "link", link_then_lock_temporary)

    try:
        with pytest.raises(ValueError, match="ledger publication failed safely"):
            migrate_legacy_ledger(source_path, manifest_path, ledger_path)

        quarantine_links = list(tmp_path.glob(".pilot.sqlite3.*.tmp"))
        assert ledger_path.exists()
        assert len(quarantine_links) == 1
        assert os.path.samefile(ledger_path, quarantine_links[0])
        assert ledger_path.stat().st_nlink == 2
    finally:
        release.set()
        if lock_process is not None:
            lock_process.join(timeout=30)
            if lock_process.is_alive():
                lock_process.terminate()
                lock_process.join(timeout=10)
            lock_process.close()
        results.close()
        results.join_thread()

    quarantine_links[0].unlink()
    assert ledger_path.stat().st_nlink == 1
    with pytest.raises(ValueError):
        PilotLedger.open_existing(
            ledger_path,
            PILOT_ID,
            expected_ledger_id=LEDGER_ID,
        )


@pytest.mark.parametrize(
    ("column", "value", "expected"),
    [
        ("pilot_id", "pilot_00000000000040008000000000000009", "pilot identity mismatch"),
        ("ledger_id", "ledger_00000000000040008000000000000009", "ledger identity mismatch"),
        ("cap_usd", "11.9999", "ledger cap mismatch"),
        ("migration_manifest_sha256", "f" * 64, "migration manifest digest mismatch"),
    ],
)
def test_integrity_rejects_changed_pilot_metadata(
    tmp_path: Path,
    column: str,
    value: str,
    expected: str,
) -> None:
    ledger, _, _, ledger_path = _migrate_fixture(tmp_path)
    with closing(sqlite3.connect(ledger_path)) as connection, connection:
        _drop_immutability_triggers(connection)
        connection.execute(f"UPDATE pilot SET {column} = ?", (value,))

    with pytest.raises(ValueError, match=expected):
        ledger.verify_integrity()


def test_integrity_rejects_wrong_expected_ids(tmp_path: Path) -> None:
    _, _, _, ledger_path = _migrate_fixture(tmp_path)

    with pytest.raises(ValueError, match="pilot identity mismatch"):
        PilotLedger.open_existing(
            ledger_path,
            "pilot_00000000000040008000000000000009",
            expected_ledger_id=LEDGER_ID,
        )
    with pytest.raises(ValueError, match="ledger identity mismatch"):
        PilotLedger.open_existing(
            ledger_path,
            PILOT_ID,
            expected_ledger_id="ledger_00000000000040008000000000000009",
        )


def test_integrity_rejects_broken_event_chain(tmp_path: Path) -> None:
    ledger, _, _, ledger_path = _migrate_fixture(tmp_path)
    with closing(sqlite3.connect(ledger_path)) as connection, connection:
        _drop_immutability_triggers(connection)
        connection.execute(
            "UPDATE events SET previous_hash = ? WHERE event_index = 2",
            ("f" * 64,),
        )

    with pytest.raises(ValueError, match="event chain mismatch"):
        ledger.verify_integrity()


def test_integrity_rejects_migration_reservation_metadata_mismatch(
    tmp_path: Path,
) -> None:
    ledger, _, _, ledger_path = _migrate_fixture(tmp_path)
    with closing(sqlite3.connect(ledger_path)) as connection, connection:
        connection.execute("DROP TRIGGER reservations_no_update")
        connection.execute(
            "UPDATE reservations SET bundle_id = ? WHERE reservation_id = ?",
            (_bundle_id(200), _typed_id("reservation", 11)),
        )
        connection.execute(
            """
            CREATE TRIGGER reservations_no_update
            BEFORE UPDATE ON reservations
            BEGIN SELECT RAISE(ABORT, 'reservations are immutable'); END
            """
        )

    with pytest.raises(ValueError, match="migration reservation mismatch"):
        ledger.verify_integrity()


def test_integrity_rejects_noncanonical_journal_mode(tmp_path: Path) -> None:
    ledger, _, _, ledger_path = _migrate_fixture(tmp_path)
    with closing(sqlite3.connect(ledger_path)) as connection, connection:
        mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
    if mode.lower() != "wal":
        pytest.skip("SQLite WAL mode is unavailable on this filesystem")

    with pytest.raises(ValueError, match="journal mode mismatch"):
        ledger.verify_integrity()


def test_integrity_rejects_schema_or_migration_entry_changes(tmp_path: Path) -> None:
    ledger, _, _, ledger_path = _migrate_fixture(tmp_path)
    with closing(sqlite3.connect(ledger_path)) as connection, connection:
        _drop_immutability_triggers(connection)
        connection.execute("DELETE FROM migration_entries WHERE entry_index = 5")
        connection.execute("PRAGMA user_version = 2")

    with pytest.raises(ValueError, match="schema version mismatch"):
        ledger.verify_integrity()


def test_integrity_rejects_unknown_view_with_private_literal(tmp_path: Path) -> None:
    ledger, _, _, ledger_path = _migrate_fixture(tmp_path)
    private_literal = "PRIVATE_SCHEMA_LITERAL_MUST_BE_REJECTED"
    with closing(sqlite3.connect(ledger_path)) as connection, connection:
        connection.execute(
            f"CREATE VIEW leaked_private_value AS SELECT '{private_literal}' AS value"
        )

    assert private_literal.encode("ascii") in ledger_path.read_bytes()
    with pytest.raises(ValueError, match="ledger schema mismatch"):
        ledger.verify_integrity()


def test_integrity_rejects_failed_sqlite_integrity_check(tmp_path: Path) -> None:
    ledger, _, _, ledger_path = _migrate_fixture(tmp_path)
    with closing(sqlite3.connect(ledger_path)) as connection, connection:
        connection.execute("PRAGMA writable_schema = ON")
        connection.execute(
            "UPDATE sqlite_master SET rootpage = 2147483647 WHERE name = 'events'"
        )
        connection.execute("PRAGMA writable_schema = OFF")

    with pytest.raises(ValueError, match="SQLite integrity check failed"):
        ledger.verify_integrity()


def test_database_excludes_legacy_labels_paths_and_provider_identifiers(
    tmp_path: Path,
) -> None:
    _, source_path, _, ledger_path = _migrate_fixture(tmp_path)
    database_bytes = ledger_path.read_bytes()

    assert b"PRIVATE synthetic legacy label" not in database_bytes
    assert os.fsencode(str(source_path)) not in database_bytes
    assert b"provider_request" not in database_bytes
    with closing(sqlite3.connect(ledger_path)) as connection, connection:
        assert {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        } == {"pilot", "migration_entries", "reservations", "events"}


def test_open_rejects_hardlink_alias_when_supported(tmp_path: Path) -> None:
    _, _, _, ledger_path = _migrate_fixture(tmp_path)
    alias = tmp_path / "alias.sqlite3"
    try:
        os.link(ledger_path, alias)
    except OSError:
        pytest.skip("hard links are not supported by this filesystem")

    with pytest.raises(ValueError, match="canonical ledger path"):
        PilotLedger.open_existing(alias, PILOT_ID, expected_ledger_id=LEDGER_ID)


def test_offline_migration_cli_is_neutral_and_provider_free(tmp_path: Path) -> None:
    source_path, manifest_path, ledger_path = _write_fixture(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(MIGRATION_SCRIPT),
            "--source-ledger",
            str(source_path),
            "--manifest",
            str(manifest_path),
            "--ledger",
            str(ledger_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    assert output.strip() == "migration complete"
    for private_value in (
        str(source_path),
        str(manifest_path),
        str(ledger_path),
        PILOT_ID,
        LEDGER_ID,
        "PRIVATE",
    ):
        assert private_value not in output
    script_source = MIGRATION_SCRIPT.read_text(encoding="utf-8").lower()
    assert "fal" not in script_source
    assert "credential" not in script_source


def test_offline_migration_cli_redacts_errors(tmp_path: Path) -> None:
    private_name = "PRIVATE-source-ledger.json"
    source_path, manifest_path, ledger_path = _write_fixture(
        tmp_path,
        mutate_source=lambda source: source["entries"][0].update(
            {"label": "PRIVATE source content"}
        ),
    )
    renamed_source = source_path.with_name(private_name)
    source_path.rename(renamed_source)

    result = subprocess.run(
        [
            sys.executable,
            str(MIGRATION_SCRIPT),
            "--source-ledger",
            str(renamed_source),
            "--manifest",
            str(manifest_path),
            "--ledger",
            str(ledger_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert output.strip() == "error: ledger migration failed"
    assert "PRIVATE" not in output
    assert str(tmp_path) not in output


def test_offline_migration_cli_redacts_argument_parser_errors(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(MIGRATION_SCRIPT),
            "--source-ledger",
            str(tmp_path / "source.json"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--ledger",
            str(tmp_path / "ledger.sqlite3"),
            "PRIVATE_SECRET_VALUE",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert output.strip() == "error: ledger migration failed"
    assert "PRIVATE_SECRET_VALUE" not in output
    assert str(tmp_path) not in output


def test_reservation_uses_task1_bundle_sha_exact_money_and_derived_balance(
    tmp_path: Path,
) -> None:
    ledger, _, _, _ = _migrate_fixture(tmp_path)
    reservation = ledger.reserve(
        _bundle_id(100),
        _typed_id("exec", 100),
        Decimal("6.0000"),
    )

    assert reservation.amount == Decimal("6.0000")
    assert reservation.bundle_id == _bundle_id(100)
    assert reservation.execution_id == _typed_id("exec", 100)
    assert reservation.id.startswith("reservation_")
    assert ledger.state(reservation.id) == "reserved"
    assert ledger.committed() == Decimal("9.5409")
    assert ledger.remaining() == Decimal("2.4591")


def test_reserve_recovers_exact_effect_after_commit_connection_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, _, _, ledger_path = _migrate_fixture(tmp_path)
    observed = _install_commit_then_disconnect(
        monkeypatch,
        readback_close_raises=True,
    )
    bundle_id = _bundle_id(100)
    execution_id = _typed_id("exec", 100)

    reservation = ledger.reserve(bundle_id, execution_id, Decimal("0.1000"))

    assert observed == {
        "write_connections": 1,
        "commit_applied": True,
        "write_connection_unusable": True,
        "fresh_readback_opened": True,
        "exact_event_readback": True,
        "exact_reservation_readback": True,
        "readback_close_called": True,
        "readback_close_raised": True,
    }
    assert reservation.bundle_id == bundle_id
    assert reservation.execution_id == execution_id
    assert reservation.amount == Decimal("0.1000")
    assert ledger.state(reservation.id) == "reserved"
    with closing(sqlite3.connect(ledger_path)) as connection:
        stored_reservation = connection.execute(
            """
            SELECT reservation_id, bundle_id, execution_id, amount_usd, migration_id
            FROM reservations WHERE reservation_id = ?
            """,
            (reservation.id,),
        ).fetchone()
    assert stored_reservation == (
        reservation.id,
        bundle_id,
        execution_id,
        "0.1000",
        None,
    )
    event = _last_event_row(ledger_path, reservation.id)
    assert event[0].startswith("event_")
    assert event[1:] == (
        None,
        "reserved",
        "0.1000",
        "reservation_created",
        None,
    )
    assert ledger.remaining() == Decimal("8.3591")


def test_transition_recovers_exact_effect_after_commit_connection_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, _, _, ledger_path = _migrate_fixture(tmp_path)
    reservation = ledger.reserve(
        _bundle_id(100),
        _typed_id("exec", 100),
        Decimal("0.1000"),
    )
    ledger.transition(reservation.id, "uploading")
    observed = _install_commit_then_disconnect(monkeypatch)

    ledger.transition(reservation.id, "submit_started")

    assert observed["write_connections"] == 1
    assert observed["commit_applied"] is True
    assert observed["write_connection_unusable"] is True
    assert observed["fresh_readback_opened"] is True
    assert observed["exact_event_readback"] is True
    assert observed["exact_reservation_readback"] is False
    assert observed["readback_close_called"] is True
    assert ledger.state(reservation.id) == "submit_started"
    event = _last_event_row(ledger_path, reservation.id)
    assert event[0].startswith("event_")
    assert event[1:] == (
        "uploading",
        "submit_started",
        "0.1000",
        "state_transition",
        None,
    )


def test_release_recovers_exact_effect_after_commit_connection_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, _, _, ledger_path = _migrate_fixture(tmp_path)
    reservation = ledger.reserve(
        _bundle_id(100),
        _typed_id("exec", 100),
        Decimal("0.1000"),
    )
    ledger.transition(reservation.id, "uploading")
    observed = _install_commit_then_disconnect(monkeypatch)

    ledger.release_pre_submit(reservation.id, "synthetic pre-submit failure")

    assert observed["write_connections"] == 1
    assert observed["commit_applied"] is True
    assert observed["write_connection_unusable"] is True
    assert observed["fresh_readback_opened"] is True
    assert observed["exact_event_readback"] is True
    assert observed["exact_reservation_readback"] is False
    assert observed["readback_close_called"] is True
    assert ledger.state(reservation.id) == "released"
    event = _last_event_row(ledger_path, reservation.id)
    assert event[0].startswith("event_")
    assert event[1:] == (
        "uploading",
        "released",
        "0.1000",
        "pre_submit_release",
        None,
    )
    assert ledger.remaining() == Decimal("8.4591")


def test_reconcile_recovers_exact_effect_after_commit_connection_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, _, _, ledger_path = _migrate_fixture(tmp_path)
    reservation = ledger.reserve(
        _bundle_id(100),
        _typed_id("exec", 100),
        Decimal("0.1000"),
    )
    for state in ("uploading", "submit_started", "submitted"):
        ledger.transition(reservation.id, state)
    evidence = hashlib.sha256(b"synthetic reconciliation evidence").hexdigest()
    observed = _install_commit_then_disconnect(monkeypatch)

    ledger.reconcile(reservation.id, "consumed", evidence_sha256=evidence)

    assert observed["write_connections"] == 1
    assert observed["commit_applied"] is True
    assert observed["write_connection_unusable"] is True
    assert observed["fresh_readback_opened"] is True
    assert observed["exact_event_readback"] is True
    assert observed["exact_reservation_readback"] is False
    assert observed["readback_close_called"] is True
    assert ledger.state(reservation.id) == "consumed"
    event = _last_event_row(ledger_path, reservation.id)
    assert event[0].startswith("event_")
    assert event[1:] == (
        "submitted",
        "consumed",
        "0.1000",
        "provider_reconciliation",
        evidence,
    )
    assert ledger.remaining() == Decimal("8.3591")


def test_write_failure_rolls_back_and_preserves_primary_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, _, _, _ = _migrate_fixture(tmp_path)
    original_connect = sqlite3.connect
    observed = {
        "write_connections": 0,
        "commit_failed_while_active": False,
        "rollback_attempted": False,
        "close_attempted": False,
        "fresh_readback_opened": False,
    }

    class FailCommitAndCleanupConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self._connection = connection

        def __getattr__(self, name: str) -> Any:
            return getattr(self._connection, name)

        def commit(self) -> None:
            observed["commit_failed_while_active"] = self._connection.in_transaction
            raise sqlite3.OperationalError("synthetic write commit failure before effect")

        def rollback(self) -> None:
            observed["rollback_attempted"] = True
            self._connection.rollback()
            raise OSError("synthetic rollback cleanup failure")

        def close(self) -> None:
            observed["close_attempted"] = True
            self._connection.close()
            raise OSError("synthetic write close cleanup failure")

    def connect(*args: Any, **kwargs: Any) -> Any:
        connection = original_connect(*args, **kwargs)
        target = args[0] if args else kwargs.get("database")
        if isinstance(target, str) and target.endswith("?mode=rw"):
            observed["write_connections"] += 1
            if observed["write_connections"] == 1:
                return FailCommitAndCleanupConnection(connection)
        if isinstance(target, str) and target.endswith("?mode=ro"):
            observed["fresh_readback_opened"] = True
        return connection

    monkeypatch.setattr(pilot_ledger.sqlite3, "connect", connect)
    bundle_id = _bundle_id(100)
    execution_id = _typed_id("exec", 100)

    with pytest.raises(
        sqlite3.OperationalError,
        match="synthetic write commit failure before effect",
    ):
        ledger.reserve(bundle_id, execution_id, Decimal("0.1000"))

    assert observed["write_connections"] == 1
    assert observed["commit_failed_while_active"] is True
    assert observed["rollback_attempted"] is True
    assert observed["close_attempted"] is True
    assert observed["fresh_readback_opened"] is True
    assert ledger.remaining() == Decimal("8.4591")
    reservation = ledger.reserve(bundle_id, execution_id, Decimal("0.1000"))
    assert ledger.state(reservation.id) == "reserved"


def test_write_close_failure_after_acknowledged_commit_preserves_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, _, _, _ = _migrate_fixture(tmp_path)
    original_connect = sqlite3.connect
    observed = {"write_connections": 0, "close_raised": False}

    class CloseThenRaiseConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self._connection = connection

        def __getattr__(self, name: str) -> Any:
            return getattr(self._connection, name)

        def close(self) -> None:
            self._connection.close()
            observed["close_raised"] = True
            raise OSError("synthetic acknowledged-write close failure")

    def connect(*args: Any, **kwargs: Any) -> Any:
        connection = original_connect(*args, **kwargs)
        target = args[0] if args else kwargs.get("database")
        if isinstance(target, str) and target.endswith("?mode=rw"):
            observed["write_connections"] += 1
            if observed["write_connections"] == 1:
                return CloseThenRaiseConnection(connection)
        return connection

    monkeypatch.setattr(pilot_ledger.sqlite3, "connect", connect)

    reservation = ledger.reserve(
        _bundle_id(100),
        _typed_id("exec", 100),
        Decimal("0.1000"),
    )

    assert observed == {"write_connections": 1, "close_raised": True}
    assert ledger.state(reservation.id) == "reserved"
    assert ledger.remaining() == Decimal("8.3591")


@pytest.mark.parametrize(
    "operation",
    ["reserve", "submit_started", "release", "reconcile"],
)
@pytest.mark.parametrize(
    ("close_after_commit", "readback_close_raises"),
    [(False, False), (True, False), (True, True)],
)
def test_every_commit_exception_uses_fresh_exact_readback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    close_after_commit: bool,
    readback_close_raises: bool,
) -> None:
    case = _prepare_write_case(tmp_path, operation)
    observed = _install_write_commit_fault(
        monkeypatch,
        case["ledger_path"],
        durable_effect=True,
        close_after_commit=close_after_commit,
        reported_in_transaction=True,
        readback_close_raises=readback_close_raises,
    )

    result = case["call"]()

    reservation = _assert_write_case_effect(case, result)
    assert observed["write_connections"] == 1
    assert observed["commit_calls"] == 1
    assert observed["readback_attempts"] == 1
    assert observed["in_transaction_reads"] == 0
    assert observed["sequence"][:4] == [
        "commit_exception",
        "rollback",
        "write_close",
        "readback_open",
    ]
    assert observed["exact_event_readback"] is True
    assert observed["exact_reservation_readback"] is (operation == "reserve")
    assert observed["readback_close_calls"] == 1
    assert observed["readback_close_raised"] is readback_close_raises
    assert observed["write_connection_unusable"] is close_after_commit
    if operation == "reserve":
        with closing(sqlite3.connect(case["ledger_path"])) as connection:
            row = connection.execute(
                """
                SELECT reservation_id, bundle_id, execution_id, amount_usd
                FROM reservations WHERE reservation_id = ?
                """,
                (reservation.id,),
            ).fetchone()
        assert row == (
            reservation.id,
            case["bundle_id"],
            case["execution_id"],
            "0.1000",
        )


@pytest.mark.parametrize(
    "operation",
    ["reserve", "submit_started", "release", "reconcile"],
)
@pytest.mark.parametrize("durable_effect", [False, True])
@pytest.mark.parametrize("exception_type", [sqlite3.IntegrityError, OSError])
def test_commit_exception_type_does_not_change_exact_classification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    durable_effect: bool,
    exception_type: type[Exception],
) -> None:
    case = _prepare_write_case(tmp_path, operation)
    commit_exception = exception_type("synthetic non-operational commit exception")
    observed = _install_write_commit_fault(
        monkeypatch,
        case["ledger_path"],
        durable_effect=durable_effect,
        close_after_commit=durable_effect,
        reported_in_transaction=not durable_effect,
        commit_exception=commit_exception,
    )

    if durable_effect:
        result = case["call"]()
        _assert_write_case_effect(case, result)
    else:
        with pytest.raises(exception_type) as exc_info:
            case["call"]()
        assert exc_info.value is commit_exception
        assert _event_count(case["ledger_path"]) == case["events_before"]

    assert observed["write_connections"] == 1
    assert observed["commit_calls"] == 1
    assert observed["readback_attempts"] == 1
    assert observed["in_transaction_reads"] == 0
    assert observed["sequence"][:4] == [
        "commit_exception",
        "rollback",
        "write_close",
        "readback_open",
    ]
    assert observed["exact_event_readback"] is True


@pytest.mark.parametrize(
    "operation",
    ["reserve", "submit_started", "release", "reconcile"],
)
def test_commit_classification_uses_the_exact_preallocated_ids_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    case = _prepare_write_case(tmp_path, operation)
    expected_reservation_id = _typed_id("reservation", 901)
    expected_event_id = _typed_id("event", 901)
    allocated_kinds: list[str] = []

    def allocate_typed_id(kind: str) -> str:
        allocated_kinds.append(kind)
        if kind == "reservation":
            return expected_reservation_id
        assert kind == "event"
        return expected_event_id

    monkeypatch.setattr(pilot_ledger, "_new_typed_id", allocate_typed_id)
    observed = _install_write_commit_fault(
        monkeypatch,
        case["ledger_path"],
        durable_effect=True,
        close_after_commit=True,
        reported_in_transaction=True,
    )

    result = case["call"]()
    reservation = _assert_write_case_effect(case, result)

    assert allocated_kinds == (
        ["reservation", "event"] if operation == "reserve" else ["event"]
    )
    assert _last_event_row(case["ledger_path"], reservation.id)[0] == expected_event_id
    if operation == "reserve":
        assert reservation.id == expected_reservation_id
    assert observed["write_connections"] == 1
    assert observed["commit_calls"] == 1
    assert observed["readback_attempts"] == 1


@pytest.mark.parametrize(
    "operation",
    ["reserve", "submit_started", "release", "reconcile"],
)
def test_acknowledged_commit_close_failure_preserves_every_write_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    case = _prepare_write_case(tmp_path, operation)
    original_connect = sqlite3.connect
    observed = {"write_connections": 0, "close_raised": False}

    class CloseThenRaiseConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self._connection = connection

        def __getattr__(self, name: str) -> Any:
            return getattr(self._connection, name)

        def close(self) -> None:
            self._connection.close()
            observed["close_raised"] = True
            raise OSError("synthetic acknowledged write close failure")

    def connect(*args: Any, **kwargs: Any) -> Any:
        connection = original_connect(*args, **kwargs)
        target = args[0] if args else kwargs.get("database")
        if isinstance(target, str) and target.endswith("?mode=rw"):
            observed["write_connections"] += 1
            if observed["write_connections"] == 1:
                return CloseThenRaiseConnection(connection)
        return connection

    monkeypatch.setattr(pilot_ledger.sqlite3, "connect", connect)

    result = case["call"]()

    _assert_write_case_effect(case, result)
    assert observed == {"write_connections": 1, "close_raised": True}


@pytest.mark.parametrize(
    "operation",
    ["reserve", "submit_started", "release", "reconcile"],
)
@pytest.mark.parametrize(
    ("reported_in_transaction", "close_after_commit"),
    [(True, False), (False, True)],
)
def test_pre_effect_commit_errors_are_freshly_classified_then_propagated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    reported_in_transaction: bool,
    close_after_commit: bool,
) -> None:
    case = _prepare_write_case(tmp_path, operation)
    observed = _install_write_commit_fault(
        monkeypatch,
        case["ledger_path"],
        durable_effect=False,
        close_after_commit=close_after_commit,
        reported_in_transaction=reported_in_transaction,
    )

    with pytest.raises(
        sqlite3.OperationalError,
        match="synthetic primary commit acknowledgement error",
    ):
        case["call"]()

    assert observed["write_connections"] == 1
    assert observed["commit_calls"] == 1
    assert observed["readback_attempts"] == 1
    assert observed["exact_event_readback"] is True
    assert _event_count(case["ledger_path"]) == case["events_before"]
    if case["reservation"] is None:
        with closing(sqlite3.connect(case["ledger_path"])) as connection:
            count = connection.execute(
                """
                SELECT COUNT(*) FROM reservations
                WHERE bundle_id = ? OR execution_id = ?
                """,
                (case["bundle_id"], case["execution_id"]),
            ).fetchone()
        assert count == (0,)
    else:
        assert case["ledger"].state(case["reservation"].id) == case["expected_from_state"]


@pytest.mark.parametrize(
    "operation",
    ["reserve", "submit_started", "release", "reconcile"],
)
@pytest.mark.parametrize(
    "readback_fault",
    ["open", "integrity", "event_mismatch"],
)
def test_unverified_commit_effect_never_normalizes_to_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    readback_fault: str,
) -> None:
    case = _prepare_write_case(tmp_path, operation)
    observed = _install_write_commit_fault(
        monkeypatch,
        case["ledger_path"],
        durable_effect=True,
        close_after_commit=False,
        reported_in_transaction=False,
        readback_fault=readback_fault,
    )

    with pytest.raises(
        sqlite3.OperationalError,
        match="synthetic primary commit acknowledgement error",
    ):
        case["call"]()

    assert observed["write_connections"] == 1
    assert observed["commit_calls"] == 1
    assert observed["readback_attempts"] == 1
    assert _event_count(case["ledger_path"]) == case["events_before"] + 1


def test_exact_reservation_mismatch_does_not_verify_ambiguous_reserve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _prepare_write_case(tmp_path, "reserve")
    observed = _install_write_commit_fault(
        monkeypatch,
        case["ledger_path"],
        durable_effect=True,
        close_after_commit=False,
        reported_in_transaction=False,
        readback_fault="reservation_mismatch",
    )

    with pytest.raises(
        sqlite3.OperationalError,
        match="synthetic primary commit acknowledgement error",
    ):
        case["call"]()

    assert observed["exact_event_readback"] is True
    assert observed["exact_reservation_readback"] is True
    assert observed["write_connections"] == 1
    assert observed["commit_calls"] == 1


def test_commit_readback_rejects_path_swap_or_extra_hardlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _prepare_write_case(tmp_path, "submit_started")
    observed = _install_write_commit_fault(
        monkeypatch,
        case["ledger_path"],
        durable_effect=True,
        close_after_commit=False,
        reported_in_transaction=False,
        readback_fault="identity_swap",
    )

    with pytest.raises(
        sqlite3.OperationalError,
        match="synthetic primary commit acknowledgement error",
    ):
        case["call"]()

    assert observed["identity_swap_attempted"] is True
    assert observed["write_connections"] == 1
    assert observed["commit_calls"] == 1
    assert observed["readback_attempts"] == 1


def test_release_idempotence_requires_pre_submit_release_provenance(
    tmp_path: Path,
) -> None:
    ledger, _, _, _ = _migrate_fixture(tmp_path)
    reservation = ledger.reserve(
        _bundle_id(100),
        _typed_id("exec", 100),
        Decimal("0.1000"),
    )
    ledger.transition(reservation.id, "uploading")
    ledger.transition(reservation.id, "submit_started")
    evidence = hashlib.sha256(b"synthetic released reconciliation").hexdigest()
    ledger.reconcile(reservation.id, "released", evidence_sha256=evidence)
    terminal_head = ledger.head_hash

    with pytest.raises(RuntimeError, match="provenance"):
        ledger.release_pre_submit(reservation.id, "PRIVATE reason must not escape")

    assert ledger.head_hash == terminal_head
    assert ledger.state(reservation.id) == "released"


def test_transition_idempotence_requires_exact_state_transition_provenance(
    tmp_path: Path,
) -> None:
    ledger, _, _, _ = _migrate_fixture(tmp_path)
    reservation = ledger.reserve(
        _bundle_id(100),
        _typed_id("exec", 100),
        Decimal("0.1000"),
    )
    ledger.transition(reservation.id, "uploading")
    ledger.transition(reservation.id, "submit_started")
    transition_head = ledger.head_hash

    ledger.transition(reservation.id, "submit_started")

    assert ledger.head_hash == transition_head
    assert ledger.state(reservation.id) == "submit_started"


@pytest.mark.parametrize("terminal_route", ["provider_reconciliation", "pre_submit_release"])
def test_transition_rejects_same_state_from_a_different_event_provenance(
    tmp_path: Path,
    terminal_route: str,
) -> None:
    ledger, _, _, _ = _migrate_fixture(tmp_path)
    reservation = ledger.reserve(
        _bundle_id(100),
        _typed_id("exec", 100),
        Decimal("0.1000"),
    )
    if terminal_route == "provider_reconciliation":
        ledger.transition(reservation.id, "uploading")
        ledger.transition(reservation.id, "submit_started")
        evidence = hashlib.sha256(b"synthetic transition provenance").hexdigest()
        ledger.reconcile(reservation.id, "consumed", evidence_sha256=evidence)
        outcome = "consumed"
    else:
        ledger.release_pre_submit(reservation.id, "synthetic pre-submit release")
        outcome = "released"
    terminal_head = ledger.head_hash

    with pytest.raises(RuntimeError, match="provenance"):
        ledger.transition(reservation.id, outcome)

    assert ledger.head_hash == terminal_head
    assert ledger.state(reservation.id) == outcome


def test_reconcile_idempotence_requires_exact_prior_evidence(
    tmp_path: Path,
) -> None:
    ledger, _, _, _ = _migrate_fixture(tmp_path)
    reservation = ledger.reserve(
        _bundle_id(100),
        _typed_id("exec", 100),
        Decimal("0.1000"),
    )
    ledger.transition(reservation.id, "uploading")
    ledger.transition(reservation.id, "submit_started")
    first_evidence = hashlib.sha256(b"synthetic first evidence").hexdigest()
    other_evidence = hashlib.sha256(b"synthetic other evidence").hexdigest()
    ledger.reconcile(reservation.id, "consumed", evidence_sha256=first_evidence)
    terminal_head = ledger.head_hash

    ledger.reconcile(reservation.id, "consumed", evidence_sha256=first_evidence)
    assert ledger.head_hash == terminal_head
    with pytest.raises(RuntimeError, match="provenance"):
        ledger.reconcile(reservation.id, "consumed", evidence_sha256=other_evidence)

    assert ledger.head_hash == terminal_head


@pytest.mark.parametrize("terminal_route", ["normal_consumed", "pre_submit_release"])
def test_reconcile_rejects_same_state_from_a_different_terminal_route(
    tmp_path: Path,
    terminal_route: str,
) -> None:
    ledger, _, _, _ = _migrate_fixture(tmp_path)
    reservation = ledger.reserve(
        _bundle_id(100),
        _typed_id("exec", 100),
        Decimal("0.1000"),
    )
    if terminal_route == "normal_consumed":
        for state in ("uploading", "submit_started", "submitted", "consumed"):
            ledger.transition(reservation.id, state)
        outcome = "consumed"
    else:
        ledger.release_pre_submit(reservation.id, "synthetic pre-submit release")
        outcome = "released"
    terminal_head = ledger.head_hash
    evidence = hashlib.sha256(b"synthetic unrelated evidence").hexdigest()

    with pytest.raises(RuntimeError, match="provenance"):
        ledger.reconcile(reservation.id, outcome, evidence_sha256=evidence)

    assert ledger.head_hash == terminal_head


def test_unverified_durable_reserve_has_explicit_read_only_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _prepare_write_case(tmp_path, "reserve")
    observed = _install_write_commit_fault(
        monkeypatch,
        case["ledger_path"],
        durable_effect=True,
        close_after_commit=True,
        reported_in_transaction=False,
        readback_fault="open",
    )

    with pytest.raises(
        sqlite3.OperationalError,
        match="synthetic primary commit acknowledgement error",
    ):
        case["call"]()
    recovered = case["ledger"].recover_reservation(
        case["bundle_id"],
        case["execution_id"],
        case["amount"],
    )

    assert observed["write_connections"] == 1
    assert observed["commit_calls"] == 1
    assert _event_count(case["ledger_path"]) == case["events_before"] + 1
    assert recovered.bundle_id == case["bundle_id"]
    assert recovered.execution_id == case["execution_id"]
    assert recovered.amount == case["amount"]
    assert case["ledger"].state(recovered.id) == "reserved"
    with pytest.raises(RuntimeError, match="already reserved"):
        case["ledger"].reserve(
            case["bundle_id"],
            case["execution_id"],
            case["amount"],
        )


def test_recover_reservation_is_read_only_and_never_creates_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, _, _, ledger_path = _migrate_fixture(tmp_path)
    reservation = ledger.reserve(
        _bundle_id(100),
        _typed_id("exec", 100),
        Decimal("0.1000"),
    )
    original_connect = sqlite3.connect
    observed = {"read_connections": 0, "write_connections": 0}
    statements: list[str] = []

    def connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        connection = original_connect(*args, **kwargs)
        target = args[0] if args else kwargs.get("database")
        if isinstance(target, str) and target.endswith("?mode=ro"):
            observed["read_connections"] += 1
            connection.set_trace_callback(statements.append)
        if isinstance(target, str) and target.endswith("?mode=rw"):
            observed["write_connections"] += 1
        return connection

    monkeypatch.setattr(pilot_ledger.sqlite3, "connect", connect)

    recovered = ledger.recover_reservation(
        reservation.bundle_id,
        reservation.execution_id,
        reservation.amount,
    )

    assert recovered == reservation
    assert observed == {"read_connections": 1, "write_connections": 0}
    normalized = [" ".join(statement.split()).casefold() for statement in statements]
    assert any(
        statement.startswith(
            "select reservation_id, bundle_id, execution_id, amount_usd, "
            "created_at_utc, migration_id from reservations"
        )
        and "where bundle_id =" in statement
        and "or execution_id =" in statement
        for statement in normalized
    )
    assert any(
        statement.startswith(
            "select from_state, to_state, amount_usd, reason_code, evidence_sha256 "
            "from events"
        )
        and "order by event_index limit 1" in statement
        for statement in normalized
    )
    assert not any(
        statement.startswith(
            ("insert ", "update ", "delete ", "create ", "drop ", "alter ", "vacuum")
        )
        or "wal_checkpoint" in statement
        or "journal_mode =" in statement
        for statement in normalized
    )
    assert all(not Path(f"{ledger_path}{suffix}").exists() for suffix in ("-wal", "-shm", "-journal"))


def test_recover_reservation_rejects_mismatch_and_migration_provenance_privately(
    tmp_path: Path,
) -> None:
    ledger, _, _, ledger_path = _migrate_fixture(tmp_path)
    reservation = ledger.reserve(
        _bundle_id(100),
        _typed_id("exec", 100),
        Decimal("0.1000"),
    )
    with closing(sqlite3.connect(ledger_path)) as connection:
        migrated = connection.execute(
            """
            SELECT bundle_id, execution_id, amount_usd
            FROM reservations WHERE migration_id IS NOT NULL
            ORDER BY rowid LIMIT 1
            """
        ).fetchone()
    assert migrated is not None

    attempts = [
        (reservation.bundle_id, reservation.execution_id, Decimal("0.2000")),
        (reservation.bundle_id, migrated[1], reservation.amount),
        (migrated[0], migrated[1], Decimal(migrated[2])),
        (_bundle_id(777), _typed_id("exec", 777), Decimal("0.1000")),
    ]
    for bundle_id, execution_id, amount in attempts:
        with pytest.raises(RuntimeError, match="recovery") as exc_info:
            ledger.recover_reservation(bundle_id, execution_id, amount)
        message = str(exc_info.value)
        assert bundle_id not in message
        assert execution_id not in message
        assert str(ledger_path) not in message


@pytest.mark.parametrize(
    "amount",
    [
        6.0,
        True,
        6,
        Decimal("6"),
        Decimal("6.00000"),
        "6",
        "6.00000",
        "6e0",
        "-1.0000",
        "0.0000",
    ],
)
def test_reservation_rejects_noncanonical_money(
    tmp_path: Path,
    amount: Any,
) -> None:
    ledger, _, _, _ = _migrate_fixture(tmp_path)

    with pytest.raises(ValueError, match="reservation amount must be canonical money"):
        ledger.reserve(_bundle_id(100), _typed_id("exec", 100), amount)
    assert ledger.remaining() == Decimal("8.4591")


def test_reservation_rejects_overspend_and_wrong_id_shapes(tmp_path: Path) -> None:
    ledger, _, _, _ = _migrate_fixture(tmp_path)

    with pytest.raises(RuntimeError, match="exceeds remaining budget"):
        ledger.reserve(
            _bundle_id(100),
            _typed_id("exec", 100),
            Decimal("8.4592"),
        )
    with pytest.raises(ValueError, match="bundle_id"):
        ledger.reserve("not-a-bundle", _typed_id("exec", 101), Decimal("1.0000"))
    with pytest.raises(ValueError, match="execution_id"):
        ledger.reserve(_bundle_id(101), "not-an-execution", Decimal("1.0000"))


def test_two_processes_cannot_overspend_remaining_budget(tmp_path: Path) -> None:
    ledger, _, _, ledger_path = _migrate_fixture(tmp_path)
    ledger.reserve(
        _bundle_id(99),
        _typed_id("exec", 99),
        Decimal("2.4591"),
    )
    assert ledger.remaining() == Decimal("6.0000")

    context = multiprocessing.get_context("spawn")
    ready = context.Queue()
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_reservation_worker,
            args=(
                str(ledger_path),
                _bundle_id(number),
                _typed_id("exec", number),
                ready,
                start,
                results,
            ),
        )
        for number in (100, 101)
    ]
    try:
        for process in processes:
            process.start()
        assert [ready.get(timeout=30) for _ in processes] == ["ready", "ready"]
        start.set()
        observed = [results.get(timeout=30) for _ in processes]
        for process in processes:
            process.join(timeout=30)
            assert not process.is_alive()
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
            process.close()
        ready.close()
        ready.join_thread()
        results.close()
        results.join_thread()

    assert [item[0] for item in observed].count("reserved") == 1
    assert [item[0] for item in observed].count("rejected") == 1
    assert observed[0][0] != observed[1][0]
    rejected = next(item for item in observed if item[0] == "rejected")
    assert rejected[1:] == (
        "RuntimeError",
        "projected cost exceeds remaining budget",
    )
    assert ledger.remaining() == Decimal("0.0000")


def test_complete_happy_path_is_durable_and_idempotent(tmp_path: Path) -> None:
    ledger, _, _, ledger_path = _migrate_fixture(tmp_path)
    bundle_id = _bundle_id(100)
    execution_id = _typed_id("exec", 100)
    snapshot = ledger.preflight_snapshot(bundle_id, execution_id)
    reservation = ledger.reserve_training(
        bundle_id,
        execution_id,
        Decimal("1.0000"),
        expected_head_sha256=snapshot.head_sha256,
    )
    initial_head = ledger.head_hash

    with pytest.raises(RuntimeError, match="provenance"):
        ledger.transition(reservation.id, "reserved")
    assert ledger.head_hash == initial_head
    ledger.transition(reservation.id, "uploading")
    assert ledger.state(reservation.id) == "uploading"
    ledger.transition(reservation.id, "submit_started")

    reopened = PilotLedger.open_existing(
        ledger_path,
        PILOT_ID,
        expected_ledger_id=LEDGER_ID,
    )
    assert reopened.state(reservation.id) == "submit_started"
    assert reopened.remaining() == Decimal("7.4591")

    reopened.transition(reservation.id, "submitted")
    reopened.transition(reservation.id, "consumed")
    consumed_head = reopened.head_hash
    reopened.transition(reservation.id, "consumed")
    assert reopened.head_hash == consumed_head
    assert reopened.state(reservation.id) == "consumed"
    assert reopened.remaining() == Decimal("7.4591")


@pytest.mark.parametrize(
    ("path", "illegal_target"),
    [
        (("reserved",), "submit_started"),
        (("reserved",), "submitted"),
        (("reserved",), "consumed"),
        (("reserved", "uploading"), "submitted"),
        (("reserved", "uploading"), "consumed"),
        (("reserved", "uploading", "submit_started"), "uploading"),
        (("reserved", "uploading", "submit_started"), "released"),
        (("reserved", "uploading", "submit_started"), "consumed"),
        (("reserved", "uploading", "submit_started", "submitted"), "released"),
        (("reserved", "uploading", "submit_started", "submitted"), "uploading"),
        (("reserved", "released"), "uploading"),
        (("reserved", "uploading", "submit_started", "submitted", "consumed"), "released"),
    ],
)
def test_transition_graph_rejects_illegal_edges(
    tmp_path: Path,
    path: tuple[str, ...],
    illegal_target: str,
) -> None:
    ledger, _, _, _ = _migrate_fixture(tmp_path)
    reservation = ledger.reserve(
        _bundle_id(100),
        _typed_id("exec", 100),
        Decimal("0.1000"),
    )
    if path == ("reserved", "released"):
        ledger.release_pre_submit(reservation.id, "synthetic pre-submit failure")
    else:
        for target in path[1:]:
            ledger.transition(reservation.id, target)

    with pytest.raises(RuntimeError, match="illegal ledger state transition"):
        ledger.transition(reservation.id, illegal_target)
    assert ledger.state(reservation.id) == path[-1]


def test_pre_submit_release_restores_budget_without_persisting_reason(
    tmp_path: Path,
) -> None:
    ledger, _, _, ledger_path = _migrate_fixture(tmp_path)
    first = ledger.reserve(
        _bundle_id(100),
        _typed_id("exec", 100),
        Decimal("1.0000"),
    )
    ledger.release_pre_submit(
        first.id,
        "PRIVATE path and upload exception details must not persist",
    )
    assert ledger.state(first.id) == "released"
    assert ledger.remaining() == Decimal("8.4591")
    released_head = ledger.head_hash
    ledger.release_pre_submit(first.id, "idempotent retry")
    assert ledger.head_hash == released_head

    second = ledger.reserve(
        _bundle_id(101),
        _typed_id("exec", 101),
        Decimal("1.0000"),
    )
    ledger.transition(second.id, "uploading")
    ledger.release_pre_submit(second.id, "upload failed before submit")
    assert ledger.state(second.id) == "released"
    assert b"PRIVATE path" not in ledger_path.read_bytes()


def test_release_after_submit_started_is_refused_and_stays_committed(
    tmp_path: Path,
) -> None:
    ledger, _, _, _ = _migrate_fixture(tmp_path)
    reservation = ledger.reserve(
        _bundle_id(100),
        _typed_id("exec", 100),
        Decimal("1.0000"),
    )
    ledger.transition(reservation.id, "uploading")
    ledger.transition(reservation.id, "submit_started")

    with pytest.raises(RuntimeError, match="cannot release after submit_started"):
        ledger.release_pre_submit(reservation.id, "ambiguous provider submit")
    assert ledger.state(reservation.id) == "submit_started"
    assert ledger.remaining() == Decimal("7.4591")


@pytest.mark.parametrize("outcome", ["released", "consumed"])
def test_ambiguous_submit_requires_evidence_backed_reconciliation(
    tmp_path: Path,
    outcome: str,
) -> None:
    ledger, _, _, ledger_path = _migrate_fixture(tmp_path)
    reservation = ledger.reserve(
        _bundle_id(100),
        _typed_id("exec", 100),
        Decimal("1.0000"),
    )
    ledger.transition(reservation.id, "uploading")
    ledger.transition(reservation.id, "submit_started")

    with pytest.raises(ValueError, match="provider evidence"):
        ledger.reconcile(reservation.id, outcome, evidence_sha256="not-a-hash")
    assert ledger.state(reservation.id) == "submit_started"

    evidence = hashlib.sha256(b"synthetic provider reconciliation evidence").hexdigest()
    ledger.reconcile(reservation.id, outcome, evidence_sha256=evidence)
    assert ledger.state(reservation.id) == outcome
    expected_remaining = Decimal("8.4591") if outcome == "released" else Decimal("7.4591")
    assert ledger.remaining() == expected_remaining

    with closing(sqlite3.connect(ledger_path)) as connection, connection:
        stored = connection.execute(
            "SELECT reason_code, evidence_sha256 FROM events ORDER BY event_index DESC LIMIT 1"
        ).fetchone()
    assert stored == ("provider_reconciliation", evidence)


def test_reconcile_is_limited_to_ambiguous_or_submitted_states(tmp_path: Path) -> None:
    ledger, _, _, _ = _migrate_fixture(tmp_path)
    reservation = ledger.reserve(
        _bundle_id(100),
        _typed_id("exec", 100),
        Decimal("1.0000"),
    )
    evidence = hashlib.sha256(b"synthetic evidence").hexdigest()

    with pytest.raises(RuntimeError, match="reconciliation is not legal"):
        ledger.reconcile(reservation.id, "released", evidence_sha256=evidence)
    with pytest.raises(ValueError, match="reconciliation outcome"):
        ledger.reconcile(reservation.id, "reserved", evidence_sha256=evidence)


@pytest.mark.parametrize("terminal_state", ["released", "consumed"])
def test_bundle_and_execution_cannot_be_replayed_after_terminal_state(
    tmp_path: Path,
    terminal_state: str,
) -> None:
    ledger, _, _, _ = _migrate_fixture(tmp_path)
    bundle_id = _bundle_id(100)
    execution_id = _typed_id("exec", 100)
    reservation = ledger.reserve(bundle_id, execution_id, Decimal("0.1000"))
    if terminal_state == "released":
        ledger.release_pre_submit(reservation.id, "synthetic release")
    else:
        for target in ("uploading", "submit_started", "submitted", "consumed"):
            ledger.transition(reservation.id, target)

    for replay_bundle, replay_execution in (
        (bundle_id, execution_id),
        (bundle_id, _typed_id("exec", 101)),
        (_bundle_id(101), execution_id),
    ):
        with pytest.raises(RuntimeError, match="execution identity was already reserved"):
            ledger.reserve(replay_bundle, replay_execution, Decimal("0.1000"))


def test_events_and_reservations_are_database_immutable(tmp_path: Path) -> None:
    ledger, _, _, ledger_path = _migrate_fixture(tmp_path)
    reservation = ledger.reserve(
        _bundle_id(100),
        _typed_id("exec", 100),
        Decimal("0.1000"),
    )
    ledger.transition(reservation.id, "uploading")

    with closing(sqlite3.connect(ledger_path)) as connection, connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "UPDATE events SET to_state = 'released' WHERE reservation_id = ?",
                (reservation.id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="reservations are immutable"):
            connection.execute(
                "DELETE FROM reservations WHERE reservation_id = ?",
                (reservation.id,),
            )
        pilot_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(pilot)")
        }
    assert not {"committed", "remaining", "total", "balance"} & pilot_columns
    assert ledger.verify_integrity() is True


def test_write_transactions_set_required_sqlite_safety_pragmas_before_begin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, _, _, _ = _migrate_fixture(tmp_path)
    statements: list[str] = []
    original_connect = sqlite3.connect

    def traced_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        connection = original_connect(*args, **kwargs)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(pilot_ledger.sqlite3, "connect", traced_connect)
    ledger.reserve(
        _bundle_id(100),
        _typed_id("exec", 100),
        Decimal("0.1000"),
    )

    normalized = [" ".join(statement.upper().split()) for statement in statements]
    begin_index = normalized.index("BEGIN IMMEDIATE")
    assert "PRAGMA BUSY_TIMEOUT = 5000" in normalized[:begin_index]
    assert "PRAGMA FOREIGN_KEYS = ON" in normalized[:begin_index]
    assert "PRAGMA SYNCHRONOUS = FULL" in normalized[:begin_index]


def test_open_verifies_all_rows_inside_one_read_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, ledger_path = _migrate_fixture(tmp_path)
    statements: list[str] = []
    original_connect = sqlite3.connect

    def traced_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        connection = original_connect(*args, **kwargs)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(pilot_ledger.sqlite3, "connect", traced_connect)
    PilotLedger.open_existing(
        ledger_path,
        PILOT_ID,
        expected_ledger_id=LEDGER_ID,
    )

    normalized = [" ".join(statement.upper().split()) for statement in statements]
    assert normalized.count("BEGIN") == 1
    begin_index = normalized.index("BEGIN")
    integrity_index = normalized.index("PRAGMA INTEGRITY_CHECK")
    event_read_index = next(
        index
        for index, statement in enumerate(normalized)
        if statement.startswith("SELECT EVENT_INDEX, EVENT_ID")
    )
    assert begin_index < integrity_index < event_read_index
    assert normalized[-1] == "ROLLBACK"


def test_snapshot_allows_legitimate_process_commit_after_read_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, _, _, ledger_path = _migrate_fixture(tmp_path)
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    start = context.Event()
    done = context.Event()
    results = context.Queue()
    process = context.Process(
        target=_post_snapshot_commit_worker,
        args=(str(ledger_path), ready, start, done, results),
    )
    original_read_connection = pilot_ledger._read_connection
    triggered = False

    @contextmanager
    def commit_after_read_transaction(path: Path) -> Any:
        nonlocal triggered
        with original_read_connection(path) as connection:
            yield connection
        if not triggered:
            triggered = True
            start.set()
            assert done.wait(30), "writer did not commit after the read transaction"

    try:
        process.start()
        assert ready.wait(30), "writer did not open the ledger"
        monkeypatch.setattr(
            pilot_ledger,
            "_read_connection",
            commit_after_read_transaction,
        )

        assert ledger.remaining() == Decimal("8.4591")
        assert results.get(timeout=30)[0] == "reserved"
        process.join(timeout=30)
        assert not process.is_alive()
        assert process.exitcode == 0
        reopened = PilotLedger.open_existing(
            ledger_path,
            PILOT_ID,
            expected_ledger_id=LEDGER_ID,
        )
        assert reopened.remaining() == Decimal("8.3591")
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        process.close()
        results.close()
        results.join_thread()


def test_preflight_snapshot_rejects_concurrent_commit_after_read_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, _, _, ledger_path = _migrate_fixture(tmp_path)
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    start = context.Event()
    done = context.Event()
    results = context.Queue()
    process = context.Process(
        target=_post_snapshot_commit_worker,
        args=(str(ledger_path), ready, start, done, results),
    )
    original_read_connection = pilot_ledger._read_connection
    triggered = False

    @contextmanager
    def commit_after_read_transaction(path: Path) -> Any:
        nonlocal triggered
        with original_read_connection(path) as connection:
            yield connection
        if not triggered:
            triggered = True
            start.set()
            assert done.wait(30), "writer did not commit after the read transaction"

    try:
        process.start()
        assert ready.wait(30), "writer did not open the ledger"
        monkeypatch.setattr(
            pilot_ledger,
            "_read_connection",
            commit_after_read_transaction,
        )

        with pytest.raises(ValueError, match="ledger changed during verification"):
            ledger.preflight_snapshot(_bundle_id(100), _typed_id("exec", 100))

        assert results.get(timeout=30)[0] == "reserved"
        process.join(timeout=30)
        assert not process.is_alive()
        assert process.exitcode == 0
        reopened = PilotLedger.open_existing(
            ledger_path,
            PILOT_ID,
            expected_ledger_id=LEDGER_ID,
        )
        assert reopened.verify_integrity() is True
        assert reopened.remaining() == Decimal("8.3591")
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        process.close()
        results.close()
        results.join_thread()
