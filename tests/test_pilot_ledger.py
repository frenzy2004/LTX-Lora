from __future__ import annotations

from contextlib import closing
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
from ltx_lora_pilot.pilot_ledger import PilotLedger, migrate_legacy_ledger


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


def test_migration_never_reports_failure_with_a_usable_published_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path, manifest_path, ledger_path = _write_fixture(tmp_path)
    original_unlink = Path.unlink

    def fail_verification(
        cls: type[PilotLedger],
        path: Path,
        expected_pilot_id: str,
        *,
        expected_ledger_id: str | None = None,
    ) -> PilotLedger:
        del cls, path, expected_pilot_id, expected_ledger_id
        raise ValueError("synthetic verification failure")

    def refuse_destination_cleanup(
        path: Path,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if path == ledger_path:
            raise PermissionError("synthetic destination lock")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(PilotLedger, "open_existing", classmethod(fail_verification))
    monkeypatch.setattr(Path, "unlink", refuse_destination_cleanup)

    with pytest.raises(ValueError, match="ledger publication verification failed"):
        migrate_legacy_ledger(source_path, manifest_path, ledger_path)
    assert not ledger_path.exists()


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
    reservation = ledger.reserve_training(
        _bundle_id(100),
        _typed_id("exec", 100),
        Decimal("1.0000"),
    )
    initial_head = ledger.head_hash

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
