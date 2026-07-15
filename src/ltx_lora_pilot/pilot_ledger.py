from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from typing import Any, Iterator, Sequence
import uuid

from .artifacts import canonical_json_bytes, sha256_file
from .budget import money


LEDGER_SCHEMA_VERSION = "pilot-budget-ledger-v1"
STAGED_LEDGER_SCHEMA_VERSION = "pilot-budget-ledger-v1-staged"
MIGRATION_SCHEMA_VERSION = "pilot-budget-migration-v1"
EVENT_SCHEMA_VERSION = "pilot-budget-event-v1"
SQLITE_USER_VERSION = 1
STAGED_SQLITE_USER_VERSION = 0
BUSY_TIMEOUT_MILLISECONDS = 5_000
CAP_USD_TEXT = "12.0000"
CAP_USD = Decimal(CAP_USD_TEXT)
LEGACY_ENTRY_COUNT = 6
LEGACY_COMMITTED_USD = Decimal("3.5409")
GENESIS_HASH = "0" * 64
SQLITE_SIDECAR_SUFFIXES = ("-journal", "-wal", "-shm")

COMMITTED_STATES = frozenset(
    {"reserved", "uploading", "submit_started", "submitted", "consumed"}
)
TERMINAL_STATES = frozenset({"released", "consumed"})
ALL_STATES = COMMITTED_STATES | {"released"}
NORMAL_TRANSITIONS = frozenset(
    {
        ("reserved", "uploading"),
        ("uploading", "submit_started"),
        ("submit_started", "submitted"),
        ("submitted", "consumed"),
    }
)
PRE_SUBMIT_RELEASE_TRANSITIONS = frozenset(
    {("reserved", "released"), ("uploading", "released")}
)
RECONCILIATION_TRANSITIONS = frozenset(
    {
        ("submit_started", "released"),
        ("submit_started", "consumed"),
        ("submitted", "released"),
        ("submitted", "consumed"),
    }
)

MONEY_PATTERN = re.compile(r"(?:0|[1-9][0-9]*)\.[0-9]{4}", re.ASCII)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}", re.ASCII)
# Narrow identity-contract exception: the approved Task 1 bundle identity is the
# content-addressed root-manifest SHA-256. All typed ledger identities below are UUIDv4.
BUNDLE_ID_PATTERN = SHA256_PATTERN
UUID4_HEX = r"[0-9a-f]{12}4[0-9a-f]{3}[89ab][0-9a-f]{15}"
PILOT_ID_PATTERN = re.compile(rf"pilot_{UUID4_HEX}", re.ASCII)
LEDGER_ID_PATTERN = re.compile(rf"ledger_{UUID4_HEX}", re.ASCII)
MIGRATION_ID_PATTERN = re.compile(rf"migration_{UUID4_HEX}", re.ASCII)
RESERVATION_ID_PATTERN = re.compile(rf"reservation_{UUID4_HEX}", re.ASCII)
EVENT_ID_PATTERN = re.compile(rf"event_{UUID4_HEX}", re.ASCII)
EXECUTION_ID_PATTERN = re.compile(rf"exec_{UUID4_HEX}", re.ASCII)
UTC_TIMESTAMP_PATTERN = re.compile(
    r"[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])"
    r"T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z",
    re.ASCII,
)

MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "pilot_id",
        "ledger_id",
        "migration_id",
        "cap_usd",
        "source_ledger_sha256",
        "created_at_utc",
        "entries",
    }
)
MANIFEST_ENTRY_FIELDS = frozenset(
    {
        "source_entry_id",
        "reservation_id",
        "bundle_id",
        "execution_id",
        "amount_usd",
        "state",
    }
)
SOURCE_FIELDS = frozenset({"cap_usd", "entries"})
SOURCE_ENTRY_FIELDS = frozenset(
    {"id", "label", "amount_usd", "status", "created_at", "finalized_at"}
)


@dataclass(frozen=True)
class Reservation:
    id: str
    amount: Decimal
    bundle_id: str
    execution_id: str


@dataclass(frozen=True)
class _IntegritySnapshot:
    committed: Decimal
    states: dict[str, str]
    amounts: dict[str, Decimal]
    head_hash: str


@dataclass(frozen=True)
class _ReservationEffect:
    reservation_id: str
    bundle_id: str
    execution_id: str
    amount_usd: str
    created_at_utc: str
    migration_id: str | None

    def database_row(self) -> tuple[Any, ...]:
        return (
            self.reservation_id,
            self.bundle_id,
            self.execution_id,
            self.amount_usd,
            self.created_at_utc,
            self.migration_id,
        )


@dataclass(frozen=True)
class _EventEffect:
    event_index: int
    event_id: str
    reservation_id: str
    from_state: str | None
    to_state: str
    amount_usd: str
    created_at_utc: str
    previous_hash: str
    reason_code: str
    evidence_sha256: str | None
    event_hash: str

    def database_row(self) -> tuple[Any, ...]:
        return (
            self.event_index,
            self.event_id,
            self.reservation_id,
            self.from_state,
            self.to_state,
            self.amount_usd,
            self.created_at_utc,
            self.previous_hash,
            self.reason_code,
            self.evidence_sha256,
            self.event_hash,
        )


TABLE_STATEMENTS = (
    """
    CREATE TABLE pilot (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        schema_version TEXT NOT NULL,
        pilot_id TEXT NOT NULL UNIQUE,
        ledger_id TEXT NOT NULL UNIQUE,
        cap_usd TEXT NOT NULL,
        migration_schema_version TEXT NOT NULL,
        migration_id TEXT NOT NULL UNIQUE,
        migration_manifest_sha256 TEXT NOT NULL,
        source_ledger_sha256 TEXT NOT NULL,
        created_at_utc TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE reservations (
        reservation_id TEXT PRIMARY KEY,
        bundle_id TEXT NOT NULL,
        execution_id TEXT NOT NULL,
        amount_usd TEXT NOT NULL,
        created_at_utc TEXT NOT NULL,
        migration_id TEXT,
        FOREIGN KEY (migration_id) REFERENCES pilot(migration_id)
    )
    """,
    """
    CREATE TABLE migration_entries (
        entry_index INTEGER PRIMARY KEY,
        migration_id TEXT NOT NULL,
        source_entry_id TEXT NOT NULL UNIQUE,
        reservation_id TEXT NOT NULL UNIQUE,
        bundle_id TEXT NOT NULL,
        execution_id TEXT NOT NULL,
        amount_usd TEXT NOT NULL,
        state TEXT NOT NULL,
        FOREIGN KEY (migration_id) REFERENCES pilot(migration_id),
        FOREIGN KEY (reservation_id) REFERENCES reservations(reservation_id)
    )
    """,
    """
    CREATE TABLE events (
        event_index INTEGER PRIMARY KEY,
        event_id TEXT NOT NULL UNIQUE,
        reservation_id TEXT NOT NULL,
        from_state TEXT,
        to_state TEXT NOT NULL,
        amount_usd TEXT NOT NULL,
        created_at_utc TEXT NOT NULL,
        previous_hash TEXT NOT NULL,
        reason_code TEXT NOT NULL,
        evidence_sha256 TEXT,
        event_hash TEXT NOT NULL UNIQUE,
        FOREIGN KEY (reservation_id) REFERENCES reservations(reservation_id)
    )
    """,
)

INDEX_STATEMENTS = (
    "CREATE UNIQUE INDEX reservations_bundle_execution_unique ON reservations(bundle_id, execution_id)",
    "CREATE UNIQUE INDEX reservations_bundle_unique ON reservations(bundle_id)",
    "CREATE UNIQUE INDEX reservations_execution_unique ON reservations(execution_id)",
    "CREATE INDEX events_reservation_index ON events(reservation_id, event_index)",
)

TRIGGER_STATEMENTS = (
    """
    CREATE TRIGGER pilot_no_insert
    BEFORE INSERT ON pilot
    WHEN EXISTS (SELECT 1 FROM pilot)
    BEGIN SELECT RAISE(ABORT, 'pilot metadata is immutable'); END
    """,
    """
    CREATE TRIGGER pilot_no_update
    BEFORE UPDATE ON pilot
    BEGIN SELECT RAISE(ABORT, 'pilot metadata is immutable'); END
    """,
    """
    CREATE TRIGGER pilot_no_delete
    BEFORE DELETE ON pilot
    BEGIN SELECT RAISE(ABORT, 'pilot metadata is immutable'); END
    """,
    """
    CREATE TRIGGER migration_entries_no_insert
    BEFORE INSERT ON migration_entries
    BEGIN SELECT RAISE(ABORT, 'migration history is immutable'); END
    """,
    """
    CREATE TRIGGER migration_entries_no_update
    BEFORE UPDATE ON migration_entries
    BEGIN SELECT RAISE(ABORT, 'migration history is immutable'); END
    """,
    """
    CREATE TRIGGER migration_entries_no_delete
    BEFORE DELETE ON migration_entries
    BEGIN SELECT RAISE(ABORT, 'migration history is immutable'); END
    """,
    """
    CREATE TRIGGER reservations_no_update
    BEFORE UPDATE ON reservations
    BEGIN SELECT RAISE(ABORT, 'reservations are immutable'); END
    """,
    """
    CREATE TRIGGER reservations_no_delete
    BEFORE DELETE ON reservations
    BEGIN SELECT RAISE(ABORT, 'reservations are immutable'); END
    """,
    """
    CREATE TRIGGER events_no_update
    BEFORE UPDATE ON events
    BEGIN SELECT RAISE(ABORT, 'events are append-only'); END
    """,
    """
    CREATE TRIGGER events_no_delete
    BEFORE DELETE ON events
    BEGIN SELECT RAISE(ABORT, 'events are append-only'); END
    """,
)

TABLE_NAMES = ("pilot", "reservations", "migration_entries", "events")
INDEX_OBJECTS = (
    ("reservations_bundle_execution_unique", "reservations"),
    ("reservations_bundle_unique", "reservations"),
    ("reservations_execution_unique", "reservations"),
    ("events_reservation_index", "events"),
)
TRIGGER_OBJECTS = (
    ("pilot_no_insert", "pilot"),
    ("pilot_no_update", "pilot"),
    ("pilot_no_delete", "pilot"),
    ("migration_entries_no_insert", "migration_entries"),
    ("migration_entries_no_update", "migration_entries"),
    ("migration_entries_no_delete", "migration_entries"),
    ("reservations_no_update", "reservations"),
    ("reservations_no_delete", "reservations"),
    ("events_no_update", "events"),
    ("events_no_delete", "events"),
)
IMPLICIT_INDEX_OBJECTS = (
    ("sqlite_autoindex_pilot_1", "pilot"),
    ("sqlite_autoindex_pilot_2", "pilot"),
    ("sqlite_autoindex_pilot_3", "pilot"),
    ("sqlite_autoindex_reservations_1", "reservations"),
    ("sqlite_autoindex_migration_entries_1", "migration_entries"),
    ("sqlite_autoindex_migration_entries_2", "migration_entries"),
    ("sqlite_autoindex_events_1", "events"),
    ("sqlite_autoindex_events_2", "events"),
)


def _normalize_sql(value: str) -> str:
    return " ".join(value.split()).strip().removesuffix(";")


def _expected_schema_objects(*, staged: bool) -> frozenset[tuple[str, str, str, str | None]]:
    objects: set[tuple[str, str, str, str | None]] = {
        ("table", name, name, _normalize_sql(statement))
        for name, statement in zip(TABLE_NAMES, TABLE_STATEMENTS, strict=True)
    }
    objects.update(
        ("index", name, table, _normalize_sql(statement))
        for (name, table), statement in zip(
            INDEX_OBJECTS,
            INDEX_STATEMENTS,
            strict=True,
        )
    )
    objects.update(
        ("index", name, table, None) for name, table in IMPLICIT_INDEX_OBJECTS
    )
    if not staged:
        objects.update(
            ("trigger", name, table, _normalize_sql(statement))
            for (name, table), statement in zip(
                TRIGGER_OBJECTS,
                TRIGGER_STATEMENTS,
                strict=True,
            )
        )
    return frozenset(objects)


EXPECTED_STAGED_SCHEMA_OBJECTS = _expected_schema_objects(staged=True)
EXPECTED_SCHEMA_OBJECTS = _expected_schema_objects(staged=False)


def _exact_object(value: Any, fields: frozenset[str], *, label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} schema mismatch")
    return value


def _strict_json_bytes(content: bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("JSON object contains a duplicate key")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        del value
        raise ValueError("JSON contains a non-finite number")

    return json.loads(
        content.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject_constant,
    )


def _canonical_money(value: Any, *, positive: bool, label: str) -> Decimal:
    if type(value) is not str or MONEY_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be canonical money")
    try:
        amount = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{label} must be canonical money") from exc
    if positive and amount <= 0:
        raise ValueError(f"{label} must be canonical money")
    return amount


def _reservation_money(value: Any) -> tuple[Decimal, str]:
    if type(value) is Decimal:
        text = str(value)
        if value.as_tuple().exponent != -4:
            raise ValueError("reservation amount must be canonical money")
    elif type(value) is str:
        text = value
    else:
        raise ValueError("reservation amount must be canonical money")
    amount = _canonical_money(text, positive=True, label="reservation amount")
    return amount, text


def _sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _bundle_identity(value: Any) -> str:
    if type(value) is not str or BUNDLE_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("bundle_id must be a content-addressed SHA-256")
    return value


def _typed_id(value: Any, pattern: re.Pattern[str], *, label: str) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise ValueError(f"{label} must be a typed opaque UUIDv4")
    return value


def _source_uuid(value: Any) -> str:
    if type(value) is not str:
        raise ValueError("legacy entry identity must be an opaque UUIDv4")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError("legacy entry identity must be an opaque UUIDv4") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError("legacy entry identity must be an opaque UUIDv4")
    return value


def _utc_timestamp(value: Any, *, label: str) -> str:
    if type(value) is not str or UTC_TIMESTAMP_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical UTC timestamp")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError(f"{label} must be a canonical UTC timestamp") from exc
    return value


def _new_typed_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _deterministic_typed_id(prefix: str, seed: bytes) -> str:
    raw = bytearray(hashlib.sha256(seed).digest()[:16])
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return f"{prefix}_{uuid.UUID(bytes=bytes(raw)).hex}"


def _current_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _is_alias_component(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(os.path, "isjunction", None)
        if is_junction is not None and is_junction(path):
            return True
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
        reparse_flag = getattr(__import__("stat"), "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        return bool(attributes & reparse_flag)
    except OSError:
        return True


def _has_alias_component(path: Path) -> bool:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.exists() and _is_alias_component(current):
            return True
    return False


def _canonical_existing_file(path: Path, *, label: str) -> Path:
    candidate = Path(path)
    canonical_label = "ledger" if label == "ledger database" else label
    if not candidate.exists() or not candidate.is_file():
        raise ValueError(f"{label} is required")
    if _has_alias_component(candidate):
        raise ValueError(f"canonical {canonical_label} path is required")
    resolved = candidate.resolve(strict=True)
    absolute = Path(os.path.abspath(candidate))
    if os.path.normcase(str(resolved)) != os.path.normcase(str(absolute)):
        raise ValueError(f"canonical {canonical_label} path is required")
    try:
        if resolved.stat().st_nlink != 1:
            raise ValueError(f"canonical {canonical_label} path is required")
    except OSError as exc:
        raise ValueError(f"canonical {canonical_label} path is required") from exc
    return resolved


def _path_entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ValueError("destination ledger path cannot be verified") from exc
    return True


def _reject_destination_sidecars(destination: Path) -> None:
    if any(
        _path_entry_exists(Path(f"{destination}{suffix}"))
        for suffix in SQLITE_SIDECAR_SUFFIXES
    ):
        raise ValueError("destination ledger sidecar exists")


def _preflight_destination_entries(destination: Path) -> None:
    if _path_entry_exists(destination):
        raise ValueError("destination ledger already exists")
    _reject_destination_sidecars(destination)


def _canonical_destination(path: Path) -> Path:
    candidate = Path(path)
    _preflight_destination_entries(candidate)
    parent = candidate.parent
    if not parent.exists() or not parent.is_dir() or _has_alias_component(parent):
        raise ValueError("canonical destination directory is required")
    resolved_parent = parent.resolve(strict=True)
    absolute_parent = Path(os.path.abspath(parent))
    if os.path.normcase(str(resolved_parent)) != os.path.normcase(str(absolute_parent)):
        raise ValueError("canonical destination directory is required")
    destination = resolved_parent / candidate.name
    _preflight_destination_entries(destination)
    return destination


def _object_identity(stat_result: os.stat_result) -> tuple[int, int]:
    return stat_result.st_dev, stat_result.st_ino


def _ensure_path_identity(
    path: Path,
    anchor: os.stat_result,
    *,
    expected_links: int,
) -> None:
    try:
        current = path.stat()
    except OSError as exc:
        raise ValueError("ledger changed during verification") from exc
    if (
        _object_identity(current) != _object_identity(anchor)
        or current.st_nlink != expected_links
    ):
        raise ValueError("ledger changed during verification")


@contextmanager
def _identity_anchor(path: Path) -> Iterator[os.stat_result]:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_BINARY", 0),
        )
        anchor = os.fstat(descriptor)
        _ensure_path_identity(path, anchor, expected_links=1)
        yield anchor
    except OSError as exc:
        raise ValueError("ledger changed during verification") from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    try:
        content = path.read_bytes()
        value = _strict_json_bytes(content)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError("migration manifest is invalid") from exc
    if type(value) is not dict:
        raise ValueError("migration manifest schema mismatch")
    _validate_manifest_value(value)
    if content != canonical_json_bytes(value):
        raise ValueError("migration manifest must be canonical JSON")
    manifest = value
    return manifest, hashlib.sha256(content).hexdigest()


def _load_and_validate_source(path: Path, manifest: dict[str, Any]) -> None:
    try:
        first_digest = sha256_file(path).sha256
        content = path.read_bytes()
    except OSError as exc:
        raise ValueError("legacy source ledger is invalid") from exc
    digest = hashlib.sha256(content).hexdigest()
    if digest != first_digest:
        raise ValueError("source ledger changed during migration")
    if digest != manifest["source_ledger_sha256"]:
        raise ValueError("source ledger hash mismatch")
    try:
        value = _strict_json_bytes(content)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError("legacy source ledger is invalid") from exc
    source = _exact_object(value, SOURCE_FIELDS, label="legacy source ledger")
    if source["cap_usd"] != CAP_USD_TEXT:
        raise ValueError("legacy source cap mismatch")
    entries = source["entries"]
    if type(entries) is not list or len(entries) != LEGACY_ENTRY_COUNT:
        raise ValueError("legacy source ledger must contain exactly six entries")
    seen_source_ids: set[str] = set()
    for source_item, manifest_item in zip(entries, manifest["entries"], strict=True):
        if type(source_item) is not dict:
            raise ValueError("legacy source entry schema mismatch")
        keys = set(source_item)
        required = SOURCE_ENTRY_FIELDS - {"finalized_at"}
        if not required <= keys or not keys <= SOURCE_ENTRY_FIELDS:
            raise ValueError("legacy source entry schema mismatch")
        source_id = _source_uuid(source_item["id"])
        if source_id in seen_source_ids:
            raise ValueError("legacy source ledger contains a duplicate identity")
        seen_source_ids.add(source_id)
        if type(source_item["label"]) is not str:
            raise ValueError("legacy source entry schema mismatch")
        amount = _canonical_money(
            source_item["amount_usd"], positive=True, label="legacy amount_usd"
        )
        state = source_item["status"]
        if type(state) is not str or state not in {"reserved", "consumed", "released"}:
            raise ValueError("legacy source entry state is invalid")
        if type(source_item["created_at"]) is not int:
            raise ValueError("legacy source entry schema mismatch")
        if "finalized_at" in source_item and type(source_item["finalized_at"]) is not int:
            raise ValueError("legacy source entry schema mismatch")
        if (
            source_id != manifest_item["source_entry_id"]
            or amount != Decimal(manifest_item["amount_usd"])
            or state != manifest_item["state"]
        ):
            raise ValueError("migration entry does not match legacy source")


def _event_payload(
    *,
    event_id: str,
    reservation_id: str,
    from_state: str | None,
    to_state: str,
    amount_usd: str,
    created_at_utc: str,
    previous_hash: str,
    reason_code: str,
    evidence_sha256: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": EVENT_SCHEMA_VERSION,
        "event_id": event_id,
        "reservation_id": reservation_id,
        "from_state": from_state,
        "to_state": to_state,
        "amount_usd": amount_usd,
        "created_at_utc": created_at_utc,
        "previous_hash": previous_hash,
        "reason_code": reason_code,
        "evidence_sha256": evidence_sha256,
    }


def _event_hash(**event_fields: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(_event_payload(**event_fields))).hexdigest()


def _configure_connection(connection: sqlite3.Connection) -> None:
    connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MILLISECONDS}")
    connection.execute("PRAGMA foreign_keys = ON")


@contextmanager
def _read_connection(path: Path) -> Iterator[sqlite3.Connection]:
    uri = f"{path.as_uri()}?mode=ro"
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            uri,
            uri=True,
            timeout=BUSY_TIMEOUT_MILLISECONDS / 1_000,
            isolation_level=None,
        )
        _configure_connection(connection)
        connection.execute("BEGIN")
        yield connection
    finally:
        if connection is not None:
            if connection.in_transaction:
                connection.rollback()
            connection.close()


@contextmanager
def _write_connection(path: Path) -> Iterator[sqlite3.Connection]:
    uri = f"{path.as_uri()}?mode=rw"
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            uri,
            uri=True,
            timeout=BUSY_TIMEOUT_MILLISECONDS / 1_000,
            isolation_level=None,
        )
        _configure_connection(connection)
        connection.execute("PRAGMA synchronous = FULL")
        yield connection
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


def _rollback_quietly(connection: sqlite3.Connection) -> None:
    try:
        connection.rollback()
    except Exception:
        pass


def _begin_immediate(connection: sqlite3.Connection) -> None:
    try:
        connection.execute("BEGIN IMMEDIATE")
    except sqlite3.OperationalError as exc:
        if "locked" in str(exc).lower() or "busy" in str(exc).lower():
            raise RuntimeError("budget ledger is busy") from None
        raise ValueError("ledger transaction failed") from exc


def _append_event(
    connection: sqlite3.Connection,
    *,
    snapshot: _IntegritySnapshot,
    event_id: str,
    reservation_id: str,
    from_state: str | None,
    to_state: str,
    amount_text: str,
    created_at_utc: str,
    reason_code: str,
    evidence_sha256: str | None = None,
) -> _EventEffect:
    row = connection.execute(
        "SELECT COALESCE(MAX(event_index), 0) + 1 FROM events"
    ).fetchone()
    event_index = row[0]
    fields = {
        "event_id": event_id,
        "reservation_id": reservation_id,
        "from_state": from_state,
        "to_state": to_state,
        "amount_usd": amount_text,
        "created_at_utc": created_at_utc,
        "previous_hash": snapshot.head_hash,
        "reason_code": reason_code,
        "evidence_sha256": evidence_sha256,
    }
    digest = _event_hash(**fields)
    effect = _EventEffect(
        event_index=event_index,
        event_id=fields["event_id"],
        reservation_id=fields["reservation_id"],
        from_state=fields["from_state"],
        to_state=fields["to_state"],
        amount_usd=fields["amount_usd"],
        created_at_utc=fields["created_at_utc"],
        previous_hash=fields["previous_hash"],
        reason_code=fields["reason_code"],
        evidence_sha256=fields["evidence_sha256"],
        event_hash=digest,
    )
    connection.execute(
        """
        INSERT INTO events (
            event_index, event_id, reservation_id, from_state, to_state,
            amount_usd, created_at_utc, previous_hash, reason_code,
            evidence_sha256, event_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        effect.database_row(),
    )
    return effect


def _create_schema(connection: sqlite3.Connection) -> None:
    for statement in TABLE_STATEMENTS:
        connection.execute(statement)
    for statement in INDEX_STATEMENTS:
        connection.execute(statement)
    connection.execute(f"PRAGMA user_version = {STAGED_SQLITE_USER_VERSION}")


def _create_immutability_triggers(connection: sqlite3.Connection) -> None:
    for statement in TRIGGER_STATEMENTS:
        connection.execute(statement)


def _initialize_database(
    path: Path,
    manifest: dict[str, Any],
    manifest_digest: str,
) -> None:
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            path,
            timeout=BUSY_TIMEOUT_MILLISECONDS / 1_000,
            isolation_level=None,
        )
        _configure_connection(connection)
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("BEGIN IMMEDIATE")
        _create_schema(connection)
        connection.execute(
            """
            INSERT INTO pilot (
                singleton, schema_version, pilot_id, ledger_id, cap_usd,
                migration_schema_version, migration_id,
                migration_manifest_sha256, source_ledger_sha256, created_at_utc
            ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                STAGED_LEDGER_SCHEMA_VERSION,
                manifest["pilot_id"],
                manifest["ledger_id"],
                manifest["cap_usd"],
                manifest["schema_version"],
                manifest["migration_id"],
                manifest_digest,
                manifest["source_ledger_sha256"],
                manifest["created_at_utc"],
            ),
        )
        previous_hash = GENESIS_HASH
        for index, entry in enumerate(manifest["entries"]):
            connection.execute(
                """
                INSERT INTO reservations (
                    reservation_id, bundle_id, execution_id, amount_usd,
                    created_at_utc, migration_id
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    entry["reservation_id"],
                    entry["bundle_id"],
                    entry["execution_id"],
                    entry["amount_usd"],
                    manifest["created_at_utc"],
                    manifest["migration_id"],
                ),
            )
            connection.execute(
                """
                INSERT INTO migration_entries (
                    entry_index, migration_id, source_entry_id, reservation_id,
                    bundle_id, execution_id, amount_usd, state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    index,
                    manifest["migration_id"],
                    entry["source_entry_id"],
                    entry["reservation_id"],
                    entry["bundle_id"],
                    entry["execution_id"],
                    entry["amount_usd"],
                    entry["state"],
                ),
            )
            event_id = _deterministic_typed_id(
                "event",
                f"{manifest_digest}:{index}".encode("ascii"),
            )
            fields = {
                "event_id": event_id,
                "reservation_id": entry["reservation_id"],
                "from_state": None,
                "to_state": entry["state"],
                "amount_usd": entry["amount_usd"],
                "created_at_utc": manifest["created_at_utc"],
                "previous_hash": previous_hash,
                "reason_code": "legacy_migration",
                "evidence_sha256": None,
            }
            digest = _event_hash(**fields)
            connection.execute(
                """
                INSERT INTO events (
                    event_index, event_id, reservation_id, from_state, to_state,
                    amount_usd, created_at_utc, previous_hash, reason_code,
                    evidence_sha256, event_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    index + 1,
                    fields["event_id"],
                    fields["reservation_id"],
                    fields["from_state"],
                    fields["to_state"],
                    fields["amount_usd"],
                    fields["created_at_utc"],
                    fields["previous_hash"],
                    fields["reason_code"],
                    fields["evidence_sha256"],
                    digest,
                ),
            )
            previous_hash = digest
        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchall()
        if integrity != [("ok",)]:
            raise ValueError("SQLite integrity check failed")
    except Exception:
        if connection is not None:
            try:
                connection.rollback()
            except sqlite3.Error:
                pass
        raise
    finally:
        if connection is not None:
            connection.close()


def _read_pilot(connection: sqlite3.Connection) -> tuple[Any, ...]:
    try:
        rows = connection.execute(
            """
            SELECT schema_version, pilot_id, ledger_id, cap_usd,
                   migration_schema_version, migration_id,
                   migration_manifest_sha256, source_ledger_sha256,
                   created_at_utc
            FROM pilot WHERE singleton = 1
            """
        ).fetchall()
    except sqlite3.DatabaseError as exc:
        if "no such table" in str(exc).lower():
            raise ValueError("migration manifest is required") from None
        raise ValueError("SQLite integrity check failed") from exc
    if len(rows) != 1:
        raise ValueError("migration manifest is required")
    return rows[0]


def _verify_sqlite_integrity(connection: sqlite3.Connection) -> None:
    try:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()
        if journal_mode != ("delete",):
            raise ValueError("journal mode mismatch")
        result = connection.execute("PRAGMA integrity_check").fetchall()
    except sqlite3.DatabaseError as exc:
        raise ValueError("SQLite integrity check failed") from exc
    if result != [("ok",)]:
        raise ValueError("SQLite integrity check failed")
    try:
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("foreign key integrity check failed")
    except sqlite3.DatabaseError as exc:
        raise ValueError("SQLite integrity check failed") from exc


def _verify_schema(
    connection: sqlite3.Connection,
    *,
    staged: bool,
) -> None:
    try:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        expected_version = (
            STAGED_SQLITE_USER_VERSION if staged else SQLITE_USER_VERSION
        )
        if version != expected_version:
            raise ValueError("schema version mismatch")
        rows = connection.execute(
            """
            SELECT type, name, tbl_name, sql
            FROM sqlite_master
            """
        ).fetchall()
    except sqlite3.DatabaseError as exc:
        raise ValueError("SQLite integrity check failed") from exc
    actual = frozenset(
        (
            row[0],
            row[1],
            row[2],
            None if row[3] is None else _normalize_sql(row[3]),
        )
        for row in rows
    )
    expected = (
        EXPECTED_STAGED_SCHEMA_OBJECTS if staged else EXPECTED_SCHEMA_OBJECTS
    )
    if actual != expected:
        raise ValueError("ledger schema mismatch")


def _manifest_from_database(
    pilot: Sequence[Any],
    migration_rows: list[tuple[Any, ...]],
) -> dict[str, Any]:
    return {
        "schema_version": pilot[4],
        "pilot_id": pilot[1],
        "ledger_id": pilot[2],
        "migration_id": pilot[5],
        "cap_usd": pilot[3],
        "source_ledger_sha256": pilot[7],
        "created_at_utc": pilot[8],
        "entries": [
            {
                "source_entry_id": row[1],
                "reservation_id": row[2],
                "bundle_id": row[3],
                "execution_id": row[4],
                "amount_usd": row[5],
                "state": row[6],
            }
            for row in migration_rows
        ],
    }


def _verify_connection(
    connection: sqlite3.Connection,
    *,
    expected_pilot_id: str,
    expected_ledger_id: str,
    staged: bool = False,
) -> _IntegritySnapshot:
    _verify_sqlite_integrity(connection)
    pilot = _read_pilot(connection)
    try:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
    except sqlite3.DatabaseError as exc:
        raise ValueError("SQLite integrity check failed") from exc
    expected_version = STAGED_SQLITE_USER_VERSION if staged else SQLITE_USER_VERSION
    if version != expected_version:
        raise ValueError("schema version mismatch")
    expected_schema_version = (
        STAGED_LEDGER_SCHEMA_VERSION if staged else LEDGER_SCHEMA_VERSION
    )
    if pilot[0] != expected_schema_version:
        raise ValueError("schema version mismatch")
    if pilot[1] != expected_pilot_id:
        raise ValueError("pilot identity mismatch")
    if pilot[2] != expected_ledger_id:
        raise ValueError("ledger identity mismatch")
    _typed_id(pilot[1], PILOT_ID_PATTERN, label="pilot_id")
    _typed_id(pilot[2], LEDGER_ID_PATTERN, label="ledger_id")
    if pilot[3] != CAP_USD_TEXT:
        raise ValueError("ledger cap mismatch")
    if pilot[4] != MIGRATION_SCHEMA_VERSION:
        raise ValueError("migration schema version mismatch")
    _typed_id(pilot[5], MIGRATION_ID_PATTERN, label="migration_id")
    _sha256(pilot[6], label="migration manifest digest")
    _sha256(pilot[7], label="source ledger hash")
    _utc_timestamp(pilot[8], label="migration timestamp")
    try:
        migration_rows = connection.execute(
            """
            SELECT entry_index, source_entry_id, reservation_id, bundle_id,
                   execution_id, amount_usd, state
            FROM migration_entries ORDER BY entry_index
            """
        ).fetchall()
    except sqlite3.DatabaseError as exc:
        raise ValueError("migration manifest is required") from exc
    if [row[0] for row in migration_rows] != list(range(LEGACY_ENTRY_COUNT)):
        raise ValueError("migration manifest entry mismatch")
    reconstructed = _manifest_from_database(pilot, migration_rows)
    try:
        _validate_manifest_value(reconstructed)
    except ValueError as exc:
        raise ValueError("migration manifest entry mismatch") from exc
    digest = hashlib.sha256(canonical_json_bytes(reconstructed)).hexdigest()
    if digest != pilot[6]:
        raise ValueError("migration manifest digest mismatch")

    try:
        reservation_rows = connection.execute(
            """
            SELECT reservation_id, bundle_id, execution_id, amount_usd,
                   created_at_utc, migration_id
            FROM reservations ORDER BY rowid
            """
        ).fetchall()
        event_rows = connection.execute(
            """
            SELECT event_index, event_id, reservation_id, from_state, to_state,
                   amount_usd, created_at_utc, previous_hash, reason_code,
                   evidence_sha256, event_hash
            FROM events ORDER BY event_index
            """
        ).fetchall()
    except sqlite3.DatabaseError as exc:
        raise ValueError("ledger event history is invalid") from exc
    reservations: dict[str, tuple[Any, ...]] = {}
    bundle_ids: set[str] = set()
    execution_ids: set[str] = set()
    migration_reservations = {row[2]: row for row in migration_rows}
    for row in reservation_rows:
        (
            reservation_id,
            bundle_id,
            execution_id,
            amount_text,
            created_at,
            reservation_migration_id,
        ) = row
        _typed_id(reservation_id, RESERVATION_ID_PATTERN, label="reservation_id")
        _bundle_identity(bundle_id)
        _typed_id(execution_id, EXECUTION_ID_PATTERN, label="execution_id")
        _canonical_money(amount_text, positive=True, label="amount_usd")
        _utc_timestamp(created_at, label="reservation timestamp")
        if (
            reservation_id in reservations
            or bundle_id in bundle_ids
            or execution_id in execution_ids
        ):
            raise ValueError("reservation replay identity mismatch")
        reservations[reservation_id] = row
        bundle_ids.add(bundle_id)
        execution_ids.add(execution_id)
        migration_row = migration_reservations.get(reservation_id)
        if migration_row is not None:
            if (
                bundle_id != migration_row[3]
                or execution_id != migration_row[4]
                or amount_text != migration_row[5]
                or created_at != pilot[8]
                or reservation_migration_id != pilot[5]
            ):
                raise ValueError("migration reservation mismatch")
        elif reservation_migration_id is not None:
            raise ValueError("reservation migration identity mismatch")

    states: dict[str, str] = {}
    amounts: dict[str, Decimal] = {}
    previous_hash = GENESIS_HASH
    for expected_index, row in enumerate(event_rows, start=1):
        (
            event_index,
            event_id,
            reservation_id,
            from_state,
            to_state,
            amount_text,
            created_at,
            stored_previous,
            reason_code,
            evidence_sha256,
            stored_hash,
        ) = row
        if event_index != expected_index:
            raise ValueError("event chain mismatch")
        _typed_id(event_id, EVENT_ID_PATTERN, label="event_id")
        if reservation_id not in reservations:
            raise ValueError("event reservation mismatch")
        amount = _canonical_money(amount_text, positive=True, label="event amount_usd")
        if amount_text != reservations[reservation_id][3]:
            raise ValueError("event amount mismatch")
        _utc_timestamp(created_at, label="event timestamp")
        _sha256(stored_previous, label="previous_hash")
        _sha256(stored_hash, label="event_hash")
        if evidence_sha256 is not None:
            _sha256(evidence_sha256, label="evidence_sha256")
        if stored_previous != previous_hash:
            raise ValueError("event chain mismatch")
        current = states.get(reservation_id)
        if from_state != current:
            raise ValueError("event state chain mismatch")
        if type(to_state) is not str or to_state not in ALL_STATES:
            raise ValueError("event state is invalid")
        if type(reason_code) is not str or not reason_code:
            raise ValueError("event reason code is invalid")
        if current is None:
            migration_row = migration_reservations.get(reservation_id)
            if migration_row is not None:
                if (
                    reason_code != "legacy_migration"
                    or evidence_sha256 is not None
                    or to_state != migration_row[6]
                    or amount_text != migration_row[5]
                ):
                    raise ValueError("migration event mismatch")
            elif (
                to_state != "reserved"
                or reason_code != "reservation_created"
                or evidence_sha256 is not None
            ):
                raise ValueError("reservation event mismatch")
        else:
            edge = (current, to_state)
            if reason_code == "state_transition":
                valid_transition = (
                    edge in NORMAL_TRANSITIONS and evidence_sha256 is None
                )
            elif reason_code == "pre_submit_release":
                valid_transition = (
                    edge in PRE_SUBMIT_RELEASE_TRANSITIONS
                    and evidence_sha256 is None
                )
            elif reason_code == "provider_reconciliation":
                valid_transition = (
                    edge in RECONCILIATION_TRANSITIONS
                    and evidence_sha256 is not None
                )
            else:
                valid_transition = False
            if not valid_transition:
                raise ValueError("event state transition mismatch")
        fields = {
            "event_id": event_id,
            "reservation_id": reservation_id,
            "from_state": from_state,
            "to_state": to_state,
            "amount_usd": amount_text,
            "created_at_utc": created_at,
            "previous_hash": stored_previous,
            "reason_code": reason_code,
            "evidence_sha256": evidence_sha256,
        }
        if _event_hash(**fields) != stored_hash:
            raise ValueError("event chain mismatch")
        states[reservation_id] = to_state
        amounts[reservation_id] = amount
        previous_hash = stored_hash
    if set(states) != set(reservations):
        raise ValueError("reservation event history mismatch")
    committed = sum(
        (amounts[key] for key, state in states.items() if state in COMMITTED_STATES),
        Decimal("0.0000"),
    )
    if committed < 0 or committed > CAP_USD:
        raise ValueError("derived ledger balance is invalid")
    _verify_schema(connection, staged=staged)
    return _IntegritySnapshot(
        committed=committed,
        states=states,
        amounts=amounts,
        head_hash=previous_hash,
    )


def _validate_manifest_value(manifest: dict[str, Any]) -> None:
    _exact_object(manifest, MANIFEST_FIELDS, label="migration manifest")
    if manifest["schema_version"] != MIGRATION_SCHEMA_VERSION:
        raise ValueError("migration manifest schema version mismatch")
    _typed_id(manifest["pilot_id"], PILOT_ID_PATTERN, label="pilot_id")
    _typed_id(manifest["ledger_id"], LEDGER_ID_PATTERN, label="ledger_id")
    _typed_id(manifest["migration_id"], MIGRATION_ID_PATTERN, label="migration_id")
    if manifest["cap_usd"] != CAP_USD_TEXT:
        raise ValueError("migration cap mismatch")
    _sha256(manifest["source_ledger_sha256"], label="source ledger hash")
    _utc_timestamp(manifest["created_at_utc"], label="migration timestamp")
    entries = manifest["entries"]
    if type(entries) is not list or len(entries) != LEGACY_ENTRY_COUNT:
        raise ValueError("migration manifest requires exactly six entries")
    seen: dict[str, set[str]] = {
        "source_entry_id": set(),
        "reservation_id": set(),
        "bundle_id": set(),
        "execution_id": set(),
    }
    committed = Decimal("0.0000")
    for value in entries:
        entry = _exact_object(value, MANIFEST_ENTRY_FIELDS, label="migration entry")
        _source_uuid(entry["source_entry_id"])
        _typed_id(entry["reservation_id"], RESERVATION_ID_PATTERN, label="reservation_id")
        _bundle_identity(entry["bundle_id"])
        _typed_id(entry["execution_id"], EXECUTION_ID_PATTERN, label="execution_id")
        amount = _canonical_money(entry["amount_usd"], positive=True, label="amount_usd")
        if type(entry["state"]) is not str or entry["state"] not in ALL_STATES:
            raise ValueError("migration entry state is invalid")
        for field in seen:
            if entry[field] in seen[field]:
                raise ValueError("migration manifest contains a duplicate identity")
            seen[field].add(entry[field])
        if entry["state"] in COMMITTED_STATES:
            committed += amount
    if committed != LEGACY_COMMITTED_USD:
        raise ValueError("migration committed total mismatch")


def _commit_error_left_verified_write_effect(
    connection: sqlite3.Connection,
    canonical: Path,
    *,
    expected_pilot_id: str,
    expected_ledger_id: str,
    event: _EventEffect,
    reservation: _ReservationEffect | None = None,
) -> bool:
    try:
        transaction_is_active = connection.in_transaction
    except Exception:
        transaction_is_active = False
    if transaction_is_active:
        return False
    try:
        connection.close()
    except Exception:
        pass
    verified_on_reopen = False
    try:
        with _read_connection(canonical) as reopened:
            _verify_connection(
                reopened,
                expected_pilot_id=expected_pilot_id,
                expected_ledger_id=expected_ledger_id,
            )
            event_rows = reopened.execute(
                """
                SELECT event_index, event_id, reservation_id, from_state,
                       to_state, amount_usd, created_at_utc, previous_hash,
                       reason_code, evidence_sha256, event_hash
                FROM events WHERE event_id = ?
                """,
                (event.event_id,),
            ).fetchall()
            if event_rows != [event.database_row()]:
                raise ValueError("committed event readback mismatch")
            if reservation is not None:
                reservation_rows = reopened.execute(
                    """
                    SELECT reservation_id, bundle_id, execution_id, amount_usd,
                           created_at_utc, migration_id
                    FROM reservations WHERE reservation_id = ?
                    """,
                    (reservation.reservation_id,),
                ).fetchall()
                if reservation_rows != [reservation.database_row()]:
                    raise ValueError("committed reservation readback mismatch")
            verified_on_reopen = True
    except Exception:
        pass
    return verified_on_reopen


def _commit_with_verified_write_effect(
    connection: sqlite3.Connection,
    canonical: Path,
    *,
    expected_pilot_id: str,
    expected_ledger_id: str,
    event: _EventEffect,
    reservation: _ReservationEffect | None = None,
) -> None:
    try:
        connection.commit()
    except Exception:
        if not _commit_error_left_verified_write_effect(
            connection,
            canonical,
            expected_pilot_id=expected_pilot_id,
            expected_ledger_id=expected_ledger_id,
            event=event,
            reservation=reservation,
        ):
            raise


class PilotLedger:
    """Canonical SQLite pilot ledger opened only after full integrity verification."""

    def __init__(
        self,
        path: Path,
        expected_pilot_id: str,
        expected_ledger_id: str,
    ) -> None:
        self.path = path
        self._expected_pilot_id = expected_pilot_id
        self._expected_ledger_id = expected_ledger_id
        self.pilot_id = expected_pilot_id
        self.ledger_id = expected_ledger_id

    @classmethod
    def open_existing(
        cls,
        path: Path,
        expected_pilot_id: str,
        *,
        expected_ledger_id: str,
    ) -> PilotLedger:
        _typed_id(expected_pilot_id, PILOT_ID_PATTERN, label="expected pilot_id")
        _typed_id(
            expected_ledger_id,
            LEDGER_ID_PATTERN,
            label="expected ledger_id",
        )
        canonical = _canonical_existing_file(Path(path), label="ledger database")
        ledger = cls(canonical, expected_pilot_id, expected_ledger_id)
        with _identity_anchor(canonical) as anchor:
            with _read_connection(canonical) as connection:
                snapshot = _verify_connection(
                    connection,
                    expected_pilot_id=expected_pilot_id,
                    expected_ledger_id=expected_ledger_id,
                )
                _ensure_path_identity(canonical, anchor, expected_links=1)
        del snapshot
        return ledger

    def _snapshot(self) -> _IntegritySnapshot:
        canonical = _canonical_existing_file(self.path, label="ledger database")
        with _identity_anchor(canonical) as anchor:
            with _read_connection(canonical) as connection:
                snapshot = _verify_connection(
                    connection,
                    expected_pilot_id=self._expected_pilot_id,
                    expected_ledger_id=self._expected_ledger_id,
                )
                _ensure_path_identity(canonical, anchor, expected_links=1)
        return snapshot

    def verify_integrity(self) -> bool:
        self._snapshot()
        return True

    def committed(self) -> Decimal:
        return money(self._snapshot().committed)

    def remaining(self) -> Decimal:
        return money(CAP_USD - self._snapshot().committed)

    @property
    def head_hash(self) -> str:
        return self._snapshot().head_hash

    def state(self, reservation_id: str) -> str:
        _typed_id(
            reservation_id,
            RESERVATION_ID_PATTERN,
            label="reservation_id",
        )
        snapshot = self._snapshot()
        try:
            return snapshot.states[reservation_id]
        except KeyError:
            raise KeyError("unknown reservation") from None

    def reserve(
        self,
        bundle_id: str,
        execution_id: str,
        amount_usd: Decimal | str,
    ) -> Reservation:
        _bundle_identity(bundle_id)
        _typed_id(execution_id, EXECUTION_ID_PATTERN, label="execution_id")
        amount, amount_text = _reservation_money(amount_usd)
        reservation_id = _new_typed_id("reservation")
        reservation_created_at = _current_utc()
        event_id = _new_typed_id("event")
        event_created_at = _current_utc()
        reservation_effect = _ReservationEffect(
            reservation_id=reservation_id,
            bundle_id=bundle_id,
            execution_id=execution_id,
            amount_usd=amount_text,
            created_at_utc=reservation_created_at,
            migration_id=None,
        )
        canonical = _canonical_existing_file(self.path, label="ledger database")
        with _write_connection(canonical) as connection:
            _begin_immediate(connection)
            try:
                snapshot = _verify_connection(
                    connection,
                    expected_pilot_id=self._expected_pilot_id,
                    expected_ledger_id=self._expected_ledger_id,
                )
                replay = connection.execute(
                    """
                    SELECT 1 FROM reservations
                    WHERE bundle_id = ? OR execution_id = ?
                    LIMIT 1
                    """,
                    (bundle_id, execution_id),
                ).fetchone()
                if replay is not None:
                    raise RuntimeError("execution identity was already reserved")
                if amount > CAP_USD - snapshot.committed:
                    raise RuntimeError("projected cost exceeds remaining budget")
                connection.execute(
                    """
                    INSERT INTO reservations (
                        reservation_id, bundle_id, execution_id, amount_usd,
                        created_at_utc, migration_id
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    reservation_effect.database_row(),
                )
                event_effect = _append_event(
                    connection,
                    snapshot=snapshot,
                    event_id=event_id,
                    reservation_id=reservation_id,
                    from_state=None,
                    to_state="reserved",
                    amount_text=amount_text,
                    created_at_utc=event_created_at,
                    reason_code="reservation_created",
                )
                _commit_with_verified_write_effect(
                    connection,
                    canonical,
                    expected_pilot_id=self._expected_pilot_id,
                    expected_ledger_id=self._expected_ledger_id,
                    event=event_effect,
                    reservation=reservation_effect,
                )
            except sqlite3.IntegrityError:
                _rollback_quietly(connection)
                raise RuntimeError("execution identity was already reserved") from None
            except Exception:
                _rollback_quietly(connection)
                raise
        return Reservation(
            id=reservation_id,
            amount=amount,
            bundle_id=bundle_id,
            execution_id=execution_id,
        )

    def reserve_training(
        self,
        bundle_id: str,
        execution_id: str,
        amount_usd: Decimal | str,
    ) -> Reservation:
        return self.reserve(bundle_id, execution_id, amount_usd)

    def transition(self, reservation_id: str, to_state: str) -> None:
        _typed_id(
            reservation_id,
            RESERVATION_ID_PATTERN,
            label="reservation_id",
        )
        if type(to_state) is not str or to_state not in ALL_STATES:
            raise ValueError("unknown ledger state")
        event_id = _new_typed_id("event")
        event_created_at = _current_utc()
        canonical = _canonical_existing_file(self.path, label="ledger database")
        with _write_connection(canonical) as connection:
            _begin_immediate(connection)
            try:
                snapshot = _verify_connection(
                    connection,
                    expected_pilot_id=self._expected_pilot_id,
                    expected_ledger_id=self._expected_ledger_id,
                )
                try:
                    current = snapshot.states[reservation_id]
                    amount = snapshot.amounts[reservation_id]
                except KeyError:
                    raise KeyError("unknown reservation") from None
                if current == to_state:
                    _rollback_quietly(connection)
                    return
                if (current, to_state) not in NORMAL_TRANSITIONS:
                    raise RuntimeError("illegal ledger state transition")
                event_effect = _append_event(
                    connection,
                    snapshot=snapshot,
                    event_id=event_id,
                    reservation_id=reservation_id,
                    from_state=current,
                    to_state=to_state,
                    amount_text=f"{amount:.4f}",
                    created_at_utc=event_created_at,
                    reason_code="state_transition",
                )
                _commit_with_verified_write_effect(
                    connection,
                    canonical,
                    expected_pilot_id=self._expected_pilot_id,
                    expected_ledger_id=self._expected_ledger_id,
                    event=event_effect,
                )
            except Exception:
                _rollback_quietly(connection)
                raise

    def release_pre_submit(self, reservation_id: str, reason: str) -> None:
        _typed_id(
            reservation_id,
            RESERVATION_ID_PATTERN,
            label="reservation_id",
        )
        if (
            type(reason) is not str
            or not reason
            or reason != reason.strip()
            or len(reason) > 1_024
            or any(ord(character) < 32 or ord(character) == 127 for character in reason)
        ):
            raise ValueError("release reason must be canonical text")
        event_id = _new_typed_id("event")
        event_created_at = _current_utc()
        canonical = _canonical_existing_file(self.path, label="ledger database")
        with _write_connection(canonical) as connection:
            _begin_immediate(connection)
            try:
                snapshot = _verify_connection(
                    connection,
                    expected_pilot_id=self._expected_pilot_id,
                    expected_ledger_id=self._expected_ledger_id,
                )
                try:
                    current = snapshot.states[reservation_id]
                    amount = snapshot.amounts[reservation_id]
                except KeyError:
                    raise KeyError("unknown reservation") from None
                if current == "released":
                    _rollback_quietly(connection)
                    return
                if current not in {"reserved", "uploading"}:
                    raise RuntimeError("cannot release after submit_started")
                event_effect = _append_event(
                    connection,
                    snapshot=snapshot,
                    event_id=event_id,
                    reservation_id=reservation_id,
                    from_state=current,
                    to_state="released",
                    amount_text=f"{amount:.4f}",
                    created_at_utc=event_created_at,
                    reason_code="pre_submit_release",
                )
                _commit_with_verified_write_effect(
                    connection,
                    canonical,
                    expected_pilot_id=self._expected_pilot_id,
                    expected_ledger_id=self._expected_ledger_id,
                    event=event_effect,
                )
            except Exception:
                _rollback_quietly(connection)
                raise

    def reconcile(
        self,
        reservation_id: str,
        outcome: str,
        *,
        evidence_sha256: str,
    ) -> None:
        _typed_id(
            reservation_id,
            RESERVATION_ID_PATTERN,
            label="reservation_id",
        )
        if type(outcome) is not str or outcome not in {"released", "consumed"}:
            raise ValueError("reconciliation outcome must be released or consumed")
        evidence = _sha256(evidence_sha256, label="provider evidence")
        event_id = _new_typed_id("event")
        event_created_at = _current_utc()
        canonical = _canonical_existing_file(self.path, label="ledger database")
        with _write_connection(canonical) as connection:
            _begin_immediate(connection)
            try:
                snapshot = _verify_connection(
                    connection,
                    expected_pilot_id=self._expected_pilot_id,
                    expected_ledger_id=self._expected_ledger_id,
                )
                try:
                    current = snapshot.states[reservation_id]
                    amount = snapshot.amounts[reservation_id]
                except KeyError:
                    raise KeyError("unknown reservation") from None
                if current == outcome:
                    _rollback_quietly(connection)
                    return
                if current not in {"submit_started", "submitted"}:
                    raise RuntimeError("reconciliation is not legal in the current state")
                event_effect = _append_event(
                    connection,
                    snapshot=snapshot,
                    event_id=event_id,
                    reservation_id=reservation_id,
                    from_state=current,
                    to_state=outcome,
                    amount_text=f"{amount:.4f}",
                    created_at_utc=event_created_at,
                    reason_code="provider_reconciliation",
                    evidence_sha256=evidence,
                )
                _commit_with_verified_write_effect(
                    connection,
                    canonical,
                    expected_pilot_id=self._expected_pilot_id,
                    expected_ledger_id=self._expected_ledger_id,
                    event=event_effect,
                )
            except Exception:
                _rollback_quietly(connection)
                raise


def _verify_staged_candidate(
    path: Path,
    *,
    expected_pilot_id: str,
    expected_ledger_id: str,
) -> tuple[str, os.stat_result]:
    canonical = _canonical_existing_file(path, label="staged ledger database")
    with _identity_anchor(canonical) as anchor:
        digest = sha256_file(canonical).sha256
        _ensure_path_identity(canonical, anchor, expected_links=1)
        with _read_connection(canonical) as connection:
            _verify_connection(
                connection,
                expected_pilot_id=expected_pilot_id,
                expected_ledger_id=expected_ledger_id,
                staged=True,
            )
            _ensure_path_identity(canonical, anchor, expected_links=1)
        if sha256_file(canonical).sha256 != digest:
            raise ValueError("staged ledger digest mismatch")
        _ensure_path_identity(canonical, anchor, expected_links=1)
    return digest, anchor


def _verify_linked_staged_destination(
    destination: Path,
    *,
    candidate_anchor: os.stat_result,
    candidate_digest: str,
    expected_pilot_id: str,
    expected_ledger_id: str,
) -> None:
    _reject_destination_sidecars(destination)
    _ensure_path_identity(destination, candidate_anchor, expected_links=2)
    if sha256_file(destination).sha256 != candidate_digest:
        raise ValueError("staged ledger digest mismatch")
    with _read_connection(destination) as connection:
        _verify_connection(
            connection,
            expected_pilot_id=expected_pilot_id,
            expected_ledger_id=expected_ledger_id,
            staged=True,
        )
        _ensure_path_identity(destination, candidate_anchor, expected_links=2)
    if sha256_file(destination).sha256 != candidate_digest:
        raise ValueError("staged ledger digest mismatch")


def _commit_error_left_verified_final_ledger(
    connection: sqlite3.Connection,
    canonical: Path,
    anchor: os.stat_result,
    *,
    expected_pilot_id: str,
    expected_ledger_id: str,
) -> bool:
    try:
        transaction_is_active = connection.in_transaction
    except Exception:
        transaction_is_active = False
    if transaction_is_active:
        return False
    verified_on_connection = False
    try:
        connection.execute("BEGIN")
        _verify_connection(
            connection,
            expected_pilot_id=expected_pilot_id,
            expected_ledger_id=expected_ledger_id,
        )
        _ensure_path_identity(canonical, anchor, expected_links=1)
        verified_on_connection = True
    except Exception:
        pass
    finally:
        try:
            read_transaction_is_active = connection.in_transaction
        except Exception:
            read_transaction_is_active = False
        if read_transaction_is_active:
            try:
                connection.rollback()
            except Exception:
                pass
    if verified_on_connection:
        return True
    try:
        connection.close()
    except Exception:
        pass
    verified_on_reopen = False
    try:
        with _read_connection(canonical) as reopened:
            _verify_connection(
                reopened,
                expected_pilot_id=expected_pilot_id,
                expected_ledger_id=expected_ledger_id,
            )
            _ensure_path_identity(canonical, anchor, expected_links=1)
            verified_on_reopen = True
    except Exception:
        pass
    return verified_on_reopen


def _activate_staged_ledger(
    destination: Path,
    *,
    candidate_digest: str,
    expected_pilot_id: str,
    expected_ledger_id: str,
) -> None:
    _reject_destination_sidecars(destination)
    canonical = _canonical_existing_file(destination, label="ledger database")
    with _identity_anchor(canonical) as anchor:
        if sha256_file(canonical).sha256 != candidate_digest:
            raise ValueError("staged ledger digest mismatch")
        _ensure_path_identity(canonical, anchor, expected_links=1)
        uri = f"{canonical.as_uri()}?mode=rw"
        connection = sqlite3.connect(
            uri,
            uri=True,
            timeout=BUSY_TIMEOUT_MILLISECONDS / 1_000,
            isolation_level=None,
        )
        committed = False
        try:
            _configure_connection(connection)
            connection.execute("PRAGMA synchronous = FULL")
            _begin_immediate(connection)
            _reject_destination_sidecars(canonical)
            if sha256_file(canonical).sha256 != candidate_digest:
                raise ValueError("staged ledger digest mismatch")
            _ensure_path_identity(canonical, anchor, expected_links=1)
            _verify_connection(
                connection,
                expected_pilot_id=expected_pilot_id,
                expected_ledger_id=expected_ledger_id,
                staged=True,
            )
            _ensure_path_identity(canonical, anchor, expected_links=1)
            cursor = connection.execute(
                """
                UPDATE pilot SET schema_version = ?
                WHERE singleton = 1 AND schema_version = ?
                """,
                (LEDGER_SCHEMA_VERSION, STAGED_LEDGER_SCHEMA_VERSION),
            )
            if cursor.rowcount != 1:
                raise ValueError("staged ledger activation mismatch")
            connection.execute(f"PRAGMA user_version = {SQLITE_USER_VERSION}")
            _create_immutability_triggers(connection)
            _verify_connection(
                connection,
                expected_pilot_id=expected_pilot_id,
                expected_ledger_id=expected_ledger_id,
            )
            _ensure_path_identity(canonical, anchor, expected_links=1)
            try:
                connection.commit()
            except Exception:
                committed = _commit_error_left_verified_final_ledger(
                    connection,
                    canonical,
                    anchor,
                    expected_pilot_id=expected_pilot_id,
                    expected_ledger_id=expected_ledger_id,
                )
                if not committed:
                    raise
            else:
                committed = True
        except Exception:
            if connection.in_transaction:
                try:
                    connection.rollback()
                except sqlite3.Error:
                    pass
            try:
                connection.close()
            except Exception:
                pass
            raise
        finally:
            if committed:
                try:
                    connection.close()
                except Exception:
                    pass


def migrate_legacy_ledger(
    source_ledger_path: Path,
    manifest_path: Path,
    destination_path: Path,
) -> PilotLedger:
    source = _canonical_existing_file(Path(source_ledger_path), label="legacy source ledger")
    manifest_file = _canonical_existing_file(Path(manifest_path), label="migration manifest")
    destination = _canonical_destination(Path(destination_path))
    manifest, manifest_digest = _load_manifest(manifest_file)
    _load_and_validate_source(source, manifest)

    temporary: Path | None = None
    temporary_base: Path | None = None
    published = False
    try:
        file_descriptor, name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        os.close(file_descriptor)
        temporary = Path(name)
        temporary_base = temporary
        _initialize_database(temporary, manifest, manifest_digest)
        with temporary.open("r+b") as database_file:
            os.fsync(database_file.fileno())
        try:
            candidate_digest, candidate_anchor = _verify_staged_candidate(
                temporary,
                expected_pilot_id=manifest["pilot_id"],
                expected_ledger_id=manifest["ledger_id"],
            )
        except Exception:
            raise ValueError("ledger publication verification failed") from None

        result = PilotLedger(
            destination,
            manifest["pilot_id"],
            manifest["ledger_id"],
        )
        _preflight_destination_entries(destination)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise ValueError("destination ledger already exists") from None
        except OSError as exc:
            raise ValueError("ledger publication failed") from exc
        published = True
        _verify_linked_staged_destination(
            destination,
            candidate_anchor=candidate_anchor,
            candidate_digest=candidate_digest,
            expected_pilot_id=manifest["pilot_id"],
            expected_ledger_id=manifest["ledger_id"],
        )
        temporary.unlink()
        temporary = None
        temporary_base = None
        _activate_staged_ledger(
            destination,
            candidate_digest=candidate_digest,
            expected_pilot_id=manifest["pilot_id"],
            expected_ledger_id=manifest["ledger_id"],
        )
        return result
    except Exception:
        if published:
            try:
                destination.unlink(missing_ok=True)
            except OSError:
                pass
            raise ValueError("ledger publication failed safely") from None
        raise
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        if temporary_base is not None:
            for suffix in SQLITE_SIDECAR_SUFFIXES:
                try:
                    Path(f"{temporary_base}{suffix}").unlink(missing_ok=True)
                except OSError:
                    pass


__all__ = [
    "CAP_USD_TEXT",
    "COMMITTED_STATES",
    "PilotLedger",
    "Reservation",
    "migrate_legacy_ledger",
]
