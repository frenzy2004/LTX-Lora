# Fal A2V Immutable Execution Bundle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fail-closed, content-addressed Fal LTX 2.3 A2V training and validation workflow that cannot upload or spend against stale, unreviewed, replayed, or over-budget artifacts.

**Architecture:** Normalized A2V groups pass deterministic media checks and an explicit human QA attestation before a deterministic ZIP and root bundle ID are created. A standing authorization policy issues a one-time receipt for that exact bundle and exact verified ledger-chain head, a canonical SQLite ledger atomically reserves the fixed cost with a head compare-and-swap, and paid execution uploads only content-addressed staged bytes after every pre-submit gate succeeds. The private platform root is deployment-approved, absolute, independent of the current working directory, and maps each opaque `pilot_id` to exactly one ledger and each `execution_id` to exactly one run directory. Post-training inference uses separate hash-bound validation bundles and the same cumulative ledger.

**Tech Stack:** Python 3.11+, standard library (`dataclasses`, `hashlib`, `json`, `sqlite3`, `urllib`, `zipfile`, `ctypes`), `ffmpeg`/`ffprobe`, `fal-client`, and `pytest`.

## Global Constraints

- The paid training endpoint is exactly `fal-ai/ltx23-trainer-v2/a2v`.
- The request is exactly rank 32, 1,000 steps, learning rate 0.0002, 89 frames, 24 fps, high resolution, and 9:16.
- `auto_scale_input`, `split_input_into_scenes`, and `debug_dataset` are false; audio normalization and pitch preservation are true.
- Training may not exceed $6.0000; validation is an allocation of at most $1.2500; cumulative committed spend may not exceed $12.0000.
- The extra $2 is not authorized by any environment variable, CLI flag, or standing policy.
- Execute mode accepts no mutable endpoint, dataset ZIP, steps, price, ledger path, or cap override.
- Every receipt-issuance, preflight, paid-training, and paid-validation CLI obtains the approved absolute private platform root only from the deployment setting `LTX_LORA_PRIVATE_ROOT`; there is no current-working-directory fallback and no ledger-path override on those operational surfaces. The one-time migration setup command may name its reviewed source, manifest, and not-yet-created canonical destination explicitly.
- The canonical private layout is `<private-root>/pilots/<pilot_id>/ledger/pilot.sqlite3` and `<private-root>/pilots/<pilot_id>/runs/<execution_id>/`; supplied `run_dir` values must equal the latter path after strict canonical validation.
- No credential lookup, Fal upload, Fal submission, or budget reservation occurs on an offline-preflight failure.
- No automated test may access a real credential or call Fal.
- Source media, private manifests, approvals, ledgers, signed URLs, request IDs, logs, and LoRA weights remain outside Git.
- Public files contain no personal names, copied chats, Drive identifiers, credentials, or private absolute paths.
- Preserve unrelated dirty worktree changes; stage only the files named by each task.

## File Map

- `src/ltx_lora_pilot/artifacts.py`: strict JSON, canonical bytes, hashes, atomic writes, and safe names.
- `src/ltx_lora_pilot/a2v_dataset.py`: normalized media validation.
- `src/ltx_lora_pilot/a2v_quality.py`: human QA, rights, counts, and split isolation.
- `src/ltx_lora_pilot/a2v_bundle.py`: deterministic ZIP and content-addressed manifests.
- `src/ltx_lora_pilot/authorization.py`: standing policy, price evidence, and one-time receipts.
- `src/ltx_lora_pilot/pilot_ledger.py`: canonical SQLite ledger and append-only events.
- `src/ltx_lora_pilot/private_workspace.py`: approved private-root loading, canonical run resolution, and pilot-ledger resolution independent of `cwd`.
- `src/ltx_lora_pilot/provider_validation.py`: strict two-item provider-validation selection bound to holdout structural records and execution configuration.
- `src/ltx_lora_pilot/preflight.py`: shared fail-closed dry-run pipeline.
- `src/ltx_lora_pilot/staging.py`: content-addressed staging and retained file guards.
- `src/ltx_lora_pilot/a2v_execution.py`: exact paid training boundary.
- `src/ltx_lora_pilot/validation_bundle.py`: separately approved validation inference.

---

### Task 1: Strict canonical artifacts

**Files:**
- Create: `src/ltx_lora_pilot/artifacts.py`
- Create: `tests/test_artifacts.py`

**Interfaces:**
- Produces: `FileDigest`, `canonical_json_bytes(value)`, `strict_load_json(path)`, `sha256_file(path)`, `atomic_write_json(path, value)`, and `safe_relative_name(name)`.
- Consumes: standard library only.

- [ ] **Step 1: Write failing canonicalization and parser tests**

```python
from pathlib import Path

import pytest

from ltx_lora_pilot.artifacts import canonical_json_bytes, strict_load_json


def test_canonical_json_is_order_independent() -> None:
    assert canonical_json_bytes({"b": 2, "a": "0.0002"}) == b'{"a":"0.0002","b":2}'


def test_strict_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"cap":"12.0000","cap":"14.0000"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        strict_load_json(path)
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `python -m pytest -q tests/test_artifacts.py`

Expected: collection fails because `ltx_lora_pilot.artifacts` does not exist.

- [ ] **Step 3: Implement canonical JSON, strict parsing, hashing, and atomic writes**

```python
@dataclass(frozen=True)
class FileDigest:
    name: str
    bytes: int
    sha256: str


def canonical_json_bytes(value: Any) -> bytes:
    _reject_unsupported(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def strict_load_json(path: Path) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=_reject_constant)
```

Implement `_reject_unsupported` to reject floats, non-string dictionary keys, non-finite numbers, and unsupported object types. Implement `sha256_file` with 1 MiB chunks. Implement `safe_relative_name` to reject absolute paths, `..`, backslashes, control characters, and non-ASCII names. Implement `atomic_write_json` with a same-directory temporary file, flush, `os.fsync`, and `os.replace`.

- [ ] **Step 4: Add tests for floats, unsafe paths, hash accuracy, and atomic output**

```python
def test_canonical_json_rejects_float() -> None:
    with pytest.raises(TypeError, match="floats are prohibited"):
        canonical_json_bytes({"learning_rate": 0.0002})


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a\\b", "bad\nname"])
def test_safe_relative_name_rejects_unsafe_input(name: str) -> None:
    with pytest.raises(ValueError):
        safe_relative_name(name)
```

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest -q tests/test_artifacts.py`

Expected: all tests pass.

```powershell
git add src/ltx_lora_pilot/artifacts.py tests/test_artifacts.py
git commit -m "feat: add strict canonical artifact utilities"
```

---

### Task 2: Structural A2V validation and human QA

**Files:**
- Modify: `src/ltx_lora_pilot/a2v_dataset.py`
- Create: `src/ltx_lora_pilot/a2v_quality.py`
- Modify: `scripts/validate_a2v_dataset.py`
- Modify: `tests/test_a2v_dataset.py`
- Create: `tests/test_a2v_quality.py`

**Interfaces:**
- Consumes: Task 1 artifact utilities.
- Produces: `A2VSpec`, `validate_a2v_directory(root, spec, trigger_phrase) -> dict`, `load_quality_attestation(path) -> dict`, and `validate_quality_and_splits(attestation, structural_report) -> dict`.

- [ ] **Step 1: Expand structural tests to cover exact normalized media**

```python
def test_a2v_group_rejects_digital_silence(tmp_path: Path) -> None:
    _make_group(tmp_path, silent=True)
    with pytest.raises(ValueError, match="digital silence"):
        validate_a2v_directory(tmp_path, spec=TEST_SPEC)


def test_a2v_group_rejects_audio_stream_in_target(tmp_path: Path) -> None:
    _make_group(tmp_path, target_has_audio=True)
    with pytest.raises(ValueError, match="target must not contain audio"):
        validate_a2v_directory(tmp_path, spec=TEST_SPEC)
```

Also cover wrong video codec, non-PCM WAV, stereo audio, wrong timestamps, symlinks, first-frame mismatch, frame-count mismatch, fps mismatch, unexpected files, unsafe group IDs, and per-file SHA-256 output.

- [ ] **Step 2: Run structural tests and confirm RED**

Run: `python -m pytest -q tests/test_a2v_dataset.py`

Expected: new validation assertions fail.

- [ ] **Step 3: Implement exact stream and silence checks**

Extend ffprobe entries to include `codec_name`, `codec_type`, `channels`, `sample_fmt`, `r_frame_rate`, and stream count. Decode PCM audio with ffmpeg to signed 16-bit samples and reject when every sample is zero. Compare first frames using SHA-256 of decoded RGB bytes.

Return this exact top-level shape:

```python
{
    "schema_version": "a2v-structural-report-v1",
    "status": "valid",
    "spec": {"width": 544, "height": 960, "frames": 89, "fps": 24, "sample_rate": 48000},
    "groups": [{"group_id": "sample_001", "files": [digest_dict, digest_dict, digest_dict, digest_dict]}],
}
```

- [ ] **Step 4: Write failing QA and split tests**

```python
def test_quality_requires_ten_train_and_five_holdout() -> None:
    attestation = make_attestation(train=9, holdout=5)
    with pytest.raises(ValueError, match="at least 10 accepted training groups"):
        validate_quality_and_splits(attestation, make_structural_report(14))


def test_quality_rejects_session_crossing_splits() -> None:
    attestation = make_attestation(train=10, holdout=5, shared_session=True)
    with pytest.raises(ValueError, match="source session crosses"):
        validate_quality_and_splits(attestation, make_structural_report(15))
```

Cover a false rights confirmation, missing required check, accepted false check, duplicate group ID, missing structural group, overlapping source interval, duplicate media digest, fewer than two location-isolated holdouts, and no held-out teeth/inner-mouth coverage.

- [ ] **Step 5: Implement strict attestation and split validation**

Define exact allowed keys for every object. Require all accepted checks except `teeth_or_inner_mouth_visible` to be true. Require 15 unique accepted groups, session isolation, interval isolation, media-hash isolation, two location-isolated holdouts, and one held-out inner-mouth case. Return accepted train IDs, holdout IDs, location coverage, and coverage counts without free-form notes.

- [ ] **Step 6: Update the validation CLI**

Require `--quality-attestation`, write a versioned structural report to `--structural-report`, and print only neutral IDs and counts. Do not echo notes, source IDs, paths, or captions.

- [ ] **Step 7: Run focused tests and commit**

Run: `python -m pytest -q tests/test_a2v_dataset.py tests/test_a2v_quality.py`

Expected: all tests pass.

```powershell
git add src/ltx_lora_pilot/a2v_dataset.py src/ltx_lora_pilot/a2v_quality.py scripts/validate_a2v_dataset.py tests/test_a2v_dataset.py tests/test_a2v_quality.py
git commit -m "feat: enforce structural and human A2V dataset gates"
```

---

### Task 3: Deterministic archive and root bundle

**Files:**
- Create: `src/ltx_lora_pilot/a2v_bundle.py`
- Create: `scripts/build_a2v_bundle.py`
- Create: `tests/test_a2v_bundle.py`

**Interfaces:**
- Consumes: Tasks 1 and 2.
- Produces: `build_training_archive`, `inspect_training_archive`, `build_dataset_manifest`, `build_root_manifest`, and `compute_bundle_id`.

- [ ] **Step 1: Write failing determinism and archive-safety tests**

```python
def test_archive_is_byte_identical_across_two_builds(tmp_path: Path) -> None:
    first = build_training_archive(FIXTURE_GROUPS, tmp_path / "one.zip")
    second = build_training_archive(FIXTURE_GROUPS, tmp_path / "two.zip")
    assert first.sha256 == second.sha256
    assert (tmp_path / "one.zip").read_bytes() == (tmp_path / "two.zip").read_bytes()


def test_bundle_id_excludes_self_hash() -> None:
    with pytest.raises(ValueError, match="must not be serialized"):
        compute_bundle_id({"bundle_id": "0" * 64})
```

Also test traversal, absolute paths, duplicate and case-colliding names, symlink attributes, encryption, non-`ZIP_STORED`, member-count limit, uncompressed-size limit, compression-ratio limit, unexpected members, changed bytes, and holdout exclusion.

- [ ] **Step 2: Run tests and confirm RED**

Run: `python -m pytest -q tests/test_a2v_bundle.py`

Expected: import fails because `a2v_bundle` does not exist.

- [ ] **Step 3: Implement deterministic ZIP and safe inspection**

Use `ZipInfo` timestamp `(1980, 1, 1, 0, 0, 0)`, fixed `external_attr`, empty `extra`/`comment`, lexical order, and `ZIP_STORED`. Write to a temporary file, fsync, replace, reopen, and inspect.

```python
def compute_bundle_id(root_manifest: dict[str, Any]) -> str:
    if "bundle_id" in root_manifest:
        raise ValueError("bundle_id must not be serialized into its digest domain")
    return hashlib.sha256(canonical_json_bytes(root_manifest)).hexdigest()
```

- [ ] **Step 4: Implement dataset and root manifests**

Bind every train and holdout group, every file hash/size, structural and attestation hashes, counts, spec, archive hash, policy, price evidence, plan, execution config, builder/validator versions, and repository commit. Exclude approval, preflight, ledger, logs, provider state, and outputs from the root digest.

- [ ] **Step 5: Implement the build command and commit**

The command accepts only a private run directory and writes under `bundle/`. Run `python -m pytest -q tests/test_a2v_bundle.py`; expect all tests to pass.

```powershell
git add src/ltx_lora_pilot/a2v_bundle.py scripts/build_a2v_bundle.py tests/test_a2v_bundle.py
git commit -m "feat: build content-addressed A2V bundles"
```

---

### Task 4: Standing authorization, price evidence, and receipt issuance

**Files:**
- Create: `src/ltx_lora_pilot/authorization.py`
- Create: `scripts/record_standing_authorization.py`
- Create: `scripts/capture_fal_price.py`
- Create: `scripts/issue_a2v_approval.py`
- Create: `tests/test_authorization.py`

**Interfaces:**
- Consumes: Task 1 canonical utilities and Task 3 bundle ID.
- Produces: `StandingAuthorization`, `PriceEvidence`, `ExecutionReceipt`, `capture_price_evidence`, `issue_execution_receipt`, and `verify_execution_receipt`.

- [ ] **Step 1: Write failing strict-policy tests**

```python
def test_policy_rejects_extra_two_dollar_cap() -> None:
    policy = valid_policy(cumulative_cap_usd="14.0000")
    with pytest.raises(ValueError, match="cumulative cap must be 12.0000"):
        StandingAuthorization.from_dict(policy)


def test_receipt_for_bundle_a_cannot_approve_bundle_b() -> None:
    receipt = issue_execution_receipt(valid_policy(), bundle_a())
    with pytest.raises(ValueError, match="bundle mismatch"):
        verify_execution_receipt(receipt, valid_policy(), bundle_b())
```

Cover endpoint, step count, training ceiling, validation allocation, unknown fields, expired policy, expired bundle, replay ID, wrong policy hash, and malformed source hash.

- [ ] **Step 2: Write failing price-evidence tests with an injected fetcher**

```python
def test_price_capture_requires_official_formula() -> None:
    fetch = lambda _url: b"The cost is 0.007 * steps."
    with pytest.raises(ValueError, match="unexpected A2V rate"):
        capture_price_evidence(fetch=fetch, now=FIXED_TIME)
```

Cover official HTTPS host allowlist, `$0.006 * steps`, `$6.00` for 1,000 steps, response SHA-256, 24-hour expiry, fetch failure, and zero credential access.

- [ ] **Step 3: Implement exact authorization dataclasses**

```python
@dataclass(frozen=True)
class StandingAuthorization:
    policy_id: str
    source_sha256: str
    endpoint: str
    executions: int
    steps: int
    training_max_usd: str
    validation_allocation_usd: str
    cumulative_cap_usd: str
    expires_at_utc: str
```

The recorder receives `--source-file`, hashes it, and writes only the hash plus fixed policy fields. It never copies source contents. The price command uses unauthenticated `urllib.request` and stores only URL, rate, response hash, retrieval time, and expiry.

- [ ] **Step 4: Implement the separate receipt issuer**

Require explicit `--bundle-id` with 64 lowercase hex characters. Recompute the bundle; verify policy hash, price evidence, endpoint, steps, costs, cap, and execution ID; then write the receipt. Do not import Fal or the ledger module.

- [ ] **Step 5: Prove the issuer has no paid capabilities**

Monkeypatch environment and network access. Assert issuer tests never read `FAL_KEY`, instantiate Fal, reserve money, upload, or submit.

- [ ] **Step 6: Run tests and commit**

Run: `python -m pytest -q tests/test_authorization.py`

Expected: all tests pass.

```powershell
git add src/ltx_lora_pilot/authorization.py scripts/record_standing_authorization.py scripts/capture_fal_price.py scripts/issue_a2v_approval.py tests/test_authorization.py
git commit -m "feat: bind A2V execution to standing authorization"
```

Task 4's receipt is an intermediate no-spend authorization artifact. Task 5B must upgrade it to strict `a2v-execution-approval-v2` and bind the verified ledger head before Task 6 or any paid boundary is implemented.

---

### Task 5: Canonical SQLite budget ledger

**Files:**
- Create: `src/ltx_lora_pilot/pilot_ledger.py`
- Create: `scripts/migrate_budget_ledger.py`
- Create: `tests/test_pilot_ledger.py`
- Modify: `src/ltx_lora_pilot/budget.py`
- Modify: `tests/test_budget.py`

**Interfaces:**
- Consumes: money helpers from `budget.py` and hashing from Task 1.
- Produces: `PilotLedger`, `Reservation`, `migrate_legacy_ledger`, `reserve`, `transition`, `release_pre_submit`, `reconcile`, `remaining`, and `verify_integrity`.

- [ ] **Step 1: Write failing migration and integrity tests**

```python
def test_migration_reproduces_exact_conservative_total(tmp_path: Path) -> None:
    ledger = migrate_fixture(tmp_path, amounts=["1.2000", "0.1099", "0.1099", "0.3272", "0.3272", "1.4667"])
    assert ledger.committed() == Decimal("3.5409")
    assert ledger.remaining() == Decimal("8.4591")


def test_execute_refuses_fresh_database(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="migration manifest is required"):
        PilotLedger.open_existing(tmp_path / "budget.sqlite3", EXPECTED_ID)
```

Cover source-ledger hash mismatch, omitted entry, changed amount/state, wrong cap, wrong IDs, broken event chain, failed `PRAGMA integrity_check`, and missing database.

- [ ] **Step 2: Write failing concurrency and replay tests**

Spawn two processes against a ledger with $6.0000 remaining. Both attempt a $6.0000 reservation; assert exactly one succeeds. Assert a second reservation for the same `(bundle_id, execution_id)` fails even after completion.

- [ ] **Step 3: Implement schema and transactional event chain**

Use tables `pilot`, `migration_entries`, `reservations`, and `events`. Events are append-only and contain `event_id`, `reservation_id`, `from_state`, `to_state`, `amount_usd`, `created_at_utc`, `previous_hash`, and `event_hash`. Use `BEGIN IMMEDIATE`, a 5-second busy timeout, foreign keys, and a unique `(bundle_id, execution_id)` constraint.

Committed states are `reserved`, `uploading`, `submit_started`, `submitted`, and `consumed`. `released` does not count. Derive totals from event history; do not store a mutable total.

- [ ] **Step 4: Implement exact pre-submit release semantics**

```python
def release_pre_submit(self, reservation_id: str, reason: str) -> None:
    current = self.state(reservation_id)
    if current not in {"reserved", "uploading"}:
        raise RuntimeError("cannot release after submit_started")
    self._append_transition(reservation_id, current, "released", reason)
```

Require `submit_started` to commit before the network call. An ambiguous submit remains committed until `reconcile` appends a provider-evidence-backed release or consumed event.

- [ ] **Step 5: Keep legacy estimators but isolate legacy execution**

Leave `estimate_training_cost` and `estimate_inference_cost` stable. Mark JSON `BudgetLedger` legacy and ensure the new A2V runner imports only `PilotLedger`.

- [ ] **Step 6: Run tests and commit**

Run: `python -m pytest -q tests/test_budget.py tests/test_pilot_ledger.py`

Expected: all tests pass, including the process race.

```powershell
git add src/ltx_lora_pilot/pilot_ledger.py src/ltx_lora_pilot/budget.py scripts/migrate_budget_ledger.py tests/test_pilot_ledger.py tests/test_budget.py
git commit -m "feat: add transactional pilot budget ledger"
```

---

### Task 5A: Atomic ledger preflight snapshot and canonical private resolver

**Files:**
- Create: `src/ltx_lora_pilot/private_workspace.py`
- Create: `tests/test_private_workspace.py`
- Modify: `src/ltx_lora_pilot/pilot_ledger.py`
- Modify: `tests/test_pilot_ledger.py`

**Interfaces:**
- Consumes: Task 5's verified existing ledger and opaque `pilot_id`, `ledger_id`, `bundle_id`, and `execution_id` values.
- Produces: `LedgerPreflightSnapshot`, `PilotLedger.preflight_snapshot(bundle_id, execution_id)`, head-guarded `reserve_training(..., expected_head_sha256=...)`, `approved_private_root_from_environment()`, `resolve_pilot_ledger(private_root, pilot_id)`, and `require_canonical_run_dir(private_root, pilot_id, execution_id, run_dir)`.

- [ ] **Step 1: Write RED tests for one atomic read snapshot**

```python
def test_preflight_snapshot_is_one_verified_read_transaction(ledger, monkeypatch) -> None:
    expected_head = ledger.head_hash
    calls = install_connection_counter(monkeypatch)
    snapshot = ledger.preflight_snapshot(BUNDLE_ID, EXECUTION_ID)
    assert calls.read_transactions == 1
    assert snapshot == LedgerPreflightSnapshot(
        pilot_id=PILOT_ID,
        ledger_id=LEDGER_ID,
        bundle_id=BUNDLE_ID,
        execution_id=EXECUTION_ID,
        head_sha256=expected_head,
        committed_usd="3.5409",
        remaining_usd="8.4591",
        replay_detected=False,
    )
```

Cover a corrupt chain, wrong pilot/ledger identity, a bundle replay, an execution replay, a concurrent writer between attempted component reads, path replacement, and any database/sidecar byte or metadata mutation. The snapshot must run full schema, SQLite, migration, event-chain, identity, balance, and replay verification inside one read transaction and one retained path-identity anchor; it must not call the existing separate `committed()`, `remaining()`, `head_hash`, or `state()` accessors.

- [ ] **Step 2: Write RED head compare-and-swap reservation tests**

```python
def test_training_reservation_rejects_changed_ledger_head(ledger) -> None:
    snapshot = ledger.preflight_snapshot(BUNDLE_ID, EXECUTION_ID)
    append_unrelated_authorized_reservation(ledger)
    with pytest.raises(RuntimeError, match="ledger head changed after approval"):
        ledger.reserve_training(
            BUNDLE_ID,
            EXECUTION_ID,
            Decimal("6.0000"),
            expected_head_sha256=snapshot.head_sha256,
        )
```

The expected head comparison occurs after `BEGIN IMMEDIATE` and full integrity verification but before inserting either reservation or event. A mismatch rolls back with no rows, sidecars, or acknowledgement ambiguity. This is the atomic boundary that makes a receipt's ledger-head binding meaningful.

- [ ] **Step 3: Write RED private-layout resolver tests**

```python
def test_ledger_resolution_is_absolute_and_independent_of_cwd(private_root, monkeypatch) -> None:
    monkeypatch.chdir(private_root.parent)
    assert resolve_pilot_ledger(private_root, PILOT_ID) == (
        private_root / "pilots" / PILOT_ID / "ledger" / "pilot.sqlite3"
    ).resolve(strict=True)
```

Require `LTX_LORA_PRIVATE_ROOT` to be present and absolute with no NUL or leading/trailing whitespace and no relative fallback. Require exactly `<root>/pilots/<pilot_id>/ledger/pilot.sqlite3` and `<root>/pilots/<pilot_id>/runs/<execution_id>`. Reject unknown ID syntax, `..`, links/reparse points, case aliases, a mismatched supplied `run_dir`, and any fallback to `Path.cwd()`. The resolver returns paths only; Task 6 performs the complete ancestor, permission, hardlink, and repository-exclusion safety audit.

- [ ] **Step 4: Implement the immutable snapshot, reservation head guard, and resolver**

```python
@dataclass(frozen=True)
class LedgerPreflightSnapshot:
    pilot_id: str
    ledger_id: str
    bundle_id: str
    execution_id: str
    head_sha256: str
    committed_usd: str
    remaining_usd: str
    replay_detected: bool
```

Render money as canonical four-place strings. `replay_detected` is true if either the candidate bundle ID or candidate execution ID already appears in `reservations`; do not expose an existing reservation ID or state. Add the required `expected_head_sha256` keyword-only parameter to `reserve_training` and propagate it to the single `BEGIN IMMEDIATE` reservation transaction.

- [ ] **Step 5: Verify exact scope and obtain independent review**

Run:

```powershell
python -m pytest -q tests/test_private_workspace.py tests/test_pilot_ledger.py tests/test_budget.py
python -m pytest -q
python -m py_compile src/ltx_lora_pilot/private_workspace.py src/ltx_lora_pilot/pilot_ledger.py
git diff --check
```

Expected: all tests pass, no provider import or credential lookup occurs, and the diff contains only the four files listed above. Obtain independent Spec PASS and Code Approved verdicts before committing.

```powershell
git add src/ltx_lora_pilot/private_workspace.py src/ltx_lora_pilot/pilot_ledger.py tests/test_private_workspace.py tests/test_pilot_ledger.py
git commit -m "feat: add atomic ledger preflight snapshot"
```

---

### Task 5B: Bind execution receipts to the verified ledger head

**Files:**
- Modify: `src/ltx_lora_pilot/authorization.py`
- Modify: `scripts/issue_a2v_approval.py`
- Modify: `tests/test_authorization.py`

**Interfaces:**
- Consumes: Task 5A's `LedgerPreflightSnapshot` reader and canonical private resolver.
- Produces: strict receipt schema `a2v-execution-approval-v2` with mandatory `ledger_head_sha256`, and `issue_execution_receipt(..., read_ledger_snapshot=...)` that issues only from a fresh non-replayed snapshot matching the bundle's pilot, ledger, bundle, and execution identities.

- [ ] **Step 1: Write RED schema and mismatch tests**

```python
def test_receipt_binds_exact_verified_ledger_head(ready_bundle, snapshot_reader) -> None:
    receipt = issue_execution_receipt(
        POLICY,
        ready_bundle,
        expected_bundle_id=BUNDLE_ID,
        read_ledger_snapshot=snapshot_reader,
    )
    assert receipt.schema_version == "a2v-execution-approval-v2"
    assert receipt.ledger_head_sha256 == snapshot_reader.result.head_sha256
```

Reject v1 receipts, a missing/extra ledger-head field, malformed or uppercase hashes, identity mismatch in any of the four snapshot identity fields, replay detected at issuance, and a receipt whose bound head differs from the later paid-preflight snapshot. A ledger change after issuance makes the receipt unusable; it is never silently rebound.

- [ ] **Step 2: Prove the issuer remains offline and non-mutating**

The CLI loads `LTX_LORA_PRIVATE_ROOT`, resolves the ledger from the bundle's `pilot_id`, opens the exact `ledger_id`, and supplies one Task 5A snapshot reader. Spy on SQLite writes, reservations, environment credential reads, Fal imports, network access, uploads, submissions, and polling; all must remain zero. Only the create-new private receipt write is allowed.

- [ ] **Step 3: Implement the v2 receipt and snapshot-bound issuer**

Add `ledger_head_sha256` to the dataclass, exact field set, parser, serializer, issuer, and verifier. The snapshot reader signature is:

```python
LedgerSnapshotReader = Callable[
    [str, str, str, str],
    LedgerPreflightSnapshot,
]
# arguments: pilot_id, ledger_id, bundle_id, execution_id
```

The issuer compares the returned snapshot identities to `_BundleFacts`, rejects `replay_detected`, and copies only `head_sha256` into the receipt. `verify_execution_receipt` validates the receipt's bound artifacts and schema but Task 6 is responsible for comparing the receipt head to the current atomic ledger snapshot. No mutable ledger path is accepted by the library or CLI.

- [ ] **Step 4: Run focused/full tests and obtain independent review**

Run:

```powershell
python -m pytest -q tests/test_authorization.py tests/test_private_workspace.py tests/test_pilot_ledger.py
python -m pytest -q
python -m py_compile src/ltx_lora_pilot/authorization.py scripts/issue_a2v_approval.py
git diff --check
```

Expected: all tests pass; exact-scope and privacy scans show only the three listed files. Obtain independent Spec PASS and Code Approved verdicts before committing.

```powershell
git add src/ltx_lora_pilot/authorization.py scripts/issue_a2v_approval.py tests/test_authorization.py
git commit -m "feat: bind approval to ledger chain head"
```

---

### Task 5C: Strict provider-validation selection schema

**Files:**
- Create: `src/ltx_lora_pilot/provider_validation.py`
- Create: `tests/test_provider_validation.py`
- Modify: `src/ltx_lora_pilot/authorization.py`
- Modify: `scripts/build_a2v_bundle.py`
- Modify: `tests/test_authorization.py`
- Modify: `tests/test_a2v_bundle.py`

**Interfaces:**
- Consumes: the canonical structural report, validated quality/split result, current candidate bytes, and canonical execution configuration.
- Produces: schema `a2v-provider-validation-selection-v1`, `build_provider_validation_selection(...)`, and `validate_provider_validation_selection(selection, structural_report, quality_summary, execution_config, candidate_dir)`.

- [ ] **Step 1: Write RED exact-schema and holdout-binding tests**

The exact canonical object is:

```python
{
    "schema_version": "a2v-provider-validation-selection-v1",
    "canonical_json_version": 1,
    "structural_report_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
    "execution_config_sha256": "2222222222222222222222222222222222222222222222222222222222222222",
    "items": [
        {
            "group_id": "grp_0000000000004000800000000000000b",
            "prompt": "A medium talking-head shot with natural speech and steady eye contact.",
            "image": {"name": "grp_0000000000004000800000000000000b_start.png", "bytes": 1024, "sha256": "3333333333333333333333333333333333333333333333333333333333333333"},
            "audio": {"name": "grp_0000000000004000800000000000000b_audio.wav", "bytes": 2048, "sha256": "4444444444444444444444444444444444444444444444444444444444444444"},
        },
        {
            "group_id": "grp_0000000000004000800000000000000c",
            "prompt": "A close talking-head shot with natural speech and subtle facial motion.",
            "image": {"name": "grp_0000000000004000800000000000000c_start.png", "bytes": 1025, "sha256": "5555555555555555555555555555555555555555555555555555555555555555"},
            "audio": {"name": "grp_0000000000004000800000000000000c_audio.wav", "bytes": 2049, "sha256": "6666666666666666666666666666666666666666666666666666666666666666"},
        },
    ],
}
```

Require exactly two distinct accepted holdout groups in canonical group-ID order. A prompt is a stripped, nonempty, NFC-normalized string of at most 1,024 UTF-8 bytes with no Unicode control/format/surrogate/private-use character. Every local image/audio name, size, and digest must match both the corresponding structural-report record and freshly hashed candidate bytes. Reject train/rejected/missing groups, duplicate media, unknown keys, aliases, URLs, path separators, control characters, noncanonical prompts, stale structural/config digests, or any item-count other than two.

- [ ] **Step 2: Correct the execution-config/provider schema split**

Write RED tests requiring execution config schema `a2v-execution-config-v2`. Remove per-item `frames`, `fps`, `resolution`, and `aspect_ratio` fields and remove the embedded validation-item list. Add exact trainer-level fields `validation_number_of_frames: 89`, `validation_frame_rate: 24`, `validation_resolution: "high"`, and `validation_aspect_ratio: "9:16"`. The immutable selection owns the two prompts and local digest records. At paid runtime only, Task 7 maps each item to Fal's exact provider item `{prompt, image_url, audio_url}` after uploading the guarded local bytes; runtime URLs are never serialized into the selection, execution config, root bundle, preflight report, or logs.

- [ ] **Step 3: Implement strict build-time validation**

Use strict JSON and `canonical_json_bytes` for both referenced object digests. `scripts/build_a2v_bundle.py` must validate the selection against the current structural report, current quality summary, execution config, and candidate files before hashing it into the root manifest. It must not merely accept any canonical JSON file at the expected filename.

- [ ] **Step 4: Run focused/full tests and obtain independent review**

Run:

```powershell
python -m pytest -q tests/test_provider_validation.py tests/test_authorization.py tests/test_a2v_bundle.py
python -m pytest -q
python -m py_compile src/ltx_lora_pilot/provider_validation.py src/ltx_lora_pilot/authorization.py scripts/build_a2v_bundle.py
git diff --check
```

Expected: all tests pass with no provider/network/credential access; exact-scope and privacy scans contain only the six files listed above. Obtain independent Spec PASS and Code Approved verdicts before committing.

```powershell
git add src/ltx_lora_pilot/provider_validation.py src/ltx_lora_pilot/authorization.py scripts/build_a2v_bundle.py tests/test_provider_validation.py tests/test_authorization.py tests/test_a2v_bundle.py
git commit -m "feat: bind provider validation holdouts"
```

---

### Task 6: Shared offline preflight

**Files:**
- Create: `src/ltx_lora_pilot/preflight.py`
- Create: `scripts/preflight_a2v.py`
- Create: `tests/test_preflight.py`

**Interfaces:**
- Consumes: Tasks 1–5C, the deployment-approved private root, and a canonical `run_dir`.
- Produces: `PreflightStatus`, `run_preflight(run_dir, confirmed_bundle_id, *, require_receipt, approved_private_root, clock)`, `PreflightStatus.require_ready()`, and sanitized `control/preflight-report.json`.

- [ ] **Step 1: Write a table-driven RED suite for every gate**

```python
@pytest.mark.parametrize(
    "mutation,expected_gate",
    [
        ("archive_byte", "root_artifact_hashes"),
        ("unsafe_archive_member", "archive_inspection"),
        ("validation_asset", "provider_validation_selection"),
        ("request_steps", "request_allowlist"),
        ("stale_price", "price_freshness"),
        ("wrong_receipt", "receipt"),
        ("wrong_ledger", "ledger_snapshot"),
    ],
)
def test_preflight_fails_closed(mutation: str, expected_gate: str, ready_run: Path) -> None:
    mutate(ready_run, mutation)
    report = run_preflight(
        ready_run,
        bundle_id(ready_run),
        require_receipt=True,
        approved_private_root=PRIVATE_ROOT,
        clock=FIXED_CLOCK,
    )
    assert report.status == "failed"
    assert report.failed_gate == expected_gate
```

Cover every ordered gate, not only the examples above. Use spies asserting zero calls to budget reservation, SQLite writes, secret resolution, upload, submit, and poll for each failure. Assert no failure calls or imports a provider module.

- [ ] **Step 2: Run tests and confirm RED**

Run: `python -m pytest -q tests/test_preflight.py`

Expected: import fails because `preflight` does not exist.

- [ ] **Step 3: Implement canonical private-root safety before artifact reads**

`approved_private_root` must be the strict absolute value loaded by the CLI from `LTX_LORA_PRIVATE_ROOT`; the library receives it explicitly for testability and never consults `cwd`. Require `run_dir` to equal `<root>/pilots/<pilot_id>/runs/<execution_id>` after the IDs are parsed, while the initial path-shape gate accepts only opaque ID-shaped path components.

Inspect every lexical path component from the filesystem anchor through the approved root, then through `run_dir`, and every root-bound input with `lstat`/Windows reparse metadata before canonical resolution can hide an alias. Reject symlinks, junctions, mount/reparse aliases, alternate-data-stream syntax, case aliases, special files, regular-file hardlink counts other than one, and any identity change during inspection. On POSIX require owner-only approved-root/run directories (`0700`) and files no broader than `0600`. On Windows fail closed if the approved-root/run DACL is unavailable, null, or grants read/write/delete/write-DAC/write-owner to Everyone, Authenticated Users, or Builtin Users; use `GetNamedSecurityInfoW` plus effective-rights checks. The approved root and `run_dir` must be outside the canonical repository root, and the repository must not be nested inside the approved private root. Tests inject the permission/reparse inspector and cover denial or inspection failure on every ancestor.

- [ ] **Step 4: Implement the exact safe gate order**

The gate-name allowlist and order are immutable:

```python
GATE_ORDER = (
    "private_root",
    "canonical_artifacts",
    "bundle_id",
    "root_artifact_hashes",
    "archive_inspection",
    "archive_structural_validation",
    "candidate_structural_rerun",
    "quality_attestation",
    "split_and_manifest",
    "provider_validation_selection",
    "request_allowlist",
    "price_freshness",
    "standing_policy",
    "receipt",
    "ledger_snapshot",
    "final_recheck",
)
```

Perform them exactly as follows:

1. Validate private-root types/layout/ancestors before reading artifact contents.
2. Strict-load canonical JSON and reject unknown schema versions, keys, aliases, floats, duplicate keys, noncanonical timestamps/money, and noncanonical bytes.
3. Recompute the root bundle ID and compare it to the confirmed ID.
4. Pin identity/size/mtime/link count and freshly hash every root-bound artifact and selected holdout file; compare all root hash/size records.
5. Inspect the already-hashed ZIP's exact canonical layout, member allowlist, sizes, attributes, count, and names before extracting one byte.
6. Create an owner-only ephemeral directory under `run_dir/.preflight-tmp/`; stream each inspected member to an `xb` file while rechecking declared size and SHA-256. Never call `ZipFile.extract` or `extractall`, never join an unvalidated member name, and delete the ephemeral tree after structural validation.
7. Rerun full structural validation against current `run_dir/candidates` bytes and require canonical equality to the bound structural report.
8. Validate the current quality attestation.
9. Validate split isolation, dataset manifest, archive membership, and root holdout records.
10. Validate `a2v-provider-validation-selection-v1` against the structural report, accepted holdouts, current candidate bytes, and execution-config digest.
11. Validate exact `a2v-execution-config-v2`, fixed trainer fields, and the exact Fal A2V endpoint without constructing runtime URLs.
12. Validate current official rate and freshness.
13. Validate the standing policy and all bound costs/expiry.
14. When required, validate the strict v2 receipt and its artifact identities; otherwise record the receipt gate as passed-without-receipt only for policy issuance.
15. Resolve `<root>/pilots/<pilot_id>/ledger/pilot.sqlite3` with no caller override and obtain exactly one Task 5A atomic snapshot; require matching pilot/ledger identity, no replay, sufficient remaining budget, and, when a receipt is required, an exact receipt-head match.
16. With a fresh clock read, recheck every expiry. Re-stat and re-hash all pinned root-bound artifacts, the receipt when required, and selected holdout bytes; require unchanged identity/metadata/content and reject any newly created ledger sidecar or private-root alias.

- [ ] **Step 5: Define the exact status/report contract**

```python
@dataclass(frozen=True)
class PreflightStatus:
    schema_version: str
    status: str
    failed_gate: str | None
    receipt_required: bool
    bundle_id: str
    execution_id: str | None
    training_groups: int | None
    holdout_groups: int | None
    provider_validation_items: int | None
    committed_usd: str | None
    remaining_usd: str | None
    training_reservation_usd: str
    remaining_after_reservation_usd: str | None
    passed_gates: tuple[str, ...]
    pilot_id: str | None = field(repr=False)
    ledger_id: str | None = field(repr=False)
    ledger_head_sha256: str | None = field(repr=False)
```

`schema_version` is exactly `a2v-preflight-report-v1`. `status` is exactly `failed`, `ready_for_policy_issuance`, or `ready_for_paid_execution`. `failed_gate` is null on success and otherwise the first failed member of `GATE_ORDER`. Money is a four-place string or null before the ledger gate. `training_reservation_usd` is always `6.0000`. The internal pilot, ledger, and head fields are never included by `to_public_dict()`.

The exact serialized object has keys `schema_version`, `status`, `failed_gate`, `receipt_required`, `bundle_id`, `execution_id`, `counts`, `budget`, and `passed_gates`. `counts` has exactly `training_groups`, `holdout_groups`, and `provider_validation_items`; `budget` has exactly `committed_usd`, `remaining_usd`, `training_reservation_usd`, and `remaining_after_reservation_usd`. No error text, path, filename, caption, prompt, note, source/session/location ID, URL, non-public hash, credential, provider ID, or ledger identity is serialized.

`require_ready()` returns `self` only when status is `ready_for_paid_execution`, receipt verification succeeded, `failed_gate is None`, `ledger_head_sha256` is present, and all gates passed in exact order. It raises neutral `PreflightNotReady("preflight is not ready for paid execution")` for `failed` and `ready_for_policy_issuance`; a policy-only result can never cross the paid boundary.

- [ ] **Step 6: Implement CLI states and constrained report writing**

The CLI accepts only `--run-dir`, `--confirm-bundle-id`, and optional `--require-receipt`. It loads the mandatory absolute deployment root from `LTX_LORA_PRIVATE_ROOT`; it has no root, ledger, execute, upload, submit, endpoint, price, step, or cap override. Without a receipt, success prints `ready_for_policy_issuance`. With `--require-receipt`, it prints `ready_for_paid_execution`. Any failed gate exits nonzero.

“Offline and read-only” means no mutation of source artifacts, ledger/database/sidecars, credentials, provider state, or budget. Owner-only ephemeral extraction under the private run and one create/replace atomic sanitized `control/preflight-report.json` are explicitly allowed. A private-root failure is printed as sanitized JSON but must not write inside an untrusted root. The command cannot import `fal_client` or any provider module.

- [ ] **Step 7: Run tests, exact-scope checks, and independent review**

Run: `python -m pytest -q tests/test_preflight.py`

Expected: all tests pass.

Also run the full suite, `py_compile`, `git diff --check`, an exact three-file scope check, and the repository privacy scan. Obtain independent Spec PASS and Code Approved verdicts before committing.

```powershell
git add src/ltx_lora_pilot/preflight.py scripts/preflight_a2v.py tests/test_preflight.py
git commit -m "feat: add fail-closed A2V preflight"
```

---

### Task 7: Content-addressed staging and paid training boundary

**Files:**
- Create: `src/ltx_lora_pilot/staging.py`
- Create: `src/ltx_lora_pilot/a2v_execution.py`
- Modify: `src/ltx_lora_pilot/fal_api.py`
- Replace: `scripts/train_a2v.py`
- Create: `tests/test_staging.py`
- Create: `tests/test_a2v_execution.py`
- Modify: `tests/test_train_a2v_script.py`
- Modify: `tests/test_training_execution.py`

**Interfaces:**
- Consumes: `run_preflight`, the canonical private resolver, Task 5A's ledger snapshot/head guard, and exact bundle/request/selection artifacts.
- Produces: `StagedArtifactGuard`, `execute_training_bundle`, and the safe A2V command.

- [ ] **Step 1: Write RED staging mutation tests**

Test create-new content-addressed staging, private permissions, exact hash/size/file identity, source rename after staging, staged replacement, write attempt while guarded, and post-upload verification.

```python
def test_uploader_receives_guarded_staged_path(ready_run: Path, private_root: Path) -> None:
    with stage_bundle(ready_run, approved_private_root=private_root) as staged:
        assert staged.training_zip.parent.name == staged.bundle_id
        assert len(staged.validation_pairs) == 2
        assert staged.verify_unchanged()
```

- [ ] **Step 2: Implement retained platform guards**

On Windows use `CreateFileW(GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING)` to deny cooperative write/delete sharing. On POSIX use mode-0700 staging, mode-0400 files, retained descriptors, and shared `flock`. Record file identity, size, and SHA-256. Stage and guard the training ZIP plus exactly the two selection-bound local image/audio pairs; the uploader receives only guarded staged paths.

- [ ] **Step 3: Write RED execution-boundary tests**

Cover no credential access before preflight/reservation, receipt-head/ledger-head race rejection, upload failure release, mutation-after-upload release, durable `submit_started` before submit, ambiguous submit remaining committed, immediate private request-ID persistence, exact endpoint/payload, no retry, no URL persistence, and redacted logs.

- [ ] **Step 4: Implement exact execution**

```python
def execute_training_bundle(run_dir, confirmed_bundle_id, *, approved_private_root, resolve_key, upload_fn, submit_fn):
    report = run_preflight(
        run_dir,
        confirmed_bundle_id,
        require_receipt=True,
        approved_private_root=approved_private_root,
        clock=system_utc_clock,
    )
    report.require_ready()
    ledger_path = resolve_pilot_ledger(approved_private_root, report.pilot_id)
    ledger = PilotLedger.open_existing(
        ledger_path,
        report.pilot_id,
        expected_ledger_id=report.ledger_id,
    )
    reservation = ledger.reserve_training(
        report.bundle_id,
        report.execution_id,
        Decimal("6.0000"),
        expected_head_sha256=report.ledger_head_sha256,
    )
    with stage_bundle(run_dir, approved_private_root=approved_private_root) as staged:
        resolve_key()
        ledger.transition(reservation.id, "uploading")
        urls = upload_staged_assets(staged, upload_fn)
        staged.require_unchanged()
        ledger.transition(reservation.id, "submit_started")
        return submit_and_persist(report, urls, reservation, ledger, submit_fn)
```

Upload or mutation failures append `released` only when no `submit_started` event exists. Submit exceptions remain committed and are never retried automatically. The provider payload uses exact top-level trainer validation fields from `a2v-execution-config-v2` and exactly two runtime validation items `{prompt, image_url, audio_url}` derived in memory from the verified selection and guarded uploads. Signed/runtime URLs are never written to an immutable artifact, preflight report, ordinary log, or public file.

- [ ] **Step 5: Replace the A2V command surface**

Accept only `--run-dir`, `--confirm-bundle-id`, and `--execute`. Load the approved absolute private root from `LTX_LORA_PRIVATE_ROOT`. Default is shared dry-run. Remove dataset ZIP, plan marker, steps, trigger, validation JSON, private-root override, budget, ledger path, and cost overrides.

- [ ] **Step 6: Run focused tests and commit**

Run: `python -m pytest -q tests/test_staging.py tests/test_a2v_execution.py tests/test_train_a2v_script.py tests/test_training_execution.py`

Expected: all tests pass with zero Fal calls.

```powershell
git add src/ltx_lora_pilot/staging.py src/ltx_lora_pilot/a2v_execution.py src/ltx_lora_pilot/fal_api.py scripts/train_a2v.py tests/test_staging.py tests/test_a2v_execution.py tests/test_train_a2v_script.py tests/test_training_execution.py
git commit -m "feat: enforce immutable A2V paid execution"
```

---

### Task 8: Separately bound paid validation inference

**Files:**
- Create: `src/ltx_lora_pilot/validation_bundle.py`
- Create: `scripts/build_validation_bundle.py`
- Create: `scripts/run_validation_bundle.py`
- Create: `tests/test_validation_bundle.py`
- Modify: `src/ltx_lora_pilot/generation.py`
- Modify: `scripts/generate.py`
- Modify: `tests/test_generation.py`
- Modify: `tests/test_generate_script.py`

**Interfaces:**
- Consumes: Task 5A's atomic snapshot/head-guarded `PilotLedger`, the canonical private resolver, the completed training output digest, validation media, and a fresh validation authorization receipt.
- Produces: `ValidationBundle`, `build_validation_bundle`, `validate_validation_bundle`, and `execute_validation_bundle`.

- [ ] **Step 1: Write RED authorization-domain tests**

Test that a training receipt cannot authorize validation, validation receipts use a distinct domain/version allowlist, expired or replayed receipts fail, and the receipt binds the exact validation-bundle digest and current ledger head.

```python
def test_training_receipt_cannot_authorize_validation(training_receipt, validation_bundle) -> None:
    with pytest.raises(ValidationAuthorizationError, match="validation receipt"):
        validate_validation_bundle(validation_bundle, receipt=training_receipt)
```

- [ ] **Step 2: Run the domain test and verify RED**

Run: `python -m pytest -q tests/test_validation_bundle.py::test_training_receipt_cannot_authorize_validation`

Expected: FAIL because `validation_bundle.py` does not exist.

- [ ] **Step 3: Write RED immutable-input tests**

Cover binding of the LoRA SHA-256, supplied-audio SHA-256, start-image SHA-256, prompt bytes, endpoint, seed, frame count, LoRA scale, maximum cost, training bundle ID, training execution ID, and ledger head. Mutating any one field must change the bundle ID and invalidate the receipt.

```python
@pytest.mark.parametrize("field", ["prompt", "seed", "num_frames", "lora_scale", "max_cost"])
def test_validation_bundle_id_changes_for_any_paid_input(field, validation_spec) -> None:
    original = build_validation_bundle(validation_spec)
    changed = build_validation_bundle(validation_spec.replace(**mutated_value(field)))
    assert changed.bundle_id != original.bundle_id
```

- [ ] **Step 4: Implement canonical validation bundles**

Implement canonical JSON with UTF-8, sorted keys, no insignificant whitespace, decimal values encoded as fixed four-place strings, and SHA-256 bundle IDs. Reject unknown fields, non-allowlisted endpoints, missing/private-path media references, zero-length files, and any `debug_dataset` equivalent.

```python
VALIDATION_DOMAIN = "ltx-lora-validation/v1"

def validation_bundle_id(payload: Mapping[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256((VALIDATION_DOMAIN + "\n" + canonical).encode("utf-8")).hexdigest()
```

- [ ] **Step 5: Write RED reservation and replay tests**

Test separate cost allocation per render, one atomic receipt-bound ledger snapshot, `BEGIN IMMEDIATE` reservation with the receipt head as `expected_head_sha256`, no credential access before validation, durable `submit_started`, immediate private provider-ID persistence, ambiguous-submit commitment, no automatic retry, replay rejection, and the global $12 cap across training plus validation.

```python
def test_validation_cost_cannot_cross_global_cap(ledger, authorized_validation_bundle) -> None:
    ledger.reserve_training("bundle", "training", Decimal("6.0000"))
    with pytest.raises(BudgetExceeded):
        execute_validation_bundle(authorized_validation_bundle, ledger=ledger, max_cost=Decimal("6.5000"))
```

- [ ] **Step 6: Implement paid validation execution**

`execute_validation_bundle` must validate the bundle and receipt, resolve the ledger only from the approved root plus `pilot_id`, obtain one atomic snapshot, require the receipt's current ledger head, reserve the exact per-render maximum with that expected head inside the write transaction, stage and rehash local inputs, resolve credentials only after preflight, persist `submit_started`, submit once, persist the provider request ID privately, and finalize or retain commitment according to evidence. No code path may reuse the training receipt or accept a caller-provided ledger path.

- [ ] **Step 7: Disable the unsafe direct generation surface**

Change `scripts/generate.py --execute` to exit with a message directing callers to `build_validation_bundle.py` and `run_validation_bundle.py`. Keep a read-only payload-preview mode only when it performs no upload, no credential access, and no provider call.

```python
if args.execute:
    parser.error("paid execution requires a separately authorized validation bundle")
```

- [ ] **Step 8: Run focused tests and commit**

Run: `python -m pytest -q tests/test_validation_bundle.py tests/test_generation.py tests/test_generate_script.py`

Expected: all tests pass with zero Fal calls.

```powershell
git add src/ltx_lora_pilot/validation_bundle.py src/ltx_lora_pilot/generation.py scripts/build_validation_bundle.py scripts/run_validation_bundle.py scripts/generate.py tests/test_validation_bundle.py tests/test_generation.py tests/test_generate_script.py
git commit -m "feat: bind every paid validation render"
```

---

### Task 9: Public documentation, safe examples, and repository privacy gates

**Files:**
- Modify: `README.md`
- Modify: `configs/pilot.example.json`
- Create: `docs/A2V_EXPERIMENT_PLAN.md`
- Modify: `docs/TEST_PLAN.md`
- Modify: `tests/test_privacy.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: the final CLI surfaces from Tasks 3-8.
- Produces: a sanitized operator sequence and repository-wide privacy regression tests.

- [ ] **Step 1: Write RED privacy and example tests**

Test that tracked files contain no secret-like Fal keys, private Drive identifiers/URLs, personal names from the private objective, Windows user-profile paths, provider request IDs, signed provider URLs, source-media filenames, or private LoRA weight paths. Test that the example config uses the exact A2V endpoint, `steps: 1000`, `debug_dataset: false`, and no executable credential field.

```python
def test_public_example_is_exact_safe_a2v_configuration(repo_root: Path) -> None:
    config = json.loads((repo_root / "configs/pilot.example.json").read_text("utf-8"))
    assert config["endpoint"] == "fal-ai/ltx23-trainer-v2/a2v"
    assert config["steps"] == 1000
    assert config["debug_dataset"] is False
    assert "api_key" not in config
```

- [ ] **Step 2: Run privacy tests and verify RED**

Run: `python -m pytest -q tests/test_privacy.py`

Expected: FAIL until the new public artifacts and exact example are sanitized.

- [ ] **Step 3: Document the no-spend-to-paid command sequence**

Document these exact stages: configure the approved absolute private root and canonical opaque run layout; normalize and review private source media; accept/reject samples; build train/holdout manifests; validate A2V groups; create the exact two-item provider-validation selection; build the deterministic training bundle twice; capture fresh public price evidence; run policy-only preflight; issue the ledger-head-bound offline receipt; run receipt-required preflight; execute one paid training bundle with an atomic head guard; build a separate validation bundle and fresh head-bound receipt for each paid render; perform blind native-speed review; publish only sanitized generated evidence.

- [ ] **Step 4: Harden `.gitignore`**

Ignore private workspaces, raw media, deterministic bundles, authorization receipts, SQLite ledgers and sidecars, provider responses, request IDs, signed URLs, downloaded weights, staging trees, review ballots, and generated evidence until a specific sanitized file is intentionally allowlisted.

- [ ] **Step 5: Run complete verification**

Run:

```powershell
python -m pytest -q
python -m pytest -q tests/test_privacy.py
git diff --check
```

Expected: all tests pass and `git diff --check` has no output. `tests/test_privacy.py` scans tracked repository content for actual secret values, private identifiers, and private paths while allowing documentation to name environment variables and security concepts without creating false positives. Any new intentional vocabulary exception must be a narrow allowlisted literal in that test, never a broad ignored path.

- [ ] **Step 6: Commit and push the sanitized implementation**

```powershell
git add README.md configs/pilot.example.json docs/A2V_EXPERIMENT_PLAN.md docs/TEST_PLAN.md tests/test_privacy.py .gitignore
git commit -m "docs: publish safe Fal A2V operator workflow"
git push -u origin feat/fal-a2v-immutable-execution
```

---

### Task 10: Private no-spend rollout, one capped training run, and empirical quality decision

**Files:**
- Create privately: `<private-root>/pilots/<pilot-id>/ledger/pilot.sqlite3`
- Create privately: `<private-root>/pilots/<pilot-id>/runs/<execution-id>/plan.md`
- Create privately: `<private-root>/pilots/<pilot-id>/runs/<execution-id>/candidates/`
- Create privately: `<private-root>/pilots/<pilot-id>/runs/<execution-id>/control/`
- Create privately: `<private-root>/pilots/<pilot-id>/runs/<execution-id>/validation/provider-validation-selection.json`
- Create privately: `<private-root>/pilots/<pilot-id>/runs/<execution-id>/bundle/`
- Create privately: `<private-root>/pilots/<pilot-id>/runs/<execution-id>/staging/`
- Create privately: `<private-root>/pilots/<pilot-id>/runs/<execution-id>/outputs/private/`
- Create privately: `<private-root>/pilots/<pilot-id>/runs/<execution-id>/review/`
- Publish only specifically reviewed generated videos under: `results/videos/`

**Interfaces:**
- Consumes: the CLIs and gates implemented in Tasks 1-9, a deployment-approved absolute `LTX_LORA_PRIVATE_ROOT`, and the user-approved private source workspace.
- Produces: one provider training outcome, capped validation renders, a private blind-review record, and a public GO/NO-GO evidence summary without private identifiers.

- [ ] **Step 1: Record standing authorization without copying private text**

Create opaque `pilot_id`, `ledger_id`, and `execution_id` values, create the exact canonical private layout with owner-only permissions, and set `LTX_LORA_PRIVATE_ROOT` to its absolute approved root. Hash the exact user-provided objective bytes and record only the SHA-256, domain `ltx-lora-standing-authorization/v1`, capture timestamp, allowed endpoint, `steps: 1000`, maximum training cost `$6.0000`, global cap `$12.0000`, and expiration under the run's private `control/` directory. Do not copy the objective, chat, names, Drive URL, or credentials into the repository.

- [ ] **Step 2: Migrate the exact conservative legacy budget history**

Import exactly six historical entries totaling `$3.5409`, verify the migration-manifest digest, event-chain head, and replay uniqueness, and assert available uncommitted budget is `$8.4591` before new reservations.

```powershell
python scripts/migrate_budget_ledger.py --source-ledger <reviewed-private-source-ledger> --manifest <private-migration-manifest> --ledger <canonical-private-ledger>
```

The destination is resolved once from the approved private root and opaque `pilot_id`; no later command accepts that path. Require the migration result's verified committed and remaining values to equal `3.5409` and `8.4591`, then confirm them again through Task 5A's single atomic snapshot during policy-only preflight.

- [ ] **Step 3: Audit and normalize source footage without provider spend**

Copy only selected source media into the canonical run workspace. Normalize into A2V groups with exact opaque filenames `<group>_start.png`, `<group>_audio.wav`, `<group>_end.mp4`, and `<group>.txt`. Record source digest, session/location label, duration, resolution, FPS, audio properties, visible-face count, framing, occlusion, mouth visibility, and accept/reject reason in the private QA manifest.

- [ ] **Step 4: Enforce the data gate**

Require at least 10 accepted training groups and 5 accepted holdout groups, no session/location overlap between splits, at least two unseen holdout locations, and at least one holdout with clear inner-mouth visibility during speech. If any condition fails, stop with `$0` new spend and report the exact missing condition.

- [ ] **Step 5: Build and verify the deterministic training bundle**

Create and validate `a2v-provider-validation-selection-v1` with exactly two accepted holdout items, each containing one canonical prompt plus the local start-image/audio hash-and-size records. Bind it to the current structural-report and `a2v-execution-config-v2` digests. Build the bundle twice from the same accepted manifest and assert byte-identical ZIP SHA-256, bundle ID, selection IDs, execution config, and file manifest. Capture public price evidence no more than 24 hours before receipt issuance. Enforce `debug_dataset: false`; no runtime/signed URL may exist in the bundle.

- [ ] **Step 6: Issue the offline receipt and run final preflight**

Run policy-only preflight first and require `ready_for_policy_issuance`; its single atomic ledger snapshot must show `$8.4591` remaining and no replay. Use the offline issuer's snapshot reader to bind the standing-authorization hash, training bundle ID, execution-config digest, price-evidence digest, exact verified ledger head, endpoint, steps, `$6.0000` reservation, and expiry in a v2 receipt. Immediately run receipt-required preflight and require `ready_for_paid_execution`, `$8.4591` remaining before reservation, and `$2.4591` after reservation. If the ledger head or any source identity changes between issuance and preflight, stop and rebuild/re-authorize; never weaken or edit the receipt.

- [ ] **Step 7: Submit exactly one 1,000-step Fal A2V training job**

Execute the confirmed `run_dir` once. The reservation transaction must compare the current chain head to the head carried privately by the paid-ready preflight result before writing. Upload the guarded archive plus exactly two guarded validation image/audio pairs, and build only the in-memory provider items `{prompt, image_url, audio_url}`. Do not retry an ambiguous submit. Persist the request ID and provider result privately, but never persist runtime URLs in the immutable bundle or ordinary logs. After reservation, conservative committed spend becomes `$9.5409`; a price requiring more than the already authorized `$6.0000` stops before reservation rather than increasing the amount.

- [ ] **Step 8: Download and hash training outputs privately**

Verify output hashes and sizes, bind them to the training execution, store weights and provider metadata only in private storage, and never commit weights, provider URLs, or request IDs.

- [ ] **Step 9: Execute separately authorized validation renders**

Build a unique validation bundle and fresh ledger-head-bound receipt for every paid render. Each validation reservation uses the receipt head as an atomic compare-and-swap guard. Allocate at most `$1.2500` total validation reservation in this pilot, never cross the `$12.0000` global cap, and test exact supplied speech in multiple unseen locations at native output speed. Stop immediately if the remaining allocation cannot safely cover a requested render.

- [ ] **Step 10: Run blind native-speed review and publish the decision**

Randomize generated and genuine controls, collect blinded `real`, `AI`, or `unsure` ballots plus identity and speech-sync scores, then reveal labels only after ballots are sealed. The pilot is **NO-GO** if reviewers can reliably identify generated clips as AI, if identity is not recognized, or if exact speech/lip synchronization fails. Publish only sanitized generated examples and aggregate metrics; do not claim “indistinguishable” unless the blinded evidence supports it.

---

## Plan Self-Review Mapping

| Design requirement | Implemented by |
|---|---|
| Canonical schemas, digest domains, unknown-field rejection | Tasks 1, 3, 4, 5B, 5C, 8 |
| Strict A2V group validation and human QA | Task 2 |
| Deterministic ZIP and immutable root bundle | Task 3 |
| Standing authorization, fresh price evidence, offline receipt issuer | Tasks 4, 5B |
| Exact legacy migration, append-only SQLite chain, transactional reservations, global cap | Task 5 |
| Atomic identity/head/balance/replay snapshot and head-guarded reservation | Task 5A |
| Canonical private root/run/ledger resolution independent of `cwd` | Tasks 5A, 6 |
| Receipt bound to exact verified ledger-chain head | Task 5B |
| Exactly two structural/config-bound provider validation pairs with no runtime URLs | Task 5C |
| Safe archive inspection-before-extraction and shared fail-closed preflight | Task 6 |
| Exact sanitized preflight report and paid-only `require_ready()` | Task 6 |
| Content-addressed staging, retained handles, one-submit paid boundary | Task 7 |
| Separate hash-bound receipt for every paid validation render | Task 8 |
| Public documentation, safe examples, repository privacy | Task 9 |
| Data-count/session gates, single capped run, blind quality verdict | Task 10 |

Before Task 1 dispatch, run the placeholder, type/interface, privacy, and conflict scans required by the writing-plans and subagent-driven-development skills. Because Tasks 1–5 predate the contract correction, Tasks 5A, 5B, and 5C must each pass TDD plus independent Spec/Code review before Task 6 begins. The plan is executable only when those scans and correction gates are clean.
