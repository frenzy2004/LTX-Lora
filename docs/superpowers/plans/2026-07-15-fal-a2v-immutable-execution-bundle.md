# Fal A2V Immutable Execution Bundle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fail-closed, content-addressed Fal LTX 2.3 A2V training and validation workflow that cannot upload or spend against stale, unreviewed, replayed, or over-budget artifacts.

**Architecture:** Normalized A2V groups pass deterministic media checks and an explicit human QA attestation before a deterministic ZIP and root bundle ID are created. A standing authorization policy issues a one-time receipt for that exact bundle, a canonical SQLite ledger atomically reserves the fixed cost, and paid execution uploads only content-addressed staged bytes after every pre-submit gate succeeds. Post-training inference uses separate hash-bound validation bundles and the same cumulative ledger.

**Tech Stack:** Python 3.11+, standard library (`dataclasses`, `hashlib`, `json`, `sqlite3`, `urllib`, `zipfile`, `ctypes`), `ffmpeg`/`ffprobe`, `fal-client`, and `pytest`.

## Global Constraints

- The paid training endpoint is exactly `fal-ai/ltx23-trainer-v2/a2v`.
- The request is exactly rank 32, 1,000 steps, learning rate 0.0002, 89 frames, 24 fps, high resolution, and 9:16.
- `auto_scale_input`, `split_input_into_scenes`, and `debug_dataset` are false; audio normalization and pitch preservation are true.
- Training may not exceed $6.0000; validation is an allocation of at most $1.2500; cumulative committed spend may not exceed $12.0000.
- The extra $2 is not authorized by any environment variable, CLI flag, or standing policy.
- Execute mode accepts no mutable endpoint, dataset ZIP, steps, price, ledger path, or cap override.
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

### Task 6: Shared offline preflight

**Files:**
- Create: `src/ltx_lora_pilot/preflight.py`
- Create: `scripts/preflight_a2v.py`
- Create: `tests/test_preflight.py`

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces: `PreflightStatus`, `run_preflight(bundle_dir, confirmed_bundle_id, require_receipt)`, and sanitized `preflight-report.json`.

- [ ] **Step 1: Write a table-driven RED suite for every gate**

```python
@pytest.mark.parametrize(
    "mutation,expected_gate",
    [
        ("archive_byte", "archive_hash"),
        ("validation_asset", "bundle_hash"),
        ("request_steps", "request_allowlist"),
        ("stale_price", "price_freshness"),
        ("wrong_receipt", "receipt"),
        ("wrong_ledger", "ledger_identity"),
    ],
)
def test_preflight_fails_closed(mutation: str, expected_gate: str, ready_bundle: Path) -> None:
    mutate(ready_bundle, mutation)
    report = run_preflight(ready_bundle, bundle_id(ready_bundle), require_receipt=True)
    assert report.status == "failed"
    assert report.failed_gate == expected_gate
```

Use spies asserting zero calls to budget reservation, secret resolution, upload, submit, and poll for each failure.

- [ ] **Step 2: Run tests and confirm RED**

Run: `python -m pytest -q tests/test_preflight.py`

Expected: import fails because `preflight` does not exist.

- [ ] **Step 3: Implement ordered sanitized gates**

Order: private-root safety; supported versions; structural rerun from freshly extracted final ZIP; attestation; split isolation; archive inspection; all hashes/sizes; exact request allowlist; price evidence; root bundle ID; standing policy; receipt when required; ledger identity, integrity, chain head, remaining budget, and replay state.

The report exposes only gate names, counts, bundle ID, costs, and status. It never includes absolute paths, captions, notes, source/session/location IDs, provider URLs, or private filenames.

- [ ] **Step 4: Implement CLI states**

Without a receipt, success prints `ready_for_policy_issuance`. With `--require-receipt`, it prints `ready_for_paid_execution`. Any failed gate exits nonzero. The command has no execute switch and cannot import `fal_client`.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest -q tests/test_preflight.py`

Expected: all tests pass.

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
- Consumes: `run_preflight`, `PilotLedger`, and exact bundle/request artifacts.
- Produces: `StagedArtifactGuard`, `execute_training_bundle`, and the safe A2V command.

- [ ] **Step 1: Write RED staging mutation tests**

Test create-new content-addressed staging, private permissions, exact hash/size/file identity, source rename after staging, staged replacement, write attempt while guarded, and post-upload verification.

```python
def test_uploader_receives_guarded_staged_path(ready_bundle: Path, tmp_path: Path) -> None:
    with stage_bundle(ready_bundle, tmp_path) as staged:
        assert staged.training_zip.parent.name == staged.bundle_id
        assert staged.verify_unchanged()
```

- [ ] **Step 2: Implement retained platform guards**

On Windows use `CreateFileW(GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING)` to deny cooperative write/delete sharing. On POSIX use mode-0700 staging, mode-0400 files, retained descriptors, and shared `flock`. Record file identity, size, and SHA-256. The uploader receives only guarded staged paths.

- [ ] **Step 3: Write RED execution-boundary tests**

Cover no credential access before preflight/reservation, upload failure release, mutation-after-upload release, durable `submit_started` before submit, ambiguous submit remaining committed, immediate private request-ID persistence, exact endpoint/payload, no retry, and redacted logs.

- [ ] **Step 4: Implement exact execution**

```python
def execute_training_bundle(bundle_dir, confirmed_bundle_id, *, ledger, resolve_key, upload_fn, submit_fn):
    report = run_preflight(bundle_dir, confirmed_bundle_id, require_receipt=True)
    report.require_ready()
    reservation = ledger.reserve_training(report.bundle_id, report.execution_id, Decimal("6.0000"))
    with stage_bundle(bundle_dir) as staged:
        resolve_key()
        ledger.transition(reservation.id, "uploading")
        urls = upload_staged_assets(staged, upload_fn)
        staged.require_unchanged()
        ledger.transition(reservation.id, "submit_started")
        return submit_and_persist(report, urls, reservation, ledger, submit_fn)
```

Upload or mutation failures append `released` only when no `submit_started` event exists. Submit exceptions remain committed and are never retried automatically.

- [ ] **Step 5: Replace the A2V command surface**

Accept only `--bundle-dir`, `--confirm-bundle-id`, and `--execute`. Default is shared dry-run. Remove dataset ZIP, plan marker, steps, trigger, validation JSON, budget, ledger path, and cost overrides.

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
- Consumes: `PilotLedger`, the completed training output digest, validation media, and a fresh validation authorization receipt.
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

Test separate cost allocation per render, `BEGIN IMMEDIATE` reservation, no credential access before validation, durable `submit_started`, immediate private provider-ID persistence, ambiguous-submit commitment, no automatic retry, replay rejection, and the global $12 cap across training plus validation.

```python
def test_validation_cost_cannot_cross_global_cap(ledger, authorized_validation_bundle) -> None:
    ledger.reserve_training("bundle", "training", Decimal("6.0000"))
    with pytest.raises(BudgetExceeded):
        execute_validation_bundle(authorized_validation_bundle, ledger=ledger, max_cost=Decimal("6.5000"))
```

- [ ] **Step 6: Implement paid validation execution**

`execute_validation_bundle` must validate the bundle and receipt, reserve the exact per-render maximum, stage and rehash local inputs, resolve credentials only after preflight, persist `submit_started`, submit once, persist the provider request ID privately, and finalize or retain commitment according to evidence. No code path may reuse the training receipt.

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

Document these exact stages: normalize and review private source media; accept/reject samples; build train/holdout manifests; validate A2V groups; build the deterministic training bundle twice; capture fresh public price evidence; issue the offline receipt; run shared dry preflight; execute one paid training bundle; build a separate validation bundle for each paid render; perform blind native-speed review; publish only sanitized generated evidence.

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
- Create privately under the approved run workspace: `authorization/standing-objective.sha256`
- Create privately under the approved run workspace: `ledger/pilot.sqlite3`
- Create privately under the approved run workspace: `dataset/qa-manifest.jsonl`
- Create privately under the approved run workspace: `dataset/training/`
- Create privately under the approved run workspace: `dataset/holdout/`
- Create privately under the approved run workspace: `bundles/training/<bundle-id>/`
- Create privately under the approved run workspace: `bundles/validation/<bundle-id>/`
- Create privately under the approved run workspace: `outputs/private/`
- Publish only specifically reviewed generated videos under: `results/videos/`

**Interfaces:**
- Consumes: the CLIs and gates implemented in Tasks 1-9 plus the user-approved private source workspace.
- Produces: one provider training outcome, capped validation renders, a private blind-review record, and a public GO/NO-GO evidence summary without private identifiers.

- [ ] **Step 1: Record standing authorization without copying private text**

Hash the exact user-provided objective bytes and record only the SHA-256, domain `ltx-lora-standing-authorization/v1`, capture timestamp, allowed endpoint, `steps: 1000`, maximum training cost `$6.0000`, global cap `$12.0000`, and expiration. Do not copy the objective, chat, names, Drive URL, or credentials into the repository.

- [ ] **Step 2: Migrate the exact conservative legacy budget history**

Import exactly six historical entries totaling `$3.5409`, verify the migration-manifest digest, event-chain head, and replay uniqueness, and assert available uncommitted budget is `$8.4591` before new reservations.

```powershell
python scripts/migrate_budget_ledger.py --manifest <private-migration-manifest> --ledger <private-ledger>
python scripts/inspect_budget.py --ledger <private-ledger> --expect-committed 3.5409 --expect-remaining 8.4591
```

- [ ] **Step 3: Audit and normalize source footage without provider spend**

Copy only selected source media into the run workspace. Normalize into A2V groups with exact filenames `<sample>_start.png`, `<sample>_audio.wav`, `<sample>_end.mp4`, and `<sample>.txt`. Record source digest, session/location label, duration, resolution, FPS, audio properties, visible-face count, framing, occlusion, mouth visibility, and accept/reject reason in the private QA manifest.

- [ ] **Step 4: Enforce the data gate**

Require at least 10 accepted training groups and 5 accepted holdout groups, no session/location overlap between splits, at least two unseen holdout locations, and at least one holdout with clear inner-mouth visibility during speech. If any condition fails, stop with `$0` new spend and report the exact missing condition.

- [ ] **Step 5: Build and verify the deterministic training bundle**

Build it twice from the same accepted manifest and assert byte-identical ZIP SHA-256, bundle ID, request JSON, validation target IDs, and file manifest. Capture public price evidence no more than 24 hours before receipt issuance. Enforce `debug_dataset: false`.

- [ ] **Step 6: Issue the offline receipt and run final preflight**

Use the offline issuer to bind the standing-authorization hash, training bundle ID, request digest, price-evidence digest, exact ledger head, endpoint, steps, `$6.0000` reservation, and expiry. Final preflight must report `$8.4591` remaining before reservation and `$2.4591` after reservation.

- [ ] **Step 7: Submit exactly one 1,000-step Fal A2V training job**

Execute the confirmed bundle once. Do not retry an ambiguous submit. Persist the request ID and provider result privately. After reservation, conservative committed spend becomes `$9.5409` unless fresh provider evidence requires a higher safe reservation before submission.

- [ ] **Step 8: Download and hash training outputs privately**

Verify output hashes and sizes, bind them to the training execution, store weights and provider metadata only in private storage, and never commit weights, provider URLs, or request IDs.

- [ ] **Step 9: Execute separately authorized validation renders**

Build a unique validation bundle and receipt for every paid render. Allocate at most `$1.2500` total validation reservation in this pilot, never cross the `$12.0000` global cap, and test exact supplied speech in multiple unseen locations at native output speed. Stop immediately if the remaining allocation cannot safely cover a requested render.

- [ ] **Step 10: Run blind native-speed review and publish the decision**

Randomize generated and genuine controls, collect blinded `real`, `AI`, or `unsure` ballots plus identity and speech-sync scores, then reveal labels only after ballots are sealed. The pilot is **NO-GO** if reviewers can reliably identify generated clips as AI, if identity is not recognized, or if exact speech/lip synchronization fails. Publish only sanitized generated examples and aggregate metrics; do not claim “indistinguishable” unless the blinded evidence supports it.

---

## Plan Self-Review Mapping

| Design requirement | Implemented by |
|---|---|
| Canonical schemas, digest domains, unknown-field rejection | Tasks 1, 3, 4, 8 |
| Strict A2V group validation and human QA | Task 2 |
| Deterministic ZIP and immutable root bundle | Task 3 |
| Standing authorization, fresh price evidence, offline receipt issuer | Task 4 |
| Exact legacy migration, append-only SQLite chain, transactional reservations, global cap | Task 5 |
| Shared fail-closed preflight with no early credential access | Task 6 |
| Content-addressed staging, retained handles, one-submit paid boundary | Task 7 |
| Separate hash-bound receipt for every paid validation render | Task 8 |
| Public documentation, safe examples, repository privacy | Task 9 |
| Data-count/session gates, single capped run, blind quality verdict | Task 10 |

Before Task 1 dispatch, run the placeholder, type/interface, privacy, and conflict scans required by the writing-plans and subagent-driven-development skills. The plan is executable only when those scans are clean.
