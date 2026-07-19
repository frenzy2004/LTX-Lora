# Fresh A2V Bundle Issuer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create an offline-only, production-safe command that issues a new immutable A2V execution run from an accepted source dataset without mutating the source or making a provider request.

**Architecture:** `a2v_refresh.py` will validate a source run statically, securely copy the accepted candidate set into a private staging tree, rebuild all target-bound artifacts, and atomically publish a new canonical run. `refresh_a2v_run.py` will expose only canonical identifiers and already-recorded artifact paths; it will have no credential, endpoint, price, or paid-execution options.

**Tech Stack:** Python 3.12, standard library filesystem primitives, existing `ltx_lora_pilot` artifact/authorization/dataset/preflight modules, pytest.

## Global Constraints

- The issuer is offline-only: it must not import a Fal/provider client, perform network I/O, resolve a credential, issue a receipt, reserve budget, upload media, or queue training.
- Resolve source only with `require_canonical_run_dir`. Derive a non-existing destination solely from a new typed-ID helper under `<private-root>/pilots/<pilot-id>/runs/<execution-id>`; do not accept arbitrary run directories.
- The target execution ID must differ from the source, be canonical, and have no pre-existing directory, link, file, or alias.
- Preserve exactly the accepted source groups and split: 12 train groups and 5 holdout groups; copy exactly four regular, independent files per group.
- Validate source hashes and schemas statically without treating an expired source policy, price evidence, or execution expiry as execution authority.
- Fresh policy and price inputs must validate at the supplied creation timestamp and outlive the supplied target expiry.
- Target config is the fixed `a2v-execution-config-v2` contract: rank 32, 1,000 steps, `0.0002` learning rate, 89 frames/24 fps, high 9:16, fixed audio controls, and `6.0000`/`1.2500`/`12.0000` ceilings.
- Validation prompts are an explicit canonical map of exactly two distinct accepted holdout group IDs; never infer prompts or select training groups.
- Publish only after all target artifacts are valid, using a same-volume no-replace primitive (`MoveFileExW` without a replace-existing flag on Windows; an explicit fail-closed no-replace implementation elsewhere). On failure remove only the tracked unique staging tree; never overwrite an existing target, source, ledger, policy source, price source, or prompt map.
- Normal CLI output must be sanitized and must not contain credentials, media paths, raw source IDs, URLs, or provider details.
- Keep all unrelated working-tree edits unstaged and unmodified.

---

## File structure

- `src/ltx_lora_pilot/private_workspace.py` — a narrow typed-ID destination derivation helper and root-contained canonical private-file resolver.
- `src/ltx_lora_pilot/preflight.py` — a public static-bundle verifier which runs provenance/integrity gates but deliberately excludes freshness, receipt, ledger, and paid-execution gates.
- `src/ltx_lora_pilot/a2v_refresh.py` — FD-safe source/control copying, exact 17/12/5 checks, deterministic target construction, no-replace publication, and a sanitized result type.
- `scripts/refresh_a2v_run.py` — narrow, neutral-error CLI that uses the existing approved-private-root environment resolution and canonical IDs only.
- `tests/test_a2v_refresh.py` and `tests/test_private_workspace.py` — real filesystem tests for the issuer, static verifier, destination resolver, no-overwrite race, and CLI boundary.
- `docs/superpowers/specs/2026-07-16-a2v-fresh-bundle-issuer-design.md` — already committed design authority; no production behavior belongs in the document.

### Task 1: Source-static verifier and isolated candidate copy

**Files:**

- Create: `src/ltx_lora_pilot/a2v_refresh.py`
- Create: `tests/test_a2v_refresh.py`
- Modify: `src/ltx_lora_pilot/private_workspace.py`
- Modify: `src/ltx_lora_pilot/preflight.py`
- Modify: `tests/test_private_workspace.py`

**Interfaces:**

- Consumes: `require_canonical_run_dir`, `approved_private_root_from_environment`, `strict_load_json`, `canonical_json_bytes`, `validate_a2v_directory`, `validate_quality_and_splits`, `validate_execution_config`, `compute_bundle_id`.
- Produces: `canonical_new_run_dir(private_root, pilot_id, execution_id) -> Path`, `require_canonical_private_file(private_root, path) -> Path`, `verify_static_a2v_bundle(private_root, run_dir, expected_bundle_id) -> StaticA2VBundle`, `verify_source_run_static(private_root, pilot_id, source_execution_id, expected_source_bundle_id) -> SourceRunSnapshot`, and `copy_accepted_candidates(snapshot, destination) -> tuple[dict[str, Any], dict[str, Any]]` for later target construction.

`canonical_new_run_dir` intentionally has no source-execution argument because it is a generic absent-destination resolver. Task 2’s `refresh_sealed_a2v_run`, which receives both source and target execution IDs, must reject equality before it derives a target or creates staging. This allocation prevents a generic path helper from receiving unrelated source provenance while still enforcing the global no-reissue-in-place invariant at the first operation that can issue a target.

- [ ] **Step 1: Write the failing source-verification test**

```python
from ltx_lora_pilot.a2v_refresh import verify_source_run_static

def test_source_static_verifier_accepts_expired_execution_authority_when_bytes_are_bound(
    ready_source_run: ReadySourceRun,
) -> None:
    snapshot = verify_source_run_static(
        private_root=ready_source_run.private_root,
        pilot_id=ready_source_run.pilot_id,
        source_execution_id=ready_source_run.execution_id,
        expected_source_bundle_id=ready_source_run.bundle_id,
    )

    assert snapshot.run_dir == ready_source_run.run_dir
    assert snapshot.quality_summary["accepted_train_group_ids"] == ready_source_run.train_ids
    assert snapshot.quality_summary["accepted_holdout_group_ids"] == ready_source_run.holdout_ids
```

Create the fixture from the existing `tests.test_preflight._write_ready_run` shape, then alter only source root/config/policy/price timestamps after its root artifacts have been written. The assertion proves static provenance is distinct from current execution authority.

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_a2v_refresh.py::test_source_static_verifier_accepts_expired_execution_authority_when_bytes_are_bound -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'ltx_lora_pilot.a2v_refresh'`.

- [ ] **Step 3: Add the minimal source-verification implementation**

```python
@dataclass(frozen=True)
class SourceRunSnapshot:
    run_dir: Path
    structural_report: dict[str, Any]
    quality_attestation: dict[str, Any]
    quality_summary: dict[str, Any]
    source_config: dict[str, Any]

def verify_source_run_static(
    *,
    private_root: Path,
    pilot_id: str,
    source_execution_id: str,
    expected_source_bundle_id: str,
) -> SourceRunSnapshot:
    run_dir = require_canonical_run_dir(
        private_root, pilot_id, source_execution_id,
        Path(private_root) / "pilots" / pilot_id / "runs" / source_execution_id,
    )
    root = _load_canonical_object(run_dir / "bundle" / "bundle-manifest.json", label="source root")
    if compute_bundle_id(root) != expected_source_bundle_id:
        raise ValueError("source bundle identity mismatch")
    _verify_root_artifacts(run_dir, root)
    stored_structural = _load_canonical_object(run_dir / "control" / "structural-report.json", label="source structural report")
    stored_quality = _load_canonical_object(run_dir / "control" / "quality-attestation.json", label="source quality attestation")
    source_config = validate_execution_config(_load_canonical_object(run_dir / "control" / "execution-config.json", label="source config"))
    reproduced = validate_a2v_directory(run_dir / "candidates", trigger_phrase=source_config["trigger_phrase"])
    if canonical_json_bytes(reproduced) != canonical_json_bytes(stored_structural):
        raise ValueError("source structural report drift")
    summary = validate_quality_and_splits(stored_quality, reproduced)
    return SourceRunSnapshot(run_dir, reproduced, stored_quality, summary, source_config)
```

Implement these focused private helpers:

```python
def _load_canonical_object(path: Path, *, label: str) -> dict[str, Any]:
    _require_regular(path)
    value = strict_load_json(path)
    if type(value) is not dict or path.read_bytes() != canonical_json_bytes(value):
        raise ValueError(f"{label} must be canonical JSON")
    return value

def _require_regular(path: Path, *, directory: bool = False) -> None:
    if _is_symlink_or_junction(path):
        raise ValueError("private refresh input must not be a link")
    mode = path.stat(follow_symlinks=False).st_mode
    if directory and not stat.S_ISDIR(mode):
        raise ValueError("private refresh input must be a directory")
    if not directory and not stat.S_ISREG(mode):
        raise ValueError("private refresh input must be a regular file")
```

Before `a2v_refresh.py` is written, expose a public `verify_static_a2v_bundle` in `preflight.py`. It must execute the existing canonical-artifact, bundle-ID, root-artifact-pin, archive-inspection, archive-structural, candidate-rerun, quality, rebuilt-manifest, provider-selection, and request-allowlist checks in the same order as `run_preflight`; it must skip only old price/policy freshness, receipt validation, ledger inspection, and final temporal recheck. It must always close the retained archive and return deep-copied/static artifacts plus the canonical path context. `verify_source_run_static` calls this public verifier and then adds the exact 17 structural groups / 12 train / 5 holdout contract check.

- [ ] **Step 4: Run the focused test to verify it passes**

Run: `pytest tests/test_a2v_refresh.py::test_source_static_verifier_accepts_expired_execution_authority_when_bytes_are_bound -v`

Expected: PASS.

- [ ] **Step 5: Add negative tests before expanding the implementation**

```python
@pytest.mark.parametrize("mutation", ["wrong_bundle", "artifact_drift", "candidate_link"])
def test_source_static_verifier_rejects_unbound_or_aliased_source(
    ready_source_run: ReadySourceRun, mutation: str
) -> None:
    if mutation == "wrong_bundle":
        expected = "0" * 64
    elif mutation == "artifact_drift":
        (ready_source_run.run_dir / "control" / "structural-report.json").write_bytes(b"{}")
        expected = ready_source_run.bundle_id
    else:
        replace_with_link(ready_source_run.candidate_paths[0])
        expected = ready_source_run.bundle_id

    with pytest.raises(ValueError):
        verify_source_run_static(
            private_root=ready_source_run.private_root,
            pilot_id=ready_source_run.pilot_id,
            source_execution_id=ready_source_run.execution_id,
            expected_source_bundle_id=expected,
        )
```

Run: `pytest tests/test_a2v_refresh.py -k source_static_verifier -v`

Expected: PASS after the implementation rejects each mutation.

- [ ] **Step 6: Implement independent candidate copying and test source immutability**

```python
def test_copy_accepted_candidates_uses_independent_regular_files(
    ready_source_run: ReadySourceRun, tmp_path: Path
) -> None:
    snapshot = verify_source_run_static(
        private_root=ready_source_run.private_root,
        pilot_id=ready_source_run.pilot_id,
        source_execution_id=ready_source_run.execution_id,
        expected_source_bundle_id=ready_source_run.bundle_id,
    )
    target = tmp_path / "staging" / "candidates"
    structural, attestation = copy_accepted_candidates(snapshot, target)

    assert len(structural["groups"]) == 17
    assert all(path.stat().st_nlink == 1 for path in target.iterdir())
    assert source_digest_map(ready_source_run.candidates) == source_digest_map(ready_source_run.candidates)
    assert {path.name for path in target.iterdir()} == expected_group_file_names(structural)
```

Implement copy by reusing `staging._copy_sealed_file` for every candidate and control input: it provides no-follow FD opening, exclusive destination creation, stream hashing, fsync, source identity rechecks, and single-link output validation. Reject a source whose path is linked, a target alias, an unexpected candidate file, or a copied digest mismatch. Do not use `copytree`, hard links, or symlinks. Require exactly 17 structural groups, 68 candidate files, 12 accepted train IDs, and 5 accepted holdout IDs before and after the staged rerun.

- [ ] **Step 7: Run Task 1 tests and commit**

Run: `pytest tests/test_a2v_refresh.py -k 'source_static_verifier or accepted_candidates' -v`

Expected: all selected tests PASS.

```bash
git add src/ltx_lora_pilot/a2v_refresh.py tests/test_a2v_refresh.py
git commit -m "feat: verify and copy sealed A2V source runs"
```

### Task 2: Deterministic fresh target construction and atomic publication

**Files:**

- Modify: `src/ltx_lora_pilot/a2v_refresh.py`
- Modify: `tests/test_a2v_refresh.py`

**Interfaces:**

- Consumes: `SourceRunSnapshot`, `build_training_archive`, `build_dataset_manifest`, `build_provider_validation_selection`, `build_root_manifest`, `compute_bundle_id`, `StandingAuthorization.from_dict`, `PriceEvidence.from_dict`.
- Produces: `refresh_sealed_a2v_run(private_root, pilot_id, source_execution_id, expected_source_bundle_id, target_execution_id, created_at_utc, expires_at_utc, fresh_price_evidence_path, fresh_standing_authorization_path, validation_prompts_path, repository_commit) -> FreshA2VRunResult`.

- [ ] **Step 1: Write the failing fresh-issuance success test**

```python
from ltx_lora_pilot.a2v_refresh import refresh_sealed_a2v_run

def test_refresh_issues_a_new_bound_run_with_exact_train_holdout_and_selection(
    ready_source_run: ReadySourceRun,
    fresh_controls: FreshControls,
) -> None:
    result = refresh_sealed_a2v_run(
        private_root=ready_source_run.private_root,
        pilot_id=ready_source_run.pilot_id,
        source_execution_id=ready_source_run.execution_id,
        expected_source_bundle_id=ready_source_run.bundle_id,
        target_execution_id=fresh_controls.execution_id,
        created_at_utc=fresh_controls.created_at_utc,
        expires_at_utc=fresh_controls.expires_at_utc,
        fresh_price_evidence_path=fresh_controls.price_path,
        fresh_standing_authorization_path=fresh_controls.policy_path,
        validation_prompts_path=fresh_controls.prompts_path,
        repository_commit="a" * 40,
    )

    assert result.execution_id == fresh_controls.execution_id
    assert result.bundle_id == compute_target_bundle_id(result.run_dir)
    assert manifest_split_counts(result.run_dir) == (12, 5)
    assert selected_holdout_ids(result.run_dir) == sorted(fresh_controls.prompts)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_a2v_refresh.py::test_refresh_issues_a_new_bound_run_with_exact_train_holdout_and_selection -v`

Expected: FAIL because `refresh_sealed_a2v_run` is not defined.

- [ ] **Step 3: Implement fresh control loading and fixed target config**

```python
@dataclass(frozen=True)
class FreshA2VRunResult:
    execution_id: str
    bundle_id: str
    run_dir: Path

def refresh_sealed_a2v_run(
    *, private_root: Path, pilot_id: str,
    source_execution_id: str, expected_source_bundle_id: str,
    target_execution_id: str, created_at_utc: str, expires_at_utc: str,
    fresh_price_evidence_path: Path,
    fresh_standing_authorization_path: Path,
    validation_prompts_path: Path,
    repository_commit: str,
) -> FreshA2VRunResult:
    snapshot = verify_source_run_static(
        private_root=private_root, pilot_id=pilot_id,
        source_execution_id=source_execution_id,
        expected_source_bundle_id=expected_source_bundle_id,
    )
    target_dir = canonical_new_run_dir(private_root, pilot_id, target_execution_id)
    _require_absent_target(target_dir, source_execution_id, target_execution_id)
    price, policy, prompts = _load_fresh_controls(
        fresh_price_evidence_path, fresh_standing_authorization_path,
        validation_prompts_path, created_at_utc, expires_at_utc,
    )
    with _fresh_private_staging(private_root, pilot_id, target_execution_id) as staging:
        return _build_validate_publish_target(
            staging, target_dir, snapshot, price, policy, prompts,
            created_at_utc, expires_at_utc, repository_commit,
        )
```

Load the three external inputs only through `require_canonical_private_file` beneath the approved private root, then copy them into staging through `staging._copy_sealed_file` before parsing. Validate staged `PriceEvidence.from_dict(price, now=created_at_utc)` and `StandingAuthorization.from_dict(policy, now=created_at_utc)`, then require their expiries to be at or after `expires_at_utc`. Validate a staged prompt map as an exact JSON object with two canonical prompt strings; pass it directly to `build_provider_validation_selection` after copied candidates exist.

Create the fixed config from source `trigger_phrase` and `negative_prompt` only. Copy no source price/policy/config bindings into the fresh config; calculate new archive, manifest, policy and price digests. Call `validate_execution_config` before writing it.

- [ ] **Step 4: Add the root- and target-safety tests before publication code**

```python
@pytest.mark.parametrize("case", ["existing_target", "same_execution", "train_prompt", "one_prompt"])
def test_refresh_rejects_target_or_prompt_contract_violations(
    ready_source_run: ReadySourceRun, fresh_controls: FreshControls, case: str
) -> None:
    kwargs = refresh_kwargs(ready_source_run, fresh_controls)
    if case == "existing_target":
        target_run_dir(kwargs).mkdir(parents=True)
    elif case == "same_execution":
        kwargs["target_execution_id"] = ready_source_run.execution_id
    elif case == "train_prompt":
        write_prompt_map(kwargs["validation_prompts_path"], ready_source_run.train_ids[:2])
    else:
        write_prompt_map(kwargs["validation_prompts_path"], [ready_source_run.holdout_ids[0]])

    with pytest.raises(ValueError):
        refresh_sealed_a2v_run(**kwargs)
    assert not target_run_dir(kwargs).exists() or case == "existing_target"
```

Run: `pytest tests/test_a2v_refresh.py -k 'target_or_prompt' -v`

Expected: FAIL until target gating and prompt validation exist.

- [ ] **Step 5: Implement staging and atomic publication**

```python
with _fresh_private_staging(private_root, pilot_id, target_execution_id) as staging:
    _make_private_layout(staging)
    structural, attestation = copy_accepted_candidates(snapshot, staging / "candidates")
    _write_target_controls(staging, structural, attestation, price, policy, execution_config)
    archive = build_training_archive(groups, staging / "bundle" / "training-data.zip")
    manifest = build_dataset_manifest(structural, attestation, archive, candidate_dir=staging / "candidates")
    selection = build_provider_validation_selection(
        structural_report=structural, quality_summary=quality_summary,
        execution_config=execution_config, candidate_dir=staging / "candidates", prompts=prompts,
    )
    root = build_root_manifest(
        execution_id=target_execution_id, created_at_utc=created_at_utc,
        expires_at_utc=expires_at_utc, repository_commit=repository_commit,
        artifacts=artifact_digests, holdout_groups=manifest["groups"]["holdout"],
    )
    _verify_target_static(staging, root)
    _publish_new_run(staging, canonical_target_run_dir)
```

`_fresh_private_staging` must be sibling to the target inside the canonical `runs` directory, owner-only, and unique. `_publish_new_run` must use `MoveFileExW(staging, target, MOVEFILE_WRITE_THROUGH)` with no replace-existing flag on Windows; if that no-replace primitive is unavailable on another platform, fail closed rather than use `os.replace` or `os.rename`. Ensure target is absent, not linked, and still nonexisting immediately before publication; a target created in the final race must cause publication failure and leave that sentinel unchanged. Securely walk and validate the staging tree before publication with the preflight private-tree model; cleanup must remove only the tracked staging identity and must fail closed if a reparse/escape is detected. On Windows, use `path.stat(follow_symlinks=False)` and the project’s junction checks for every directory traversal boundary.

- [ ] **Step 6: Add full artifact and failure-cleanup tests**

```python
def test_refresh_failure_leaves_no_partial_target_and_never_changes_source(
    ready_source_run: ReadySourceRun, fresh_controls: FreshControls, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_before = tree_digests(ready_source_run.run_dir)
    monkeypatch.setattr("ltx_lora_pilot.a2v_refresh.build_root_manifest", fail_after_staging)

    with pytest.raises(RuntimeError, match="forced"):
        refresh_sealed_a2v_run(**refresh_kwargs(ready_source_run, fresh_controls))

    assert tree_digests(ready_source_run.run_dir) == source_before
    assert not target_run_dir(refresh_kwargs(ready_source_run, fresh_controls)).exists()
    assert staging_children(ready_source_run.private_root) == []
```

Also test deterministic issuance by invoking two independent private roots with byte-identical explicit inputs and asserting byte-for-byte equality of all target files except their intentionally distinct canonical root location. Add a target-publication race test that creates a sentinel target after the final absence check and before the no-replace move. It must assert the issuer raises, the sentinel contents remain unchanged, the source tree digests remain unchanged, and the staging session is removed only if its original tracked identity remains safe.

- [ ] **Step 7: Run Task 2 tests and commit**

Run: `pytest tests/test_a2v_refresh.py -v`

Expected: all issuer tests PASS.

```bash
git add src/ltx_lora_pilot/a2v_refresh.py tests/test_a2v_refresh.py
git commit -m "feat: issue immutable fresh A2V bundles"
```

### Task 3: Narrow CLI and policy-only preflight integration

**Files:**

- Create: `scripts/refresh_a2v_run.py`
- Modify: `tests/test_a2v_refresh.py`

**Interfaces:**

- Consumes: `refresh_sealed_a2v_run`.
- Produces: a sanitized command that emits only canonical JSON with `status`, `execution_id`, and `bundle_id`, plus a target that passes `run_preflight(run_dir, bundle_id, require_receipt=False, approved_private_root=private_root, clock=clock)` with its configured ledger.

- [ ] **Step 1: Write the failing CLI success and boundary tests**

```python
def test_refresh_cli_issues_target_and_exposes_no_paid_or_provider_options(
    ready_source_run: ReadySourceRun, fresh_controls: FreshControls
) -> None:
    completed = subprocess.run(refresh_command(ready_source_run, fresh_controls), capture_output=True, text=True)

    assert completed.returncode == 0
    assert json.loads(completed.stdout)["status"] == "issued"
    assert "fal" not in completed.stdout.lower()
    help_text = subprocess.run([sys.executable, str(REFRESH_SCRIPT), "--help"], capture_output=True, text=True).stdout
    for forbidden in ("--private-root", "--fal-key", "--endpoint", "--budget", "--execute", "--submit", "--media-url"):
        assert forbidden not in help_text

def test_fresh_issued_target_passes_policy_only_preflight(
    ready_source_run: ReadySourceRun, fresh_controls: FreshControls
) -> None:
    result = refresh_sealed_a2v_run(**refresh_kwargs(ready_source_run, fresh_controls))
    provision_matching_ledger(ready_source_run.private_root, fresh_controls.execution_id)
    status = run_preflight(
        result.run_dir, result.bundle_id, require_receipt=False,
        approved_private_root=ready_source_run.private_root, clock=lambda: FIXED_TIME,
    )
    assert status.ready is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_a2v_refresh.py -k 'refresh_cli or policy_only_preflight' -v`

Expected: FAIL because the CLI file does not exist and the result does not expose an execution run path.

- [ ] **Step 3: Implement the neutral CLI**

```python
parser.add_argument("--pilot-id", required=True)
parser.add_argument("--source-execution-id", required=True)
parser.add_argument("--expected-source-bundle-id", required=True)
parser.add_argument("--target-execution-id", required=True)
parser.add_argument("--created-at-utc", required=True)
parser.add_argument("--expires-at-utc", required=True)
parser.add_argument("--price-evidence", type=Path, required=True)
parser.add_argument("--standing-authorization", type=Path, required=True)
parser.add_argument("--validation-prompts", type=Path, required=True)
parser.add_argument("--repository-commit", required=True)
```

Use `approved_private_root_from_environment()` inside the CLI; do not expose a `--private-root` option. Use a custom `ArgumentParser.error` that emits exactly `A2V_REFRESH_ARGUMENT_ERROR`. Catch every issuer exception and emit exactly `A2V_REFRESH_FAILED`; do not print exception text. On success print `canonical_json_bytes` of the three public fields. Do not import `ltx_lora_pilot.fal_api`, `a2v_execution`, `httpx`, `urllib`, `socket`, or any credential resolver.

- [ ] **Step 4: Add import/network regression tests**

```python
def test_refresh_module_is_offline_only() -> None:
    source = REFRESH_MODULE.read_text(encoding="utf-8")
    for forbidden in ("fal_api", "a2v_execution", "httpx", "requests", "urllib", "socket", "os.environ"):
        assert forbidden not in source
```

Add a monkeypatch that raises if `socket.socket` is constructed while `refresh_sealed_a2v_run` executes. The actual issuance test must remain green under that patch.

- [ ] **Step 5: Run focused and full verification**

Run:

```bash
pytest tests/test_a2v_refresh.py -v
pytest tests/test_preflight.py tests/test_a2v_execution.py tests/test_provider_validation.py tests/test_a2v_bundle.py -v
pytest -q
python -m py_compile src/ltx_lora_pilot/a2v_refresh.py scripts/refresh_a2v_run.py
```

Expected: all tests PASS, no tracebacks, and compilation exits 0.

- [ ] **Step 6: Review, commit, push, and report the handoff condition**

Run:

```bash
git diff --check
git status --short
git add src/ltx_lora_pilot/a2v_refresh.py scripts/refresh_a2v_run.py tests/test_a2v_refresh.py
git commit -m "feat: add offline A2V refresh command"
git push origin feat/fal-a2v-immutable-execution
```

Expected: only the issuer files are committed; pre-existing unrelated edits remain unstaged. Report that the only remaining external requirement before the single paid call is a newly rotated private provider secret plus fresh price/policy/receipt artifacts.

## Self-review

- **Spec coverage:** Task 1 covers static source integrity, canonical source resolution, source expiry isolation, and independent file copies. Task 2 covers fresh control binding, exact split preservation, two heldout validations, fixed config, atomic publication, determinism, and cleanup. Task 3 covers the narrow CLI, no-provider boundary, policy-only preflight, full regression testing, commit, and push.
- **Placeholder scan:** No `TBD`, `TODO`, `implement later`, or unspecified error-handling steps are present. Each code task includes concrete inputs, commands, expected results, and target interfaces.
- **Type consistency:** `SourceRunSnapshot` is created by `verify_source_run_static` and consumed by copy/refresh helpers. `FreshA2VRunResult` is produced by `refresh_sealed_a2v_run` and consumed by the CLI/preflight tests. All artifact builders use existing exact-schema validators.
