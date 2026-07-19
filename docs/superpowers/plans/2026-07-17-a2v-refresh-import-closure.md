# A2V Refresh Import-Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `scripts/refresh_a2v_run.py` issue a fresh immutable A2V run without loading network, credential, receipt, or ledger-capable modules.

**Architecture:** Separate the immutable A2V schema validators and static provenance verifier from the paid-execution authority layer. The refresh CLI and issuer will depend only on the pure contract and static-verification modules; dynamic preflight will retain receipt and ledger validation while consuming the same static verifier.

**Tech Stack:** Python 3.12, pytest, canonical JSON artifacts, Windows private-path controls.

## Global Constraints

- The CLI must expose exactly the ten existing non-help options and preserve its neutral public output/errors.
- The CLI import closure must exclude `urllib`, `socket`, `ltx_lora_pilot.authorization`, `ltx_lora_pilot.pilot_ledger`, `ltx_lora_pilot.preflight`, `ltx_lora_pilot.a2v_execution`, and `ltx_lora_pilot.fal_api` even under `--help`.
- Static source integrity, split preservation, DACL checks, and policy-only preflight behavior must remain equivalent.
- Do not read a provider key, make a paid call, mutate a source bundle, or stage private footage outside the existing private workspace.

---

### Task 1: Lock the runtime boundary with a clean-process test

**Files:**
- Modify: `tests/test_a2v_refresh.py`

**Interfaces:**
- Produces a subprocess probe that loads the real refresh script under `python -S` and reports prohibited loaded module names.

- [x] **Step 1: Write the failing test**

```python
completed = subprocess.run([sys.executable, "-S", "-c", probe], capture_output=True, text=True)
assert json.loads(completed.stdout) == []
```

- [x] **Step 2: Verify RED**

Run: `pytest tests/test_a2v_refresh.py::test_refresh_cli_import_closure_excludes_network_and_authority_modules -v`

Observed: FAIL because the loaded set contains authorization, pilot ledger, preflight, socket, and urllib modules.

- [x] **Step 3: Add the exact option allowlist regression**

```python
assert declared == {
    "--pilot-id", "--source-execution-id", "--expected-source-bundle-id",
    "--target-execution-id", "--created-at-utc", "--expires-at-utc",
    "--price-evidence", "--standing-authorization", "--validation-prompts",
    "--repository-commit",
}
```

### Task 2: Extract pure A2V contracts

**Files:**
- Create: `src/ltx_lora_pilot/a2v_contracts.py`
- Modify: `src/ltx_lora_pilot/authorization.py`
- Modify: `src/ltx_lora_pilot/provider_validation.py`
- Modify: `src/ltx_lora_pilot/staging.py`

**Interfaces:**
- Produces `PriceEvidence`, `StandingAuthorization`, `validate_execution_config`, and their schemas/constants from a module that imports no network or ledger implementation.
- Preserves imports from `ltx_lora_pilot.authorization` by re-exporting the moved public contracts.

- [x] **Step 1: Move the common contracts without changing their validation rules**

Create `a2v_contracts.py` containing the shared A2V constants, exact-field/timestamp/money/hash validators, `PriceEvidence`, `StandingAuthorization`, and `validate_execution_config`. It must import only standard-library validation dependencies.

- [x] **Step 2: Leave authority-only behavior in `authorization.py`**

Keep `capture_price_evidence`, receipt issue/verification, and their file/ledger operations in `authorization.py`; import/re-export the contract names from `a2v_contracts.py`. Import `urllib.request` only inside the actual price-fetch implementation.

- [x] **Step 3: Repoint static consumers**

```python
from .a2v_contracts import validate_execution_config
```

Use that import in `provider_validation.py` and `staging.py`, never the authority module.

- [x] **Step 4: Run contract and dependent tests**

Run: `pytest tests/test_authorization.py tests/test_provider_validation.py tests/test_staging.py -v`

Expected: PASS.

### Task 3: Extract static provenance verification

**Files:**
- Create: `src/ltx_lora_pilot/a2v_static_verification.py`
- Modify: `src/ltx_lora_pilot/preflight.py`
- Modify: `src/ltx_lora_pilot/a2v_refresh.py`

**Interfaces:**
- Produces `verify_static_a2v_bundle(private_root, run_dir, expected_bundle_id) -> StaticA2VBundle` and the static-gate internals used by dynamic preflight.
- Consumes only pure contracts, bundle/dataset/quality/artifact/private-workspace/provider-validation modules, and standard library APIs.

- [x] **Step 1: Move the existing static verifier as one source of truth**

Move `StaticA2VBundle`, private-path fingerprinting/DACL checks, canonical artifact loading, archive inspection, static gate verification, and `verify_static_a2v_bundle` from `preflight.py` into `a2v_static_verification.py`. Preserve the validation order and gate strings exactly.

- [x] **Step 2: Make dynamic preflight consume the extracted verifier**

```python
from .a2v_static_verification import _StaticGateFailure, _verify_static_gates, verify_static_a2v_bundle
```

Keep receipt freshness, ledger snapshots, final rechecks, and `run_preflight` in `preflight.py`; do not duplicate static validation.

- [x] **Step 3: Make refresh use the pure verifier and DACL checker**

Replace imports of `authorization` and `preflight` in `a2v_refresh.py` with the pure contracts/static verifier. Remove the `_preflight._WINDOWS_DACL_CHECK` reach-through while retaining the same private-path check.

- [x] **Step 4: Run verifier and refresh tests**

Run: `pytest tests/test_preflight.py tests/test_a2v_refresh.py -v`

Expected: all existing gate attribution, static provenance, and fresh-run tests pass.

### Task 4: Prove closure, preserve behavior, and review

**Files:**
- Modify: `tests/test_a2v_refresh.py`

- [x] **Step 1: Run the clean-process probe**

Run: `pytest tests/test_a2v_refresh.py::test_refresh_cli_import_closure_excludes_network_and_authority_modules -v`

Expected: PASS with `[]` from the real import closure.

- [x] **Step 2: Run all affected and full verification**

Run:

```powershell
pytest tests/test_a2v_refresh.py tests/test_preflight.py tests/test_authorization.py tests/test_provider_validation.py tests/test_staging.py -v
pytest -q
python -m py_compile scripts/refresh_a2v_run.py src/ltx_lora_pilot/a2v_contracts.py src/ltx_lora_pilot/a2v_static_verification.py
git diff --check
```

Expected: all pass; only task-owned files are staged; existing unrelated user changes remain untouched.

- [x] **Step 3: Independent review and push**

Review the implementation against the clean-process import closure and regression suite, then commit only the task-owned code/tests/docs and push `feat/fal-a2v-immutable-execution`.
