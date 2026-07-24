import hashlib
import json
from pathlib import Path
import re


PRIVATE_DIRECTORY_NAMES = {"private_inputs", "private_work", ".pilot_state", "outputs"}
PUBLIC_GENERATED_MEDIA_ROOT = Path("results/videos")
PRIVATE_MEDIA_SUFFIXES = {
    ".aac",
    ".avi",
    ".flac",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".safetensors",
    ".wav",
    ".webm",
    ".zip",
}
SECRET_PATTERNS = (
    re.compile(r"(?im)^\s*FAL_KEY\s*=\s*[^\s#]+$"),
    re.compile(r"[A-Za-z0-9_-]{24,}:[A-Za-z0-9_-]{24,}"),
)

APPROVED_PUBLIC_VIDEOS = {
    "a2v-lora-supplied-audio.mp4": {
        "bytes": 7_424_907,
        "sha256": "4bf3ac831afc266a6786dacd1e86cef3fa52b71137efa289f8993c6772db5d8a",
        "quality_status": "rejected_obviously_ai",
        "approval_date": "2026-07-14",
        "output_classification": "generated_output",
    },
    "i2v-lora-reference-conditioned.mp4": {
        "bytes": 3_157_507,
        "sha256": "0758df2edeb1717e61c496ba80aea305cab61143d4a392f38bfabae686831d57",
        "quality_status": "exploratory_single_sample",
        "approval_date": "2026-07-14",
        "output_classification": "generated_output",
    },
    "raindeer-round-1-01-t2v.mp4": {
        "bytes": 641_739,
        "sha256": "bcc67ec02b2efe29cea46d8fb65a404fe95be09be4de021b7a1fc980518bd9bb",
        "quality_status": "raindeer_round_proof",
        "approval_date": "2026-07-24",
        "output_classification": "generated_output",
    },
    "raindeer-round-1-02-t2v.mp4": {
        "bytes": 575_920,
        "sha256": "44d6d947f452922f385de5cd5b35e7e6d971d5995d5193d2ab5a3f289c510bdf",
        "quality_status": "raindeer_round_proof",
        "approval_date": "2026-07-24",
        "output_classification": "generated_output",
    },
    "raindeer-round-1-03-i2v.mp4": {
        "bytes": 704_577,
        "sha256": "1814f86710f47219c4195265c244e414cb0e6dd55c66646872b119004c059d16",
        "quality_status": "raindeer_round_proof",
        "approval_date": "2026-07-24",
        "output_classification": "generated_output",
    },
    "raindeer-round-2-01-t2v.mp4": {
        "bytes": 1_054_664,
        "sha256": "fe49dacf601003697c31d984631eb020ec53fc71ed855119276010bed1d60ae8",
        "quality_status": "raindeer_round_proof",
        "approval_date": "2026-07-24",
        "output_classification": "generated_output",
    },
    "raindeer-round-2-02-t2v.mp4": {
        "bytes": 906_956,
        "sha256": "9b8ebd5cb31c70ec9d77a786e26d55b6c9432c73a415f39870ed9d995c9c9cb5",
        "quality_status": "raindeer_round_proof",
        "approval_date": "2026-07-24",
        "output_classification": "generated_output",
    },
    "raindeer-round-2-03-i2v.mp4": {
        "bytes": 1_719_333,
        "sha256": "fed400613843e5d69becb5c2d730c73f2237e69e7d7651b0071688699c6096ba",
        "quality_status": "raindeer_round_proof",
        "approval_date": "2026-07-24",
        "output_classification": "generated_output",
    },
    "raindeer-round-3-01-t2v.mp4": {
        "bytes": 1_285_473,
        "sha256": "fa56807e2f8343468a00af3cce3563faaa984b0e0ba0a298bb8a256d9ebeac11",
        "quality_status": "raindeer_round_proof",
        "approval_date": "2026-07-24",
        "output_classification": "generated_output",
    },
    "raindeer-round-3-02-t2v.mp4": {
        "bytes": 1_414_712,
        "sha256": "9e1d6576115a6b9bb8cab185aec5e0d198fa5e584a3e26f03f6d30dac1fcda1b",
        "quality_status": "raindeer_round_proof",
        "approval_date": "2026-07-24",
        "output_classification": "generated_output",
    },
    "raindeer-round-3-03-i2v.mp4": {
        "bytes": 2_003_421,
        "sha256": "82155f9b1da65a86c8c3a3909f6977ca41a6ef8806ced6109e5c9cbdfe59751d",
        "quality_status": "raindeer_round_proof",
        "approval_date": "2026-07-24",
        "output_classification": "generated_output",
    },
    "raindeer-videoclaw-round1-burger-review-20s.mp4": {
        "bytes": 3_229_437,
        "sha256": "49cca58d1db34e03961c82402a23126689f132ab024d207538ae5211e85e2e6b",
        "quality_status": "raindeer_round1_render_only_burger_review_proof",
        "approval_date": "2026-07-24",
        "output_classification": "generated_output",
    },
    "t2v-lora-prompt-only-identity-failure.mp4": {
        "bytes": 2_332_942,
        "sha256": "4931530948986ea676496e63ca00838954deea5f92b1cabb6480365549498d35",
        "quality_status": "rejected_identity_failure",
        "approval_date": "2026-07-14",
        "output_classification": "generated_output",
    },
    "sync-v3-real-video-control.mp4": {
        "bytes": 16_305_156,
        "sha256": "7af11bb61f1f1b475c6ab8b99fd7d32c7392bddf3e1f3e7252537db0dac156ff",
        "quality_status": "promising_preservation_first_control_pending_blinded_review",
        "approval_date": "2026-07-15",
        "output_classification": "edited_real_footage_output",
    },
}


def _write_test_video_manifest(root: Path, video_path: Path) -> None:
    manifest_path = root / PUBLIC_GENERATED_MEDIA_ROOT / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": [
                    {
                        "filename": video_path.name,
                        "sha256": hashlib.sha256(video_path.read_bytes()).hexdigest(),
                        "bytes": video_path.stat().st_size,
                        "quality_status": "exploratory_single_sample",
                        "approval_date": "2026-07-14",
                        "manual_review": {
                            "output_classification": "generated_output",
                            "consent_or_authorization": "confirmed",
                            "embedded_source_asset_metadata": False,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def collect_private_artifact_violations(root: Path) -> list[str]:
    manifest_path = root / PUBLIC_GENERATED_MEDIA_ROOT / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest = {}

    manifest_entries = {}
    if isinstance(manifest, dict):
        for entry in manifest.get("files", []):
            if isinstance(entry, dict) and isinstance(entry.get("filename"), str):
                manifest_entries[entry["filename"]] = entry

    violations = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if ".git" in relative.parts or not path.is_file():
            continue
        if any(part in PRIVATE_DIRECTORY_NAMES for part in relative.parts):
            violations.append(str(relative))
            continue
        if path.suffix.lower() not in PRIVATE_MEDIA_SUFFIXES:
            continue

        entry = manifest_entries.get(path.name)
        is_approved_public_clip = (
            relative.parent == PUBLIC_GENERATED_MEDIA_ROOT
            and path.suffix.lower() == ".mp4"
            and entry is not None
            and entry.get("bytes") == path.stat().st_size
            and entry.get("sha256") == hashlib.sha256(path.read_bytes()).hexdigest()
        )
        if not is_approved_public_clip:
            violations.append(str(relative))
    return violations


def test_video_manifest_records_exact_approved_files() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / PUBLIC_GENERATED_MEDIA_ROOT / "manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["schema_version"] == 1
    assert len(manifest["files"]) == len(APPROVED_PUBLIC_VIDEOS)
    entries = {entry["filename"]: entry for entry in manifest["files"]}
    assert entries.keys() == APPROVED_PUBLIC_VIDEOS.keys()

    for filename, expected in APPROVED_PUBLIC_VIDEOS.items():
        entry = entries[filename]
        video_path = root / PUBLIC_GENERATED_MEDIA_ROOT / filename
        assert entry["bytes"] == expected["bytes"] == video_path.stat().st_size
        assert entry["sha256"] == expected["sha256"]
        assert hashlib.sha256(video_path.read_bytes()).hexdigest() == expected["sha256"]
        assert entry["quality_status"] == expected["quality_status"]
        assert entry["approval_date"] == expected["approval_date"]
        assert entry["manual_review"] == {
            "output_classification": expected["output_classification"],
            "consent_or_authorization": "confirmed",
            "embedded_source_asset_metadata": False,
        }


def test_public_video_policy_requires_manifest_name_size_and_hash(tmp_path: Path) -> None:
    video_path = tmp_path / PUBLIC_GENERATED_MEDIA_ROOT / "approved.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"approved")
    _write_test_video_manifest(tmp_path, video_path)

    assert collect_private_artifact_violations(tmp_path) == []

    unlisted_path = video_path.with_name("unlisted.mp4")
    unlisted_path.write_bytes(b"unlisted")
    assert str(unlisted_path.relative_to(tmp_path)) in collect_private_artifact_violations(tmp_path)
    unlisted_path.unlink()

    video_path.write_bytes(b"wrong-size")
    assert str(video_path.relative_to(tmp_path)) in collect_private_artifact_violations(tmp_path)

    video_path.write_bytes(b"tampered")
    assert video_path.stat().st_size == len(b"approved")
    assert str(video_path.relative_to(tmp_path)) in collect_private_artifact_violations(tmp_path)


def test_private_paths_and_media_suffixes_remain_forbidden(tmp_path: Path) -> None:
    private_note = tmp_path / "private_inputs" / "notes.txt"
    private_note.parent.mkdir(parents=True)
    private_note.write_text("private", encoding="utf-8")

    outside_video = tmp_path / "results" / "unapproved.mp4"
    outside_video.parent.mkdir(parents=True)
    outside_video.write_bytes(b"private video")

    public_audio = tmp_path / PUBLIC_GENERATED_MEDIA_ROOT / "unapproved.wav"
    public_audio.parent.mkdir(parents=True)
    public_audio.write_bytes(b"private audio")

    violations = collect_private_artifact_violations(tmp_path)
    assert {
        str(private_note.relative_to(tmp_path)),
        str(outside_video.relative_to(tmp_path)),
        str(public_audio.relative_to(tmp_path)),
    }.issubset(violations)


def test_repository_has_no_private_artifacts() -> None:
    root = Path(__file__).resolve().parents[1]
    violations = collect_private_artifact_violations(root)
    assert not violations, f"private artifacts found in: {violations}"


def test_repository_text_has_no_embedded_credentials() -> None:
    root = Path(__file__).resolve().parents[1]
    text_suffixes = {".md", ".py", ".json", ".toml", ".txt", ".example", ".gitignore"}
    violations = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if ".git" in relative.parts or not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            violations.append(str(relative))
    assert not violations, f"embedded credentials found in: {violations}"


def test_gitignore_excludes_private_artifacts() -> None:
    root = Path(__file__).resolve().parents[1]
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")
    for required in ("private_inputs/", "private_work/", ".pilot_state/", "outputs/", "*.safetensors"):
        assert required in gitignore
