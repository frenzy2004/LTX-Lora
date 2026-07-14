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


def test_repository_has_no_private_artifacts() -> None:
    root = Path(__file__).resolve().parents[1]
    violations = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if ".git" in relative.parts or not path.is_file():
            continue
        if any(part in PRIVATE_DIRECTORY_NAMES for part in relative.parts):
            violations.append(str(relative))
        is_approved_generated_clip = (
            relative.parent == PUBLIC_GENERATED_MEDIA_ROOT and path.suffix.lower() == ".mp4"
        )
        if path.suffix.lower() in PRIVATE_MEDIA_SUFFIXES and not is_approved_generated_clip:
            violations.append(str(relative))
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
