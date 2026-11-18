"""Runtime loading of Agent Skills (SKILL.md directories).

Only the short metadata (``name`` + ``description``) is shown to the routing
LLM.  The full skill body is loaded *after* a skill is selected, so skill
instructions never live permanently in the base system prompt.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

import config

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_CACHE: dict[str, dict[str, Any]] = {}
_CACHE_MTIME: dict[str, float] = {}
_CACHE_LOCK = threading.Lock()


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Return ``(frontmatter, body)`` without requiring a YAML library."""
    if not text.startswith("---"):
        return "", text
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return "", text
    closing = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            closing = index
            break
    if closing is None:
        return "", text
    return "\n".join(lines[1:closing]), "\n".join(lines[closing + 1 :]).lstrip("\n")


def _parse_metadata(header: str) -> dict[str, Any]:
    if not header.strip():
        return {}
    try:
        import yaml

        parsed = yaml.safe_load(header)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Minimal fallback: ``key: value`` lines.
    metadata: dict[str, Any] = {}
    for line in header.splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            metadata[key] = value
    return metadata


def _list_support_files(directory: Path) -> list[str]:
    if not directory.exists():
        return []
    files = []
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(directory)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if any(part.startswith(".") for part in relative.parts):
            continue
        files.append(str(relative))
    return sorted(files)


def _read_skill(directory: Path) -> dict[str, Any] | None:
    skill_file = directory / "SKILL.md"
    if not skill_file.exists():
        return None
    text = skill_file.read_text(encoding="utf-8")
    header, body = _split_frontmatter(text)
    metadata = _parse_metadata(header)
    name = str(metadata.get("name") or directory.name).strip()
    description = str(metadata.get("description") or "").strip()
    if not name or not description:
        return None
    if not _NAME_RE.match(name):
        return None

    return {
        "name": name,
        "description": description,
        "body": body.strip(),
        "instructions": body.strip(),
        "path": str(directory),
        "scripts": _list_support_files(directory / "scripts"),
        "references": _list_support_files(directory / "references"),
        "assets": _list_support_files(directory / "assets"),
    }


def list_available_skills() -> list[dict[str, str]]:
    """Return metadata for every valid skill directory."""
    skills: list[dict[str, str]] = []
    if not config.SKILLS_DIR.exists():
        return skills

    for directory in sorted(config.SKILLS_DIR.iterdir()):
        if not directory.is_dir() or directory.name.startswith("."):
            continue
        skill_file = directory / "SKILL.md"
        if not skill_file.exists():
            continue
        mtime = skill_file.stat().st_mtime
        cached = _CACHE.get(directory.name)
        if cached is not None and _CACHE_MTIME.get(directory.name) == mtime:
            skills.append(
                {
                    "name": cached["name"],
                    "description": cached["description"],
                    "path": cached["path"],
                }
            )
            continue
        skill = _read_skill(directory)
        if skill is None:
            continue
        with _CACHE_LOCK:
            _CACHE[directory.name] = skill
            _CACHE_MTIME[directory.name] = mtime
        skills.append(
            {
                "name": skill["name"],
                "description": skill["description"],
                "path": skill["path"],
            }
        )
    return skills


def load_skill(name: str) -> dict[str, Any] | None:
    """Load one skill by name, returning ``None`` if it is missing or invalid."""
    name = (name or "").strip()
    if not _NAME_RE.match(name):
        return None

    directory = config.SKILLS_DIR / name
    skill_file = directory / "SKILL.md"
    if not skill_file.exists():
        # Allow the frontmatter name to differ slightly from the directory name.
        for available in list_available_skills():
            if available["name"] == name:
                directory = Path(available["path"])
                skill_file = directory / "SKILL.md"
                break
        else:
            return None

    mtime = skill_file.stat().st_mtime
    cached = _CACHE.get(directory.name)
    if cached is not None and _CACHE_MTIME.get(directory.name) == mtime:
        return dict(cached)

    skill = _read_skill(directory)
    if skill is None:
        return None
    with _CACHE_LOCK:
        _CACHE[directory.name] = skill
        _CACHE_MTIME[directory.name] = mtime
    return dict(skill)


def skill_catalog_text() -> str:
    """A compact catalogue (metadata only) for the skill-selection prompt."""
    lines = []
    for skill in list_available_skills():
        lines.append(f"- {skill['name']}: {skill['description']}")
    return "\n".join(lines)
