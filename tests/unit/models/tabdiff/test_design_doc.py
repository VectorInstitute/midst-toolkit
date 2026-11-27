"""Optional docs test: paths named in personal `.notes/tabdiff-DESIGN.md` exist.

DESIGN.md lives under `.notes/` (gitignored). This test skips when the file is absent.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
TABDIFF_DIR = REPO_ROOT / "src" / "midst_toolkit" / "models" / "tabdiff"
DESIGN_DOC = REPO_ROOT / ".notes" / "tabdiff-DESIGN.md"
PATH_PATTERN = re.compile(r"`([\w./-]+\.(?:py|toml|yaml|md|json))`")


def _section(text: str, heading: str) -> str:
    """Return the body of a level-2 markdown section, without the heading line."""
    marker = f"## {heading}"
    if marker not in text:
        raise AssertionError(f"section '{heading}' not found in DESIGN.md")
    after = text.split(marker, 1)[1]
    next_heading = after.find("\n## ")
    if next_heading >= 0:
        after = after[:next_heading]
    return after


@pytest.mark.skipif(not DESIGN_DOC.is_file(), reason="personal .notes/tabdiff-DESIGN.md not present")
def test_current_layout_paths_exist() -> None:
    section = _section(DESIGN_DOC.read_text(), "Current layout")
    paths = PATH_PATTERN.findall(section)
    assert paths, "no file paths found in the Current layout section"
    missing = [path for path in paths if path != "DESIGN.md" and not (TABDIFF_DIR / path).exists()]
    assert not missing, f"DESIGN.md references missing files: {missing}"
