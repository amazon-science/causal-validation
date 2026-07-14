"""MkDocs hook: pin the jQuery version used by rendered notebook pages.

The ``mknotebooks`` plugin injects a Jupyter-widgets renderer snippet into every
rendered notebook page. That snippet references an old jQuery release from a CDN.
This hook normalises any older jQuery reference in the generated HTML to a single
current, pinned release so the documentation site loads a consistent version.
"""

from __future__ import annotations

import re

# Minimum jQuery version to keep; anything below is bumped to PINNED_VERSION.
MIN_VERSION = (3, 5, 0)
PINNED_VERSION = "3.7.1"

# Matches CDN references such as:
#   https://cdnjs.cloudflare.com/ajax/libs/jquery/2.0.3/jquery.min.js
#   https://code.jquery.com/jquery-1.12.4.min.js
_CDNJS_RE = re.compile(r"(ajax/libs/jquery/)(\d+)\.(\d+)\.(\d+)(/jquery)")
_JQUERY_COM_RE = re.compile(r"(jquery-)(\d+)\.(\d+)\.(\d+)(\.min\.js)")


def _below_min(major: int, minor: int, patch: int) -> bool:
    return (major, minor, patch) < MIN_VERSION


def _bump_cdnjs(match: re.Match) -> str:
    major, minor, patch = (int(match.group(i)) for i in (2, 3, 4))
    if _below_min(major, minor, patch):
        return f"{match.group(1)}{PINNED_VERSION}{match.group(5)}"
    return match.group(0)


def _bump_jquery_com(match: re.Match) -> str:
    major, minor, patch = (int(match.group(i)) for i in (2, 3, 4))
    if _below_min(major, minor, patch):
        return f"{match.group(1)}{PINNED_VERSION}{match.group(5)}"
    return match.group(0)


def on_post_page(output: str, page, config) -> str:  # noqa: ANN001, ARG001
    """Pin older jQuery references in a rendered page's HTML."""
    patched = _CDNJS_RE.sub(_bump_cdnjs, output)
    patched = _JQUERY_COM_RE.sub(_bump_jquery_com, patched)
    return patched
