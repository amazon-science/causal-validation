"""MkDocs hook: replace vulnerable jQuery references in generated pages.

The ``mknotebooks`` plugin injects a Jupyter-widgets renderer snippet into every
rendered notebook page. That snippet hardcodes an old jQuery version
(``jquery/2.0.3/jquery.min.js``) loaded from a CDN. jQuery < 3.5.0 is subject to
known Cross-Site Scripting and Prototype Pollution vulnerabilities
(AWS AppSec ACAT rule ``JQueryVulnerableVersion``).

This hook runs after each page is rendered and rewrites any referenced jQuery
version below the minimum safe version to a pinned, patched version. It keeps the
docs functional (the widget renderer still loads jQuery) while removing the
vulnerable dependency from the deployed ``gh-pages`` output.
"""

from __future__ import annotations

import re

# Minimum non-vulnerable jQuery version and the pinned replacement we upgrade to.
MIN_SAFE = (3, 5, 0)
SAFE_VERSION = "3.7.1"

# Matches CDN references like:
#   https://cdnjs.cloudflare.com/ajax/libs/jquery/2.0.3/jquery.min.js
#   https://code.jquery.com/jquery-1.12.4.min.js
_CDNJS_RE = re.compile(r"(ajax/libs/jquery/)(\d+)\.(\d+)\.(\d+)(/jquery)")
_JQUERY_COM_RE = re.compile(r"(jquery-)(\d+)\.(\d+)\.(\d+)(\.min\.js)")


def _is_vulnerable(major: int, minor: int, patch: int) -> bool:
    return (major, minor, patch) < MIN_SAFE


def _bump_cdnjs(match: re.Match) -> str:
    major, minor, patch = (int(match.group(i)) for i in (2, 3, 4))
    if _is_vulnerable(major, minor, patch):
        return f"{match.group(1)}{SAFE_VERSION}{match.group(5)}"
    return match.group(0)


def _bump_jquery_com(match: re.Match) -> str:
    major, minor, patch = (int(match.group(i)) for i in (2, 3, 4))
    if _is_vulnerable(major, minor, patch):
        return f"{match.group(1)}{SAFE_VERSION}{match.group(5)}"
    return match.group(0)


def on_post_page(output: str, page, config) -> str:  # noqa: ANN001, ARG001
    """Rewrite vulnerable jQuery references in a rendered page's HTML."""
    patched = _CDNJS_RE.sub(_bump_cdnjs, output)
    patched = _JQUERY_COM_RE.sub(_bump_jquery_com, patched)
    return patched
