"""Where the inputs live.

Kept in one place because several entry points need the same two things -- the
rerelease ``pak0.pak`` and a BSPC binary -- and because a path that is right on
one machine is wrong on the next: earlier revisions of CLAUDE.md and the
wiki's BACKGROUND page named ``/mnt/c/...`` locations from a Windows-side
install that this host does not even mount.

So nothing here is a hardcoded path. Each lookup walks a candidate list and
returns the first that exists, and every caller can override it from the
command line. What is listed below is only where a *stock* install puts
things.

**Your own paths go in** :data:`LOCAL_FILE` -- ``kexnav.local``, beside
``kexnav.py``, which git ignores. One ``<what> <path>`` pair per line, ``~``
expanded, ``#`` starting a comment, repeat a key for a second location, and
everything there is tried before the stock candidates::

    pak   ~/games/quake2-rerelease/baseq2/pak0.pak
    bspc  ~/src/bspc/bspc

Nothing breaks when that file is absent, which is the normal case for a fresh
clone: the stock candidates and ``PATH`` still apply, and ``--pak`` /
``--bspc`` always win.
"""

import os
import re
import shutil

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: This machine's own paths, tried before the stock candidates. Untracked, so
#: a checkout carries nobody else's folder layout. Format in the module
#: docstring; absent is fine.
LOCAL_FILE = os.path.join(_ROOT, "kexnav.local")

#: Retail rerelease ``pak0.pak``, holding the 174 hand-authored nav files and
#: the 222 BSPs they pair with. Where Steam puts it, nothing more -- a build
#: of your own belongs in :data:`LOCAL_FILE`.
PAK_CANDIDATES = (
    os.path.expanduser("~/.steam/steam/steamapps/common/"
                       "Quake 2/rerelease/baseq2/pak0.pak"),
    os.path.expanduser("~/.local/share/Steam/steamapps/common/"
                       "Quake 2/rerelease/baseq2/pak0.pak"),
    "/mnt/c/Program Files (x86)/Steam/steamapps/common/"
    "Quake 2/rerelease/baseq2/pak0.pak",
    "C:/Program Files (x86)/Steam/steamapps/common/"
    "Quake 2/rerelease/baseq2/pak0.pak",
)

#: Q3's BSPC 2.1h. Empty on purpose: BSPC ships no binary and installs
#: nowhere in particular, so there is no stock location to guess at. A build
#: on ``PATH`` is found without being listed; anywhere else goes in
#: :data:`LOCAL_FILE`.
BSPC_CANDIDATES = ()

_COMMENT = re.compile(r"\s#.*$")


def local_candidates(what, path=None):
    """Paths listed for `what` in :data:`LOCAL_FILE`, in file order.

    A missing or unreadable file yields nothing. A local override is a
    convenience, never a requirement, so this never raises -- a typo costs you
    the override, not the run.
    """
    out = []
    try:
        with open(path or LOCAL_FILE, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return out
    for line in lines:
        line = _COMMENT.sub("", line.strip())
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0] == what:
            out.append(os.path.expanduser(parts[1].strip()))
    return out


def _hint(key, example):
    """How to make a lookup that just failed succeed next time."""
    return (f"Pass it explicitly, or add a line to {LOCAL_FILE}:"
            f"\n  {key}  {example}")


def bspc_hint():
    """The same advice for :func:`find_bspc`, which returns None rather than
    raising -- its callers are the ones that decide a missing BSPC is fatal."""
    return _hint("bspc", "/path/to/bspc/bspc")


class NotFound(Exception):
    """Raised when none of the candidates for an input exist."""

    def __init__(self, what, key, candidates, example):
        super().__init__(
            f"no {what} found. Tried:\n  "
            + ("\n  ".join(candidates) if candidates else "(nothing)")
            + "\n\n" + _hint(key, example))


def _first(what, key, candidates, example, check=os.path.isfile):
    for path in candidates:
        if check(path):
            return path
    raise NotFound(what, key, candidates, example)


def find_pak(override=None):
    """The rerelease pak0.pak."""
    if override:
        return override
    return _first("rerelease pak0.pak", "pak",
                  local_candidates("pak") + list(PAK_CANDIDATES),
                  "/path/to/rerelease/baseq2/pak0.pak")


def find_bspc(override=None):
    """A BSPC binary, or None if there is none -- callers that only
    cross-check should degrade rather than fail."""
    if override:
        return override
    for path in local_candidates("bspc") + list(BSPC_CANDIDATES):
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return shutil.which("bspc")
