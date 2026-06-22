"""Sandbox guard shared by every file-touching tool.

`agent_workspace/` is the only directory the file tools may read or write; every
path goes through `resolve()`, which rejects absolute paths, `..` traversal, and
out-pointing symlinks. This confines the file tools only -- run_python/run_bash
start in the workspace but can still open any path or hit the network.

The underscore prefix keeps this out of tool discovery.
"""

import os

# Resolved once, following symlinks, so containment comparisons are exact.
WORKSPACE = os.path.realpath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "agent_workspace")
)

os.makedirs(WORKSPACE, exist_ok=True)


def resolve(path):
    """Resolve `path` relative to the workspace, or raise if it escapes.

    `os.path.join` drops the root for an absolute `path`, so it resolves to
    itself and fails containment. `realpath` collapses `..` and symlinks, so
    neither climbs out. A non-existent tail (a file about to be created) is fine.
    """
    target = os.path.realpath(os.path.join(WORKSPACE, path))
    if target != WORKSPACE and not target.startswith(WORKSPACE + os.sep):
        raise ValueError(
            f"path {path!r} escapes the agent workspace; access denied")
    return target


def relative(path):
    """Render an in-workspace absolute path back as workspace-relative."""
    rel = os.path.relpath(path, WORKSPACE)
    return "." if rel == "." else rel
