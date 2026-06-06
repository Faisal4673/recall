"""The sandbox guard shared by every file-touching tool.

`agent_workspace/` is the ONLY directory the agent is allowed to read or write.
Every tool that takes a path runs it through `resolve()` first, which turns a
caller-supplied (relative) path into an absolute one and refuses anything that
would land outside the workspace -- absolute paths, `..` traversal, and symlinks
that point out are all rejected.

Note the honest limit (see comments.md): this confines the *file* tools only.
The execution tools (run_python / run_bash) start inside the workspace but can
still escape it -- arbitrary code can open any path or hit the network. The
sandbox boundary is real for files, aspirational for code execution.

The underscore prefix marks this as shared infrastructure: the tool-discovery
convention skips `_`-prefixed modules, so this is never exposed as a tool.
"""

import os

# The workspace lives next to the project root (the parent of this functs/
# package). Resolve it once, following any symlinks, so comparisons are exact.
WORKSPACE = os.path.realpath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "agent_workspace")
)

# Make sure the sandbox exists before any tool tries to use it.
os.makedirs(WORKSPACE, exist_ok=True)


def resolve(path):
    """Resolve `path` relative to the workspace and confine it there.

    Returns an absolute path guaranteed to sit inside WORKSPACE. Raises
    ValueError otherwise. `os.path.join` drops the root when `path` is absolute,
    so an absolute path resolves to itself and fails the containment check --
    exactly what we want. `realpath` collapses `..` and resolves symlinks, so
    neither can be used to climb out. A non-existent tail (e.g. a file we're
    about to create) resolves fine: only the existing prefix is followed.
    """
    target = os.path.realpath(os.path.join(WORKSPACE, path))
    if target != WORKSPACE and not target.startswith(WORKSPACE + os.sep):
        raise ValueError(
            f"path {path!r} escapes the agent workspace; access denied")
    return target


def relative(path):
    """Render an absolute in-workspace path back as workspace-relative.

    Used for human-readable tool output so the agent sees `notes/todo.txt`
    rather than the machine's full absolute path (and never learns the host
    layout outside its sandbox)."""
    rel = os.path.relpath(path, WORKSPACE)
    return "." if rel == "." else rel
