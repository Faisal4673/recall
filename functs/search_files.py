"""Tool: grep-style regex search over file contents in the agent workspace.

Walks the workspace subtree, matches each text file line-by-line, and reports
`path:lineno: line`.
"""

import os
import re

from . import _workspace

# Bound the work so a large tree can't run away or flood context.
MAX_MATCHES = 100
MAX_FILE_BYTES = 1_000_000

SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_files",
        "description": (
            "Search file contents in the workspace for a regular-expression "
            "pattern. Returns matching lines as 'path:line: text'. Use this to "
            "locate code or text without reading whole files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Python regular expression to search for.",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "Workspace-relative directory to search under. "
                        "Defaults to the whole workspace ('.')."
                    ),
                },
            },
            "required": ["pattern"],
        },
    },
}


def run(pattern, path="."):
    try:
        regex = re.compile(pattern)
    except re.error as err:
        return f"Error: invalid regex {pattern!r}: {err}"
    try:
        root = _workspace.resolve(path)
    except ValueError as err:
        return f"Error: {err}"
    if not os.path.isdir(root):
        return f"Error: {path!r} is not a directory."

    matches = []
    for dirpath, _dirs, files in os.walk(root):
        for name in sorted(files):
            full = os.path.join(dirpath, name)
            # Skip oversized files and binaries (anything not UTF-8 decodable).
            try:
                if os.path.getsize(full) > MAX_FILE_BYTES:
                    continue
                with open(full, encoding="utf-8") as f:
                    lines = f.readlines()
            except (OSError, UnicodeDecodeError):
                continue

            rel = _workspace.relative(full)
            for lineno, line in enumerate(lines, start=1):
                if regex.search(line):
                    matches.append(f"{rel}:{lineno}: {line.rstrip()}")
                    if len(matches) >= MAX_MATCHES:
                        matches.append(f"[stopped at {MAX_MATCHES} matches]")
                        return "\n".join(matches)

    if not matches:
        return f"No matches for {pattern!r}."
    return "\n".join(matches)
