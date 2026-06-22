"""Tool: write text to a file inside the agent workspace.

Creates or overwrites; parent directories are created automatically.
"""

import os

from . import _workspace

SCHEMA = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": (
            "Write text to a file within the workspace, creating it (and any "
            "parent directories) if needed and overwriting it if it exists."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative path of the file to write.",
                },
                "content": {
                    "type": "string",
                    "description": "The full text to write into the file.",
                },
            },
            "required": ["path", "content"],
        },
    },
}


def run(path, content):
    try:
        target = _workspace.resolve(path)
    except ValueError as err:
        return f"Error: {err}"
    if os.path.isdir(target):
        return f"Error: {path!r} is a directory, not a file."

    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        f.write(content)

    return (f"Wrote {len(content)} bytes to "
            f"{_workspace.relative(target)}.")
