"""Tool: create a directory (and parents) inside the agent workspace.

Kept as its own tool after merging create-file into write_file: making an empty
directory is the one creation primitive writing a file doesn't already cover.
"""

import os

from . import _workspace

SCHEMA = {
    "type": "function",
    "function": {
        "name": "make_directory",
        "description": (
            "Create a directory within the workspace, including any missing "
            "parent directories. Succeeds quietly if it already exists."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative directory to create.",
                },
            },
            "required": ["path"],
        },
    },
}


def run(path):
    try:
        target = _workspace.resolve(path)
    except ValueError as err:
        return f"Error: {err}"
    if os.path.isfile(target):
        return f"Error: {path!r} already exists as a file."

    os.makedirs(target, exist_ok=True)
    return f"Created directory {_workspace.relative(target)}."
