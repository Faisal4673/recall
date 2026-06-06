"""Tool: list the contents of a directory inside the agent workspace."""

import os

from . import _workspace

SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_directory",
        "description": (
            "List the files and subdirectories in a directory within the "
            "workspace. Directories are marked with a trailing slash."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Workspace-relative directory to list. Defaults to the "
                        "workspace root ('.')."
                    ),
                },
            },
            "required": [],
        },
    },
}


def run(path="."):
    try:
        target = _workspace.resolve(path)
    except ValueError as err:
        return f"Error: {err}"
    if not os.path.exists(target):
        return f"Error: {path!r} does not exist."
    if not os.path.isdir(target):
        return f"Error: {path!r} is not a directory."

    # Sort for stable output; append "/" to directories so the agent can tell
    # files and folders apart at a glance.
    entries = []
    for name in sorted(os.listdir(target)):
        full = os.path.join(target, name)
        entries.append(name + "/" if os.path.isdir(full) else name)

    if not entries:
        return f"{_workspace.relative(target)} is empty."
    return f"Contents of {_workspace.relative(target)}:\n" + "\n".join(entries)
