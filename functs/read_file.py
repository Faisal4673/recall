"""Tool: read the contents of a text file inside the agent workspace."""

import os

from . import _workspace

# Cap how much we hand back so a huge file can't blow up the context window.
MAX_BYTES = 100_000

SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "Read and return the text contents of a file within the workspace."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative path of the file to read.",
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
    if not os.path.exists(target):
        return f"Error: {path!r} does not exist."
    if not os.path.isfile(target):
        return f"Error: {path!r} is not a file."

    try:
        with open(target, encoding="utf-8") as f:
            content = f.read(MAX_BYTES + 1)
    except UnicodeDecodeError:
        return f"Error: {path!r} is not a UTF-8 text file."

    if len(content) > MAX_BYTES:
        # Truncate rather than refuse, and say so, so the agent knows there's more.
        return content[:MAX_BYTES] + f"\n\n[truncated at {MAX_BYTES} bytes]"
    return content
