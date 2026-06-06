"""Tool: run a Python script inside the agent workspace.

Takes a snippet of Python source, runs it with the project interpreter in a
subprocess whose working directory is the workspace, and returns its stdout,
stderr, and exit code.

Honest boundary (see _workspace and comments.md): cwd is pinned to the
workspace, but Python can open any path or reach the network -- this tool is NOT
confined to the sandbox the way the file tools are. It's a convenience for a
trusted, local, learning setup, not a security boundary.
"""

import subprocess
import sys

from . import _workspace

# Kill runaway scripts so one bad call can't hang the whole agent.
TIMEOUT_SECONDS = 30

SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "Execute a Python script and return its stdout, stderr, and exit "
            "code. Runs with the workspace as the working directory. Use for "
            "computation, inspecting data, or testing code."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source to execute.",
                },
            },
            "required": ["code"],
        },
    },
}


def run(code):
    try:
        # `-c` runs the source directly; cwd pins execution to the workspace.
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=_workspace.WORKSPACE,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"Error: script exceeded the {TIMEOUT_SECONDS}s time limit."

    return _format(result)


def _format(result):
    # Stitch stdout/stderr/exit code into one readable block, omitting empties.
    parts = []
    if result.stdout:
        parts.append("stdout:\n" + result.stdout.rstrip())
    if result.stderr:
        parts.append("stderr:\n" + result.stderr.rstrip())
    parts.append(f"exit code: {result.returncode}")
    return "\n".join(parts)
