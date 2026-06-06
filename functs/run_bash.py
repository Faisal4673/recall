"""Tool: run a bash script inside the agent workspace.

Takes a shell snippet, runs it with bash in a subprocess whose working directory
is the workspace, and returns stdout, stderr, and exit code.

Honest boundary (see _workspace and comments.md): cwd is pinned to the
workspace, but bash can `cd` out, read any file, or reach the network -- this
tool is NOT confined to the sandbox the way the file tools are. It's the widest
capability the agent has; keep that in mind for an untrusted setup.
"""

import subprocess

from . import _workspace

TIMEOUT_SECONDS = 30

SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_bash",
        "description": (
            "Execute a bash script and return its stdout, stderr, and exit "
            "code. Runs with the workspace as the working directory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "script": {
                    "type": "string",
                    "description": "Bash script to execute.",
                },
            },
            "required": ["script"],
        },
    },
}


def run(script):
    try:
        result = subprocess.run(
            ["bash", "-c", script],
            cwd=_workspace.WORKSPACE,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"Error: script exceeded the {TIMEOUT_SECONDS}s time limit."

    # Same stdout/stderr/exit-code shape as run_python for a consistent contract.
    parts = []
    if result.stdout:
        parts.append("stdout:\n" + result.stdout.rstrip())
    if result.stderr:
        parts.append("stderr:\n" + result.stderr.rstrip())
    parts.append(f"exit code: {result.returncode}")
    return "\n".join(parts)
