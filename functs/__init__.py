"""Tool definitions for the agent, one tool per module.

Per-file convention (this is the contract the agent loop will rely on):

    Every tool module that is NOT underscore-prefixed exposes exactly two names:

      SCHEMA : dict
          The DeepSeek/OpenAI function-calling tool definition, shaped as
          {"type": "function",
           "function": {"name", "description", "parameters": {JSON Schema}}}.
          `function.name` MUST equal the module name, so a returned tool_call
          name maps straight back to the module that handles it.

      run(**kwargs) -> str
          The implementation. It receives the arguments the model supplied
          (already JSON-decoded), does the work, and returns a string -- the
          content sent back in the role:"tool" reply. Tools return error text
          rather than raising, so one bad call can't kill the loop.

    Underscore-prefixed modules (e.g. _workspace) are shared helpers and are
    skipped by discovery, never exposed as tools.

`load_tools()` below is the discovery step that turns this convention into a
registry. The agent loop (added later, separately) just calls it -- it never
hardcodes tool names.
"""

import importlib
import pkgutil


def load_tools():
    """Discover every tool module and return (schemas, dispatch).

    schemas:  list of SCHEMA dicts to pass as `tools=` on the API call.
    dispatch: {tool_name: run_callable} to execute a returned tool_call.
    """
    schemas = []
    dispatch = {}
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue  # shared helper, not a tool
        module = importlib.import_module(f"{__name__}.{info.name}")
        schemas.append(module.SCHEMA)
        dispatch[module.SCHEMA["function"]["name"]] = module.run
    return schemas, dispatch
