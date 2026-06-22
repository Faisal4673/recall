"""Tool definitions, one tool per module.

Each non-underscore module exposes two names:

    SCHEMA : the OpenAI function-calling definition; `function.name` must equal
             the module name so a tool_call maps back to its handler.
    run(**kwargs) -> str : the implementation. Returns error text rather than
             raising, so one bad call can't kill the loop.

Underscore-prefixed modules (e.g. _workspace) are shared helpers, skipped by
discovery.
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
