import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from search import search_cache
from functs import load_tools
import ui

# Load DEEPSEEK_API_KEY from the .env file into the environment.
load_dotenv()

# DeepSeek is OpenAI-compatible, so we reuse the OpenAI client with its base URL.
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

# Read the system prompt from disk so it can be edited without touching code.
with open("system_prompt.txt") as f:
    system_prompt = f.read()

# The global KV cache is a persistent log of the conversation across sessions.
KV_CACHE_PATH = "global_kv_cache.txt"

# Discover the tools once at startup. `tool_schemas` is advertised to the API on
# every call; `tool_dispatch` maps a tool name back to the callable that runs it.
tool_schemas, tool_dispatch = load_tools()


def is_plain_text(message):
    # A turn that's just text, safe to merge with an adjacent same-role turn.
    # Assistant messages carrying tool_calls and tool-result messages hold
    # structure (tool_calls / tool_call_id) that must survive verbatim, so they
    # are never "plain" and never merged.
    return message["role"] != "tool" and not message.get("tool_calls")


def normalize_messages(messages):
    # Produce an API-valid message list from the live conversation. Two jobs:
    #
    #  1. Merge consecutive same-role *plain text* turns into one, so injected
    #     recall turns don't break the user/assistant alternation some models
    #     (e.g. deepseek-reasoner) require. We copy entries so the canonical
    #     conversation is left untouched.
    #
    #  2. Keep tool exchanges valid. An assistant message with tool_calls must be
    #     followed by a tool result for each call before anything else; a tool
    #     result is only valid right after the call it answers. We pass those
    #     messages through unmerged and drop any orphaned tool result -- a
    #     {"role": "tool"} whose tool_call_id wasn't opened by a preceding
    #     assistant tool_calls message. Injection shouldn't create orphans, but
    #     this guarantees the user -> tool_call -> tool_result -> response order
    #     stays intact no matter what lands in the conversation.
    normalized = []
    open_tool_ids = set()  # tool_call ids still awaiting their result
    for message in messages:
        role = message["role"]

        if role == "assistant" and message.get("tool_calls"):
            # Opens one or more tool calls that must be answered next.
            normalized.append(dict(message))
            open_tool_ids = {tc["id"] for tc in message["tool_calls"]}
            continue

        if role == "tool":
            # Keep only results that answer a currently-open call; drop orphans.
            if message.get("tool_call_id") in open_tool_ids:
                normalized.append(dict(message))
                open_tool_ids.discard(message["tool_call_id"])
            continue

        # Any other message closes the tool-call window.
        open_tool_ids = set()

        if normalized and normalized[-1]["role"] == role and is_plain_text(normalized[-1]):
            # Same role as the previous plain turn: fold the content together.
            previous = normalized[-1].get("content") or ""
            current = message.get("content") or ""
            normalized[-1]["content"] = previous + "\n" + current
        else:
            normalized.append(dict(message))
    return normalized


def cache_message(message):
    # Append one full message dict as a JSON line. Storing the whole dict (not
    # just role/content) is what lets tool_calls and tool results persist across
    # sessions alongside ordinary turns. Reload the cache later with:
    #   conversation = [json.loads(line) for line in open(KV_CACHE_PATH)]
    with open(KV_CACHE_PATH, "a") as f:
        f.write(json.dumps(message) + "\n")


# On first run the cache doesn't exist yet, so seed it with the system prompt.
if not os.path.exists(KV_CACHE_PATH):
    cache_message({"role": "system", "content": system_prompt})

# The conversation starts with the system prompt and grows as we chat.
conversation = [{"role": "system", "content": system_prompt}]


def signature(message):
    # Identity used for injection dedup, stable across every message kind so a
    # re-retrieved turn matches what's already in the conversation exactly.
    # Tool messages share content shapes, so they're keyed by their structure
    # (tool_call ids / tool_call_id) rather than text alone.
    if message.get("tool_calls"):
        calls = tuple((c["id"], c["function"]["name"], c["function"]["arguments"])
                      for c in message["tool_calls"])
        return ("assistant", calls)
    if message["role"] == "tool":
        return ("tool", message.get("tool_call_id"), message.get("content") or "")
    return (message["role"], message.get("content") or "")


# Signatures of everything already in the conversation. Maintained incrementally
# so dedup is an O(1) lookup, never a re-scan of the whole list.
seen = {signature(conversation[0])}


def add_to_conversation(message):
    # Append a message and record it so future injections can skip duplicates.
    conversation.append(message)
    seen.add(signature(message))


def execute_tool(name, arguments):
    # Run a tool the model asked for and return its string result. Tools return
    # error text rather than raising, and we wrap the call too, so a bad tool
    # call feeds an error back to the model instead of killing the loop.
    func = tool_dispatch.get(name)
    if func is None:
        return f"Error: unknown tool {name!r}."
    try:
        kwargs = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        return f"Error: could not parse arguments for {name}: {arguments!r}"

    # Echo the call so the user can see what the agent is doing, truncating long
    # string args (e.g. file contents) so the trace stays readable.
    shown = {k: (v[:60] + "..." if isinstance(v, str) and len(v) > 60 else v)
             for k, v in kwargs.items()}
    print(f"\n[running {name}({shown})]")
    try:
        return func(**kwargs)
    except Exception as err:
        return f"Error running {name}: {err}"


def chat_loop():
    # Simple agent loop: read input, respond (calling tools as needed), remember.
    ui.welcome()
    while True:
        try:
            user_input = ui.animated_input("You")
        except (EOFError, KeyboardInterrupt):
            print()  # tidy newline, then exit on Ctrl-D / Ctrl-C
            break

        # Search the cache for relevant past turns. Each result is a whole turn
        # cluster -- a self-contained, API-valid sequence that may include
        # assistant(tool_calls)/tool-result rounds.
        relevant = search_cache(user_input)

        # Inject each cluster atomically: skip one already fully present (so a
        # re-retrieved turn doesn't compound), otherwise add the whole turn in
        # order. Injecting whole turns -- never fragments -- keeps tool exchanges
        # valid: a tool result always arrives with the call it answers.
        for cluster in relevant:
            if all(signature(message) in seen for message in cluster):
                continue
            for message in cluster:
                add_to_conversation(message)

        # Add the user's message to the conversation history and persist it.
        user_message = {"role": "user", "content": user_input}
        add_to_conversation(user_message)
        cache_message(user_message)

        # Inner loop: let the model call tools and see their results, repeating
        # until it returns a plain text answer with no further tool calls.
        while True:
            stream = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=normalize_messages(conversation),
                tools=tool_schemas,
                stream=True,
            )

            # Stream the reply, printing text as it arrives and assembling any
            # tool calls from their streamed fragments. Tool-call deltas arrive
            # per index: id and function.name come once, arguments accumulates.
            # Thinking models stream their reasoning in `reasoning_content`
            # first; we show an animated "thinking" indicator until the actual
            # answer (or a tool call) begins, then print under a red "agent:".
            reply = ""
            tool_calls = {}    # index -> {"id", "name", "arguments"}
            thinking = None    # the running thinking animation, if any
            answer_started = False
            for chunk in stream:
                if not chunk.choices:
                    continue  # final usage-only chunk has no choices
                delta = chunk.choices[0].delta

                if getattr(delta, "reasoning_content", None) and thinking is None:
                    thinking = ui.thinking_indicator()

                if delta.content:
                    if thinking is not None:
                        thinking.stop()
                        thinking = None
                    if not answer_started:
                        print(ui.AGENT_LABEL, end="", flush=True)
                        answer_started = True
                    reply += delta.content
                    print(delta.content, end="", flush=True)

                if delta.tool_calls:
                    if thinking is not None:
                        thinking.stop()
                        thinking = None
                    for tc in delta.tool_calls:
                        slot = tool_calls.setdefault(
                            tc.index, {"id": "", "name": "", "arguments": ""})
                        if tc.id:
                            slot["id"] = tc.id
                        if tc.function and tc.function.name:
                            slot["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            slot["arguments"] += tc.function.arguments

            if thinking is not None:
                thinking.stop()  # stream ended while still "thinking"
            if answer_started:
                print()  # newline after the streamed response

            # Assemble the assistant message in API shape: its text plus any
            # tool calls, ordered by index. content is None for a pure tool call.
            ordered = [tool_calls[i] for i in sorted(tool_calls)]
            assistant_message = {"role": "assistant", "content": reply or None}
            if ordered:
                assistant_message["tool_calls"] = [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"], "arguments": c["arguments"]}}
                    for c in ordered
                ]
            add_to_conversation(assistant_message)
            cache_message(assistant_message)

            # No tools requested: this is the final answer for this user turn.
            if not ordered:
                break

            # Run each requested tool and feed its result back as a tool message,
            # one per tool_call_id, then loop so the model can use the results.
            for c in ordered:
                result = execute_tool(c["name"], c["arguments"])
                tool_message = {
                    "role": "tool",
                    "tool_call_id": c["id"],
                    "content": result,
                }
                add_to_conversation(tool_message)
                cache_message(tool_message)


if __name__ == "__main__":
    chat_loop()
