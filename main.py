import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from functs import load_tools
import ui

load_dotenv()

# DeepSeek is OpenAI-compatible, so reuse the OpenAI client with its base URL.
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

with open("system_prompt.txt") as f:
    system_prompt = f.read()

KV_CACHE_PATH = "global_kv_cache.txt"

# tool_schemas is advertised to the API; tool_dispatch maps a name to its run().
tool_schemas, tool_dispatch = load_tools()


def cache_message(message):
    # Append one message as a JSON line. Only user inputs and final assistant
    # answers are cached; tool-call and tool-result messages are not persisted.
    with open(KV_CACHE_PATH, "a") as f:
        f.write(json.dumps(message) + "\n")


if not os.path.exists(KV_CACHE_PATH):
    cache_message({"role": "system", "content": system_prompt})

# Past turns aren't auto-loaded; the agent reaches them via the `recall` tool.
conversation = [{"role": "system", "content": system_prompt}]


def execute_tool(name, arguments):
    # Tools return error text rather than raising, and the call is wrapped too,
    # so a bad tool call feeds an error back to the model instead of crashing.
    func = tool_dispatch.get(name)
    if func is None:
        return f"Error: unknown tool {name!r}."
    try:
        kwargs = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        return f"Error: could not parse arguments for {name}: {arguments!r}"

    # Echo the call, truncating long string args so the trace stays readable.
    shown = {k: (v[:60] + "..." if isinstance(v, str) and len(v) > 60 else v)
             for k, v in kwargs.items()}
    print(f"\n[running {name}({shown})]")
    try:
        return func(**kwargs)
    except Exception as err:
        return f"Error running {name}: {err}"


def chat_loop():
    ui.welcome()
    while True:
        try:
            user_input = ui.animated_input("You")
        except (EOFError, KeyboardInterrupt):
            print()  # tidy newline, then exit on Ctrl-D / Ctrl-C
            break

        user_message = {"role": "user", "content": user_input}
        conversation.append(user_message)
        cache_message(user_message)

        # Inner loop: let the model call tools and see their results until it
        # returns a plain text answer with no further tool calls.
        while True:
            stream = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=conversation,
                tools=tool_schemas,
                stream=True,
            )

            # Stream the reply, printing text and assembling tool calls from
            # their fragments. Tool-call deltas arrive per index: id and name
            # come once, arguments accumulates. Reasoning streams first in
            # `reasoning_content`, shown via the thinking indicator.
            reply = ""
            tool_calls = {}    # index -> {"id", "name", "arguments"}
            thinking = None
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

            # Assemble the assistant message: text plus any tool calls ordered
            # by index. content is None for a pure tool call.
            ordered = [tool_calls[i] for i in sorted(tool_calls)]
            assistant_message = {"role": "assistant", "content": reply or None}
            if ordered:
                assistant_message["tool_calls"] = [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"], "arguments": c["arguments"]}}
                    for c in ordered
                ]
            conversation.append(assistant_message)

            # No tools requested: the final answer. Only this plain-text answer
            # is persisted; the tool-call rounds stay in this session only.
            if not ordered:
                cache_message(assistant_message)
                break

            # Run each tool and feed its result back as a tool message, then
            # loop so the model can use the results. Tool messages aren't cached.
            for c in ordered:
                result = execute_tool(c["name"], c["arguments"])
                tool_message = {
                    "role": "tool",
                    "tool_call_id": c["id"],
                    "content": result,
                }
                conversation.append(tool_message)


if __name__ == "__main__":
    chat_loop()
