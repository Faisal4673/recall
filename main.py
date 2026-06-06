import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from search import search_cache

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


def normalize_messages(messages):
    # Merge consecutive same-role turns into one so roles strictly alternate.
    # Injecting retrieved cache turns can create back-to-back user/assistant
    # entries; some models (e.g. deepseek-reasoner) require alternation and a
    # final user message. We copy entries so the canonical conversation is
    # left untouched.
    normalized = []
    for message in messages:
        if normalized and normalized[-1]["role"] == message["role"]:
            # Same role as the previous entry: fold the content together.
            normalized[-1]["content"] += "\n" + message["content"]
        else:
            normalized.append(dict(message))
    return normalized


def write_to_cache(role, content):
    # Append one message as a JSON dict (one per line) so the cache holds the
    # same {"role", "content"} structure the API consumes. Reload it later with:
    #   conversation = [json.loads(line) for line in open(KV_CACHE_PATH)]
    message = {"role": role, "content": content}
    with open(KV_CACHE_PATH, "a") as f:
        f.write(json.dumps(message) + "\n")


# On first run the cache doesn't exist yet, so seed it with the system prompt.
if not os.path.exists(KV_CACHE_PATH):
    write_to_cache("system", system_prompt)

# The conversation starts with the system prompt and grows as we chat.
conversation = [{"role": "system", "content": system_prompt}]

# Simple agent loop: read input, respond, remember, repeat.
while True:
    user_input = input("You: ")

    # Break the input into keywords, search the cache, and splice the best
    # matching real turns into the live conversation before the new prompt.
    relevant = search_cache(user_input)
    conversation.extend(relevant)

    # Add the user's message to the conversation history and persist it.
    conversation.append({"role": "user", "content": user_input})
    write_to_cache("user", user_input)

    # Generate a response, streaming it back token by token as it's produced.
    # Normalize first so injected turns don't break role alternation.
    stream = client.chat.completions.create(
        model="deepseek-chat",
        messages=normalize_messages(conversation),
        stream=True,
    )

    # Print each fragment as it arrives and accumulate the full reply.
    print("Assistant: ", end="", flush=True)
    reply = ""
    for chunk in stream:
        piece = chunk.choices[0].delta.content
        if piece:
            reply += piece
            print(piece, end="", flush=True)
    print()  # newline after the streamed response

    # Add the assistant's reply to the history and persist it.
    conversation.append({"role": "assistant", "content": reply})
    write_to_cache("assistant", reply)
