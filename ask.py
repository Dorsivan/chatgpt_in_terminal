#!/usr/bin/env python3
import os
import sys
from typing import Iterable

from openai import OpenAI


MODEL_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-oss-20b")
MODEL_URL = os.getenv("MODEL_URL", "http://afc484d866d7843fa94860d25e3baeb8-2068025677.us-east-2.elb.amazonaws.com/demo-llm/gpt-oss-20b/v1")
SYSTEM_PROMPT = "You are an assistant that receives questions from http://afc484d866d7843fa94860d25e3baeb8-2068025677.us-east-2.elb.amazonaws.com/demo-llm/gpt-oss-20b/v1a user using a terminal. As such, you answers are displayed in the terminal, and are expected to be mostly short, concise and not use formats like .md"


def read_prompt_from_args_or_stdin(argv: list[str]) -> str:
    # Usage:
    #   ask.py "your question"
    #   echo "your question" | ask.py
    if len(argv) > 1:
        return " ".join(argv[1:]).strip()

    if not sys.stdin.isatty():
        return sys.stdin.read().strip()

    print("Usage: ask.py \"your question\"  (or pipe text into stdin)", file=sys.stderr)
    raise SystemExit(2)


def stream_text_events(events: Iterable[object]) -> int:
    """
    The Responses streaming API emits SSE events. We print deltas for:
      - response.output_text.delta  (partial text chunks)
    See event types in the API reference. :contentReference[oaicite:2]{index=2}
    """
    exit_code = 0
    try:
        for event in events:
            # In the Python SDK, events expose fields as attributes.
            etype = getattr(event, "type", None)

            if etype == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    print(delta, end="", flush=True)

            elif etype == "response.refusal.delta":
                # Optional: if a refusal happens, you'll see it streamed too.
                delta = getattr(event, "delta", "")
                if delta:
                    print(delta, end="", flush=True)
                    exit_code = 3

            elif etype == "response.failed":
                # Print something useful if available
                resp = getattr(event, "response", None)
                err = getattr(resp, "error", None) if resp else None
                msg = getattr(err, "message", None) if err else "Response failed."
                print(f"\n[error] {msg}", file=sys.stderr)
                return 1

            # You can also observe:
            # - response.created / response.in_progress / response.completed
            # - response.output_text.done (finalized text)
            # but we don't need them for a simple terminal streamer. :contentReference[oaicite:3]{index=3}

    except KeyboardInterrupt:
        print("\n[interrupted]", file=sys.stderr)
        return 130

    print()  # newline after streaming finishes
    return exit_code


def main() -> int:
    prompt = read_prompt_from_args_or_stdin(sys.argv)

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        return 2

    client = OpenAI(base_url=MODEL_URL)

    # Streaming is enabled with stream=True. :contentReference[oaichttp://afc484d866d7843fa94860d25e3baeb8-2068025677.us-east-2.elb.amazonaws.com/demo-llm/gpt-oss-20b/v1ite:4]{index=4}
    events = client.responses.create(
        model=MODEL_DEFAULT,
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt}
                ]
            }
        ],
        # input=prompt,
        stream=True,
    )

    return stream_text_events(events)


if __name__ == "__main__":
    raise SystemExit(main())
