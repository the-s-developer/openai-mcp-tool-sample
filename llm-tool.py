import asyncio
import json
import argparse
from typing import Dict

from openai import AsyncOpenAI
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

# ⚠️  Replace with your own key or load from env var
MCP_URL = "http://localhost:8931/mcp"

# ────────────────────────────────────────────────────────────────────────────
# Helper to execute a tool via MCP
# ────────────────────────────────────────────────────────────────────────────

async def call_remote_tool_via_mcp(session: ClientSession, tool_name: str, arguments: dict):
    """Call a remote MCP tool and return its result."""
    return await session.call_tool(tool_name, arguments)


# ────────────────────────────────────────────────────────────────────────────
# Chat loop – accepts an initial user prompt
# ────────────────────────────────────────────────────────────────────────────

async def chat_loop(goal_prompt: str = "What's the weather like in Paris today?") -> None:
    client = AsyncOpenAI(api_key="")

    async with streamablehttp_client(MCP_URL) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Build OpenAI‑tool list with {"type":"function", "function":{…}}
            tools = (await session.list_tools()).tools
            print("session.list_tools():", tools)

            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": getattr(t, "description", "No description."),
                        "parameters": getattr(
                            t,
                            "inputSchema",
                            {
                                "type": "object",
                                "properties": {},
                                "required": [],
                            },
                        ),
                    },
                }
                for t in tools
            ]

            # Conversation history starts with the user‑provided goal prompt
            messages = [
                {
                    "role": "user",
                    "content": goal_prompt,
                }
            ]

            # Loop until model is satisfied (no more tool calls)
            while True:
                # Request streaming completion
                stream = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    tools=openai_tools,
                    stream=True,
                )

                assistant_content = ""
                partial_calls: Dict[int, Dict[str, str]] = {}

                async for chunk in stream:
                    for choice in chunk.choices:
                        delta = choice.delta

                        # Accumulate normal text
                        if delta.content:
                            assistant_content += delta.content

                        # Accumulate tool‑call argument deltas
                        if delta.tool_calls:
                            for tc in delta.tool_calls:
                                idx = tc.index
                                if idx not in partial_calls:
                                    partial_calls[idx] = {
                                        "id": tc.id,
                                        "name": tc.function.name,
                                        "type": tc.type,  # usually "function"
                                        "arguments": "",
                                    }
                                partial_calls[idx]["arguments"] += tc.function.arguments or ""

                # ──────────────────────────────────────────────
                # CASE 1 → Model issued tool_calls (continue loop)
                # ──────────────────────────────────────────────
                if partial_calls:
                    # Append assistant message that invoked tool(s)
                    messages.append(
                        {
                            "role": "assistant",
                            "content": assistant_content or None,
                            "tool_calls": [
                                {
                                    "type": data["type"],
                                    "id": data["id"],
                                    "function": {
                                        "name": data["name"],
                                        "arguments": data["arguments"],
                                    },
                                }
                                for data in partial_calls.values()
                            ],
                        }
                    )

                    # Execute each tool and feed results back
                    for data in partial_calls.values():
                        try:
                            args = (
                                json.loads(data["arguments"]) if data["arguments"] else {}
                            )
                        except json.JSONDecodeError:
                            print(
                                f"⚠️ Could not parse args for {data['name']}: {data['arguments']}",
                            )
                            args = {}

                        print(f"🔧 Calling {data['name']} → {args}")
                        result = await call_remote_tool_via_mcp(session, data["name"], args)
                        print(f"✅ Result: {result}\n")

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": data["id"],
                                "content": json.dumps(result, default=str),
                            }
                        )

                    # Go back to top of while‑loop for another assistant turn
                    continue

                # ──────────────────────────────────────────────
                # CASE 2 → Model produced final answer (stop)
                # ──────────────────────────────────────────────
                print("\n💬 Assistant:", assistant_content.strip())
                break


# ────────────────────────────────────────────────────────────────────────────
# Entry point: accept goal_prompt as an optional positional CLI argument
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an OpenAI chat loop with optional initial user prompt."
    )
    parser.add_argument(
        "goal_prompt",
        nargs="?",
        default="What's the weather like in Istanbul today?",
        help="Initial user prompt for the conversation",
    )
    args = parser.parse_args()

    asyncio.run(chat_loop(args.goal_prompt))
