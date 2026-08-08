"""Manual LLM agent that manages coffee-shop inventory via the REST API."""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from openai import BadRequestError

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
LOG_PATH = Path(__file__).resolve().parent / "conversation_log.csv"

SYSTEM_PROMPT = """You are an inventory assistant for Carla's coffee shop supply store.
Help her check stock, register new products, record deliveries and sales, and
report low-stock items using the available tools.

Guidelines:
- Speak in a natural, conversational tone — not like a form.
- Deliveries / received stock: use a positive delta with update_stock.
- Sales / used stock: use a negative delta with update_stock.
- When the user refers to a product by name, call list_inventory first to
  resolve the product name to its id, then call update_stock with that id.
- If a product does not exist and the user wants to add it, use add_product.
- For low-stock questions, use get_low_stock_alerts (default threshold is 10).
- After tools finish, give a clear short summary of what changed or what you found.
- When list_inventory or get_low_stock_alerts returns an ASCII table, include that
  exact table in your reply (copy it as-is). Add a short sentence before or after;
  do not convert it to JSON or a different layout.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_inventory",
            "description": (
                "Return the full product list from inventory. "
                "Use this to look up a product id before updating stock by name."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_product",
            "description": "Register a new product in inventory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Product name",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Initial quantity on hand (non-negative)",
                    },
                    "unit": {
                        "type": "string",
                        "description": "Unit of measure (e.g. units, kg, liters)",
                    },
                },
                "required": ["name", "quantity", "unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_stock",
            "description": (
                "Adjust stock for an existing product by id. "
                "Positive delta = incoming stock; negative delta = outgoing stock. "
                "Resolve product names to ids with list_inventory first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "integer",
                        "description": "Numeric product id from inventory",
                    },
                    "delta": {
                        "type": "integer",
                        "description": "Quantity change to apply",
                    },
                },
                "required": ["product_id", "delta"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_low_stock_alerts",
            "description": (
                "Return products whose quantity is below a threshold "
                "(default 10)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "threshold": {
                        "type": "integer",
                        "description": "Alert when quantity is below this value",
                    },
                },
            },
        },
    },
]


def _ensure_log() -> None:
    if not LOG_PATH.exists():
        with LOG_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["actor", "message", "tool_call", "timestamp"]
            )
            writer.writeheader()


def log_event(actor: str, message: str, tool_call: str = "") -> None:
    """Append one event to conversation_log.csv (never overwrite)."""
    _ensure_log()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["actor", "message", "tool_call", "timestamp"]
        )
        writer.writerow(
            {
                "actor": actor,
                "message": message,
                "tool_call": tool_call,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )


def format_products_table(products: List[Dict[str, Any]]) -> str:
    """Render product dicts as a fixed-width ASCII table for the terminal."""
    headers = ["id", "name", "quantity", "unit"]
    if not products:
        empty = ["(none)", "-", "-", "-"]
        widths = [max(len(h), len(v)) for h, v in zip(headers, empty)]
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
        sep = "-+-".join("-" * w for w in widths)
        row = " | ".join(v.ljust(w) for v, w in zip(empty, widths))
        return f"{header_line}\n{sep}\n{row}"

    rows = [
        [
            str(p.get("id", "")),
            str(p.get("name", "")),
            str(p.get("quantity", "")),
            str(p.get("unit", "")),
        ]
        for p in products
    ]
    widths = [
        max(len(headers[i]), max(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep = "-+-".join("-" * w for w in widths)
    body = "\n".join(
        " | ".join(cell.ljust(w) for cell, w in zip(row, widths)) for row in rows
    )
    return f"{header_line}\n{sep}\n{body}"


def execute_tool(name: str, arguments: Dict[str, Any]) -> str:
    """Call the matching inventory API endpoint and return a result string."""
    try:
        with httpx.Client(base_url=API_BASE_URL, timeout=30.0) as client:
            if name == "list_inventory":
                response = client.get("/inventory")
            elif name == "add_product":
                response = client.post("/inventory", json=arguments)
            elif name == "update_stock":
                product_id = arguments["product_id"]
                response = client.patch(
                    f"/inventory/{product_id}",
                    json={"delta": arguments["delta"]},
                )
            elif name == "get_low_stock_alerts":
                params = {}
                if "threshold" in arguments and arguments["threshold"] is not None:
                    params["threshold"] = arguments["threshold"]
                response = client.get("/inventory/alerts", params=params)
            else:
                return json.dumps({"error": f"Unknown tool: {name}"})

            try:
                body: Any = response.json()
            except Exception:
                body = response.text

            if response.is_error:
                return json.dumps(
                    {
                        "error": True,
                        "status_code": response.status_code,
                        "detail": body,
                    }
                )

            # Show list/alert results as ASCII tables in the terminal and to the LLM
            if name in {"list_inventory", "get_low_stock_alerts"} and isinstance(
                body, list
            ):
                title = (
                    "Inventory"
                    if name == "list_inventory"
                    else "Low-stock alerts"
                )
                table = format_products_table(body)
                display = f"{title}\n{table}"
                print(display)
                print()
                return display

            return json.dumps(body)
    except httpx.RequestError as exc:
        return json.dumps(
            {
                "error": True,
                "detail": (
                    f"Could not reach the inventory API at {API_BASE_URL}. "
                    f"Is uvicorn running? ({exc})"
                ),
            }
        )


def _chat_with_tools(client: OpenAI, messages: List[Dict[str, Any]]):
    """Call the LLM with tools; retry briefly on Groq tool_use_failed glitches."""
    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            return client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
        except BadRequestError as exc:
            last_error = exc
            err_text = str(exc).lower()
            if "tool_use_failed" not in err_text or attempt == 2:
                raise
    raise last_error  # pragma: no cover


def run_agent_turn(
    client: OpenAI,
    messages: List[Dict[str, Any]],
    user_text: str,
) -> str:
    """Observe → Think → Act → Update until the model returns a final reply."""
    messages.append({"role": "user", "content": user_text})
    log_event("user", user_text)

    while True:
        # Think
        completion = _chat_with_tools(client, messages)
        assistant_message = completion.choices[0].message
        tool_calls = assistant_message.tool_calls or []

        assistant_entry: Dict[str, Any] = {
            "role": "assistant",
            "content": assistant_message.content or "",
        }
        if tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ]
        messages.append(assistant_entry)

        # Terminate when the model gives a final answer with no tool calls
        if not tool_calls:
            final_text = (assistant_message.content or "").strip()
            log_event("agent", final_text)
            return final_text

        # Act + Update: run each tool and inject results into history
        for tc in tool_calls:
            fn_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            log_event(
                "tool",
                f"Calling {fn_name} with {json.dumps(args)}",
                tool_call=fn_name,
            )
            result = execute_tool(fn_name, args)
            log_event("tool", result, tool_call=fn_name)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )


def main() -> None:
    if not GROQ_API_KEY:
        print(
            "GROQ_API_KEY is missing. Add it to your .env file and try again.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    print("Coffee shop inventory agent ready. Type 'quit' or 'exit' to stop.")
    print(f"API: {API_BASE_URL} | Model: {GROQ_MODEL}\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_text:
            continue
        if user_text.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break

        try:
            reply = run_agent_turn(client, messages, user_text)
        except Exception as exc:
            print(f"Agent error: {exc}", file=sys.stderr)
            continue

        print(f"Agent: {reply}\n")


if __name__ == "__main__":
    main()
