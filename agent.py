"""Math agent that solves questions using tools in a ReAct loop."""

import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from calculator import calculate

load_dotenv()

# Qwen via DashScope OpenAI-compatible API
MODEL = OpenAIChatModel(
    "qwen-plus",
    provider=OpenAIProvider(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.environ["DASHSCOPE_API_KEY"],
    ),
)

agent = Agent(
    MODEL,
    system_prompt=(
        "You are a helpful assistant. Solve each question step by step. "
        "Use the calculator tool for arithmetic. "
        "Use the product_lookup tool when a question mentions products from the catalog. "
        "If a question cannot be answered with the information given, say so."
    ),
)


@agent.tool_plain
def calculator_tool(expression: str) -> str:
    """Evaluate a math expression and return the result.

    Examples: "847 * 293", "10000 * (1.07 ** 5)", "23 % 4"
    """
    return calculate(expression)


@agent.tool_plain
def product_lookup(product_name: str) -> str:
    """Look up the price of a product by name.
    Use this when a question asks about product prices from the catalog.
    """
    with open("products.json") as f:
        products = json.load(f)
    if product_name in products:
        return str(products[product_name])
    return f"Product '{product_name}' not found. Available products: {list(products.keys())}"


def load_questions(path: str = "math_questions.md") -> list[str]:
    """Load numbered questions from the markdown file."""
    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and line[0].isdigit() and ". " in line[:4]:
                questions.append(line.split(". ", 1)[1])
    return questions


def run_with_retry(question: str, max_retries: int = 5):
    """Run the agent, retrying on rate-limit or connection errors."""
    for attempt in range(max_retries):
        try:
            return agent.run_sync(question)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                wait = 60
                print(f"  [Rate limit hit, waiting {wait}s before retry...]\n")
                time.sleep(wait)
            elif "Connection error" in msg or "getaddrinfo" in msg or "ConnectError" in msg:
                wait = 10
                print(f"  [Network error, waiting {wait}s before retry...]\n")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed after {max_retries} retries")


def main():
    questions = load_questions()
    for i, question in enumerate(questions, 1):
        print(f"## Question {i}")
        print(f"> {question}\n")

        result = run_with_retry(question)

        print("### Trace")
        for message in result.all_messages():
            for part in message.parts:
                kind = part.part_kind
                if kind in ("user-prompt", "system-prompt"):
                    continue
                elif kind == "text":
                    print(f"- **Reason:** {part.content}")
                elif kind == "tool-call":
                    print(f"- **Act:** `{part.tool_name}({part.args})`")
                elif kind == "tool-return":
                    print(f"- **Result:** `{part.content}`")

        print(f"\n**Answer:** {result.output}\n")
        print("---\n")
        time.sleep(60)


if __name__ == "__main__":
    main()
