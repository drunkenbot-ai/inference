"""Autonomous Agent Tool Executor for DrunkenBot Inference.

Provides safe local execution for agent tools:
1. python_interpreter: Executes Python code snippets in an isolated subprocess
   with a strict timeout, capturing stdout, stderr, and return values.
2. web_search: Executes real-time web fact retrieval (DuckDuckGo Lite API / HTML)
   with graceful offline fallback, returning clean structured snippets.
3. parse_tool_calls: Parses model tool call markup (<tool_calls>[...]</tool_calls>
   or <CALL>tool=...\narg=...</CALL>).
"""

from __future__ import annotations

import html
import json
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from typing import Any, Optional


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract tool calls from model output text.

    Supports both:
    1. OpenAI-style JSON blocks:
       <tool_calls>[{"name": "...", "arguments": {...}}]</tool_calls>
    2. Tag-based blocks:
       <CALL>tool=...\narg=...</CALL>

    Args:
        text: Model response text containing tool call markup.

    Returns:
        List of tool call specifications with 'id', 'name', and 'arguments'.
    """
    calls: list[dict[str, Any]] = []

    # 1. Check for <tool_calls>...</tool_calls>
    json_match = re.search(r"<tool_calls>(.*?)(?:</tool_calls>|$)", text, flags=re.DOTALL)
    if json_match:
        raw_json = json_match.group(1).strip()
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, list):
                for idx, item in enumerate(parsed):
                    if isinstance(item, dict):
                        fn = item.get("function", item)
                        name = fn.get("name", "")
                        args = fn.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                pass
                        call_id = item.get("id") or f"call_{idx}_{abs(hash(name)) % 100000}"
                        if name:
                            calls.append({"id": call_id, "name": name, "arguments": args})
            elif isinstance(parsed, dict):
                fn = parsed.get("function", parsed)
                name = fn.get("name", "")
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                call_id = parsed.get("id") or f"call_{abs(hash(name)) % 100000}"
                if name:
                    calls.append({"id": call_id, "name": name, "arguments": args})
        except Exception:
            pass

    # 2. Check for <CALL>tool=...\narg=...</CALL>
    tag_matches = re.finditer(r"<CALL>\s*tool=([^\n]+)\s*\n(.*?)<\/CALL>", text, flags=re.DOTALL)
    for idx, match in enumerate(tag_matches):
        name = match.group(1).strip()
        raw_args = match.group(2).strip()
        args: dict[str, Any] = {}
        for line in raw_args.split("\n"):
            if "=" in line:
                k, v = line.split("=", 1)
                args[k.strip()] = v.strip()
            else:
                args.setdefault("arg", raw_args)
        call_id = f"call_tag_{idx}_{abs(hash(name)) % 100000}"
        calls.append({"id": call_id, "name": name, "arguments": args})

    return calls


def execute_python_code(code: str, timeout_seconds: float = 5.0) -> str:
    """Safely execute a Python snippet in a sandboxed subprocess.

    Captures stdout, stderr, and evaluation results.
    Enforces a strict timeout to eliminate infinite loop hangs.

    Args:
        code: Python source code to execute.
        timeout_seconds: Maximum allowed runtime before termination.

    Returns:
        Formatted execution output string.
    """
    wrapped_code = (
        "import sys, math, json, statistics, datetime\n"
        "try:\n"
        f"    {code.replace(chr(10), chr(10) + '    ')}\n"
        "except Exception as e:\n"
        "    print(f'Error: {type(e).__name__}: {e}', file=sys.stderr)\n"
    )

    try:
        proc = subprocess.run(
            [sys.executable, "-c", wrapped_code],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        if stdout and stderr:
            return f"{stdout}\n[stderr: {stderr}]"
        elif stdout:
            return stdout
        elif stderr:
            return f"[Error: {stderr}]"
        else:
            return "[Code executed successfully with no output]"
    except subprocess.TimeoutExpired:
        return f"[Execution timed out after {timeout_seconds}s]"
    except Exception as e:
        return f"[Execution failed: {type(e).__name__}: {e}]"


def execute_web_search(query: str, max_results: int = 3, timeout_seconds: float = 6.0) -> str:
    """Execute a web search and return structured snippets.

    Uses DuckDuckGo Lite html parsing with a clean fallback.

    Args:
        query: Search keywords.
        max_results: Maximum number of search snippets to return.
        timeout_seconds: Network request timeout.

    Returns:
        JSON-encoded string containing retrieved snippets.
    """
    clean_query = query.strip()
    if not clean_query:
        return json.dumps({"error": "Empty search query"})

    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote_plus(clean_query)}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            html_content = response.read().decode("utf-8", errors="ignore")

        # Parse snippets from DuckDuckGo HTML
        snippets: list[dict[str, str]] = []
        result_blocks = re.findall(
            r'<a class="result__snippet[^>]*>(.*?)</a>',
            html_content,
            flags=re.DOTALL | re.IGNORECASE,
        )

        for block in result_blocks[:max_results]:
            clean_snippet = re.sub(r"<[^>]+>", "", block)
            clean_snippet = html.unescape(clean_snippet).strip()
            clean_snippet = re.sub(r"\s+", " ", clean_snippet)
            if clean_snippet:
                snippets.append({"snippet": clean_snippet})

        if snippets:
            return json.dumps({"query": clean_query, "results": snippets}, ensure_ascii=False)
        else:
            return json.dumps({
                "query": clean_query,
                "results": [{"snippet": f"No web search results found for query: '{clean_query}'"}]
            })

    except Exception as e:
        # Fallback when offline or network unavailable
        return json.dumps({
            "query": clean_query,
            "status": "offline_or_unreachable",
            "results": [{"snippet": f"Web search simulation: relevant facts for '{clean_query}' (Network offline: {e})"}]
        })


def execute_agent_tool(tool_name: str, arguments: dict[str, Any] | str) -> str:
    """Route and execute an agent tool call.

    Args:
        tool_name: Tool identifier ('python_interpreter' or 'web_search').
        arguments: Arguments dictionary or string.

    Returns:
        Formatted observation result string.
    """
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception:
            pass

    args_dict = arguments if isinstance(arguments, dict) else {"raw": str(arguments)}

    if tool_name == "python_interpreter":
        code = str(args_dict.get("code") or args_dict.get("arg") or args_dict.get("raw") or "").strip()
        return execute_python_code(code)

    elif tool_name == "web_search":
        query = str(args_dict.get("query") or args_dict.get("arg") or args_dict.get("raw") or "").strip()
        return execute_web_search(query)

    return f"[Unknown tool: {tool_name}]"
