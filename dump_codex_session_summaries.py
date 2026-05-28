#!/usr/bin/env python3
"""
Dump human user prompts and Codex replies for each session JSONL.

By default this skips subagent sessions such as approval-review logs, because
they mostly duplicate the parent transcript. Use --include-subagents to include
them too.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ENV_RE = re.compile(r"<environment_context>.*?</environment_context>", re.DOTALL)
AGENTS_RE = re.compile(
    r"# AGENTS\.md instructions for .*?<INSTRUCTIONS>.*?</INSTRUCTIONS>",
    re.DOTALL,
)
APPROVAL_RE = re.compile(
    r"The following is the Codex agent history.*?>>> APPROVAL REQUEST END\n?",
    re.DOTALL,
)


@dataclass
class TurnDump:
    user: str
    assistant_messages: list[str] = field(default_factory=list)


@dataclass
class SessionDump:
    path: Path
    session_id: str | None = None
    timestamp: str | None = None
    cwd: str | None = None
    branch: str | None = None
    repository_url: str | None = None
    cli_version: str | None = None
    is_subagent: bool = False
    prefill_tokens: int | None = None
    decode_tokens: int | None = None
    cached_prefill_tokens: int | None = None
    reasoning_decode_tokens: int | None = None
    total_tokens: int | None = None
    turns: list[TurnDump] = field(default_factory=list)


def iter_session_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("*.jsonl"))


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if text is None:
            text = item.get("input_text")
        if text is None:
            text = item.get("output_text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def clean_user_text(text: str) -> str:
    text = ENV_RE.sub("", text)
    text = AGENTS_RE.sub("", text)
    text = APPROVAL_RE.sub("", text)
    return text.strip()


def is_context_only(text: str) -> bool:
    cleaned = clean_user_text(text)
    if not cleaned:
        return True
    if cleaned.startswith("The following is the Codex agent history"):
        return True
    if cleaned.startswith("The following is the Codex agent history added"):
        return True
    return False


def parse_session(path: Path, root: Path) -> SessionDump:
    dump = SessionDump(path=path.relative_to(root))

    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"warning: {path}:{line_no}: invalid json: {exc}", file=sys.stderr)
                continue

            payload = record.get("payload")
            if record.get("type") is None and isinstance(record.get("id"), str):
                dump.session_id = record.get("id")
                dump.timestamp = record.get("timestamp")
                git = record.get("git")
                if isinstance(git, dict):
                    dump.branch = git.get("branch")
                    dump.repository_url = git.get("repository_url")
                continue
            if payload is None:
                payload = record
            if not isinstance(payload, dict):
                continue

            if record.get("type") == "event_msg" and payload.get("type") == "token_count":
                info = payload.get("info")
                if not isinstance(info, dict):
                    continue
                usage = info.get("total_token_usage")
                if not isinstance(usage, dict):
                    continue
                dump.prefill_tokens = usage.get("input_tokens")
                dump.decode_tokens = usage.get("output_tokens")
                dump.cached_prefill_tokens = usage.get("cached_input_tokens")
                dump.reasoning_decode_tokens = usage.get("reasoning_output_tokens")
                dump.total_tokens = usage.get("total_tokens")
                continue

            if record.get("type") == "session_meta":
                dump.session_id = payload.get("id")
                dump.timestamp = payload.get("timestamp") or record.get("timestamp")
                dump.cwd = payload.get("cwd")
                git = payload.get("git")
                if isinstance(git, dict):
                    dump.branch = git.get("branch")
                    dump.repository_url = git.get("repository_url")
                dump.cli_version = payload.get("cli_version")
                dump.is_subagent = payload.get("thread_source") == "subagent" or bool(
                    payload.get("source", {}).get("subagent")
                    if isinstance(payload.get("source"), dict)
                    else False
                )
                continue

            if record.get("type") == "response_item" and payload.get("type") == "message":
                message = payload
            elif record.get("type") == "message":
                message = record
            else:
                continue

            role = message.get("role")
            text = content_to_text(message.get("content")).strip()
            if not text:
                continue

            if role == "user":
                if not is_context_only(text):
                    dump.turns.append(TurnDump(user=clean_user_text(text)))
            elif role == "assistant" and dump.turns:
                dump.turns[-1].assistant_messages.append(text)

    return dump


def render_markdown(dumps: list[SessionDump]) -> str:
    chunks: list[str] = []
    for dump in dumps:
        title = dump.session_id or dump.path.stem
        chunks.append(f"## {title}")
        chunks.append(f"- file: `{dump.path}`")
        if dump.timestamp:
            chunks.append(f"- timestamp: `{dump.timestamp}`")
        if dump.cwd:
            chunks.append(f"- cwd: `{dump.cwd}`")
        if dump.branch:
            chunks.append(f"- branch: `{dump.branch}`")
        if dump.repository_url:
            chunks.append(f"- repository_url: `{dump.repository_url}`")
        if dump.is_subagent:
            chunks.append("- subagent: `true`")
        if dump.prefill_tokens is not None or dump.decode_tokens is not None:
            chunks.append(f"- prefill_tokens: `{dump.prefill_tokens}`")
            chunks.append(f"- decode_tokens: `{dump.decode_tokens}`")
            if dump.cached_prefill_tokens is not None:
                chunks.append(f"- cached_prefill_tokens: `{dump.cached_prefill_tokens}`")
            if dump.reasoning_decode_tokens is not None:
                chunks.append(f"- reasoning_decode_tokens: `{dump.reasoning_decode_tokens}`")
            if dump.total_tokens is not None:
                chunks.append(f"- total_tokens: `{dump.total_tokens}`")

        chunks.append("\n### turns")
        if dump.turns:
            for turn_idx, turn in enumerate(dump.turns, 1):
                chunks.append(f"\n#### prompt {turn_idx}\n\n{turn.user}")
                if turn.assistant_messages:
                    for reply_idx, reply in enumerate(turn.assistant_messages, 1):
                        label = "reply" if len(turn.assistant_messages) == 1 else f"reply {reply_idx}"
                        chunks.append(f"\n#### codex {label}\n\n{reply}")
                else:
                    chunks.append("\n#### codex reply\n\n(no assistant reply)")
        else:
            chunks.append("\n(no prompts)")
        chunks.append("")
    return "\n".join(chunks)


def render_jsonl(dumps: list[SessionDump]) -> str:
    lines: list[str] = []
    for dump in dumps:
        lines.append(
            json.dumps(
                {
                    "file": str(dump.path),
                    "session_id": dump.session_id,
                    "timestamp": dump.timestamp,
                    "cwd": dump.cwd,
                    "branch": dump.branch,
                    "repository_url": dump.repository_url,
                    "is_subagent": dump.is_subagent,
                    "token_usage": {
                        "prefill_tokens": dump.prefill_tokens,
                        "decode_tokens": dump.decode_tokens,
                        "cached_prefill_tokens": dump.cached_prefill_tokens,
                        "reasoning_decode_tokens": dump.reasoning_decode_tokens,
                        "total_tokens": dump.total_tokens,
                    },
                    "turns": [
                        {
                            "user": turn.user,
                            "assistant_messages": turn.assistant_messages,
                        }
                        for turn in dump.turns
                    ],
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)


def render_totals(dumps: list[SessionDump]) -> str:
    totals = {
        "sessions": len(dumps),
        "prefill_tokens": 0,
        "decode_tokens": 0,
        "cached_prefill_tokens": 0,
        "reasoning_decode_tokens": 0,
        "total_tokens": 0,
    }
    for dump in dumps:
        totals["prefill_tokens"] += dump.prefill_tokens or 0
        totals["decode_tokens"] += dump.decode_tokens or 0
        totals["cached_prefill_tokens"] += dump.cached_prefill_tokens or 0
        totals["reasoning_decode_tokens"] += dump.reasoning_decode_tokens or 0
        totals["total_tokens"] += dump.total_tokens or 0
    return json.dumps(totals, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        type=Path,
        help="sessions directory root, default: current directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="write output to this file instead of stdout",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "jsonl"),
        default="markdown",
        help="output format, default: markdown",
    )
    parser.add_argument(
        "--include-subagents",
        action="store_true",
        help="include subagent sessions such as approval-review logs",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="include sessions with no extracted prompts",
    )
    parser.add_argument(
        "--totals-only",
        action="store_true",
        help="print only aggregate token totals for selected sessions as JSON",
    )
    parser.add_argument(
        "--branch",
        help="include only sessions whose session_meta git.branch matches this value",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    dumps = [parse_session(path, root) for path in iter_session_files(root)]
    if args.branch:
        dumps = [dump for dump in dumps if dump.branch == args.branch]
    if not args.include_subagents:
        dumps = [dump for dump in dumps if not dump.is_subagent]
    if not args.include_empty:
        dumps = [dump for dump in dumps if dump.turns]

    if args.totals_only:
        output = render_totals(dumps)
    elif args.format == "jsonl":
        output = render_jsonl(dumps)
    else:
        output = render_markdown(dumps)

    if args.output:
        args.output.write_text(output + ("\n" if output else ""), encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
