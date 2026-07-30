#!/usr/bin/env python3
"""
nanocode - A tiny terminal coding agent.

A minimalist coding agent that can read, write, and edit files,
run shell commands, search the web, and manage tasks.
"""

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

import requests
from openai import OpenAI

# Configuration
BASE_URL = "https://openrouter.ai/api/v1"
API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "poolside/laguna-s-2.1:free"
FIRECRAWL_API_KEY = os.environ["FIRECRAWL_API_KEY"]
MAX_WEB_CONTENT_LENGTH = 5000

# Initialize client
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""
    id: str
    name: str
    arguments: str = ""


@dataclass
class AgentConfig:
    """Configuration for the agent."""
    auto_approve: bool = False
    plan_mode: bool = False
    max_iterations: int = 50


class Tool:
    """Base class for all tools."""
    
    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    is_read_only: bool = False

    def execute(self, args: dict[str, Any]) -> str:
        """Execute the tool with given arguments."""
        raise NotImplementedError

    def to_schema(self) -> dict[str, Any]:
        """Convert tool to OpenAI function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ReadFileTool(Tool):
    """Read a file from disk."""
    
    name = "read_file"
    description = "Read a file from disk and return its contents."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read."},
        },
        "required": ["path"],
    }
    is_read_only = True

    def execute(self, args: dict[str, Any]) -> str:
        try:
            with open(args["path"], encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return f"Error: File not found: {args['path']}"
        except PermissionError:
            return f"Error: Permission denied: {args['path']}"
        except Exception as e:
            return f"Error reading file: {e}"


class WriteFileTool(Tool):
    """Write content to a file."""
    
    name = "write_file"
    description = "Write content to a file, creating or overwriting it."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to write."},
            "content": {"type": "string", "description": "Content to write."},
        },
        "required": ["path", "content"],
    }
    is_read_only = False

    def execute(self, args: dict[str, Any]) -> str:
        try:
            with open(args["path"], "w", encoding="utf-8") as f:
                f.write(args["content"])
            return f"Wrote {args['path']}"
        except Exception as e:
            return f"Error writing file: {e}"


class EditFileTool(Tool):
    """Replace an exact string in a file."""
    
    name = "edit_file"
    description = "Replace an exact string in a file with a new string."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to edit."},
            "old_string": {"type": "string", "description": "Exact string to replace."},
            "new_string": {"type": "string", "description": "Replacement string."},
        },
        "required": ["path", "old_string", "new_string"],
    }
    is_read_only = False

    def execute(self, args: dict[str, Any]) -> str:
        try:
            with open(args["path"], encoding="utf-8") as f:
                content = f.read()
            
            if args["old_string"] not in content:
                return f"Error: old_string not found in {args['path']}"
            
            updated_content = content.replace(args["old_string"], args["new_string"])
            with open(args["path"], "w", encoding="utf-8") as f:
                f.write(updated_content)
            return f"Edited {args['path']}"
        except FileNotFoundError:
            return f"Error: File not found: {args['path']}"
        except Exception as e:
            return f"Error editing file: {e}"


class GrepTool(Tool):
    """Search files under a directory for lines matching a regex."""
    
    name = "grep"
    description = "Search files under a directory for lines matching a regex."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex to search for."},
            "path": {"type": "string", "description": "Directory to search in.", "default": "."},
        },
        "required": ["pattern"],
    }
    is_read_only = True

    def execute(self, args: dict[str, Any]) -> str:
        try:
            regex = re.compile(args["pattern"])
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"

        matches = []
        search_path = args.get("path", ".")
        
        for dirpath, _, filenames in os.walk(search_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    with open(filepath, encoding="utf-8") as f:
                        for lineno, line in enumerate(f, 1):
                            if regex.search(line):
                                matches.append(f"{filepath}:{lineno}: {line.rstrip()}")
                except (UnicodeDecodeError, OSError):
                    continue
        
        return "\n".join(matches) if matches else "No matches found."


class BashTool(Tool):
    """Run a shell command."""
    
    name = "bash"
    description = "Run a shell command and return its output."
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to run."},
        },
        "required": ["command"],
    }
    is_read_only = False

    def execute(self, args: dict[str, Any]) -> str:
        try:
            result = subprocess.run(
                args["command"], 
                shell=True, 
                capture_output=True, 
                text=True,
                timeout=30
            )
            return result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 30 seconds"
        except Exception as e:
            return f"Error running command: {e}"


class TodoWriteTool(Tool):
    """Manage task lists."""
    
    name = "todo_write"
    description = "Write the current task list. Replaces the whole list each call."
    parameters = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "status": {"type": "string", "enum": ["pending", "in_progress", "done"]},
                    },
                    "required": ["content", "status"],
                },
            },
        },
        "required": ["items"],
    }
    is_read_only = True

    def __init__(self):
        self.items: list[dict[str, str]] = []

    def execute(self, args: dict[str, Any]) -> str:
        self.items = args["items"]
        marks = {"pending": " ", "in_progress": "~", "done": "x"}
        return "\n".join(
            f"[{marks[item['status']]}] {item['content']}" 
            for item in self.items
        )


class WebFetchTool(Tool):
    """Fetch a URL and return its content."""
    
    name = "web_fetch"
    description = "Fetch a URL and return its content as readable text."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch."},
        },
        "required": ["url"],
    }
    is_read_only = True

    def execute(self, args: dict[str, Any]) -> str:
        try:
            response = requests.post(
                "https://api.firecrawl.dev/v1/scrape",
                headers={"Authorization": f"Bearer {FIRECRAWL_API_KEY}"},
                json={"url": args["url"], "formats": ["markdown"]},
                timeout=30
            )
            response.raise_for_status()
            return response.json()["data"]["markdown"][:MAX_WEB_CONTENT_LENGTH]
        except requests.RequestException as e:
            return f"Error fetching URL: {e}"
        except (KeyError, IndexError) as e:
            return f"Error parsing response: {e}"


class WebSearchTool(Tool):
    """Search the web for information."""
    
    name = "web_search"
    description = "Search the web and return the top results (title, URL, description)."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
        },
        "required": ["query"],
    }
    is_read_only = True

    def execute(self, args: dict[str, Any]) -> str:
        try:
            response = requests.post(
                "https://api.firecrawl.dev/v2/search",
                headers={"Authorization": f"Bearer {FIRECRAWL_API_KEY}"},
                json={"query": args["query"], "limit": 5, "sources": ["web"]},
                timeout=30
            )
            response.raise_for_status()
            results = response.json()["data"]["web"]
            return "\n\n".join(
                f"{r['title']}\n{r['url']}\n{r.get('description', '')}" 
                for r in results
            )
        except requests.RequestException as e:
            return f"Error searching web: {e}"
        except (KeyError, IndexError) as e:
            return f"Error parsing search results: {e}"


class SpawnAgentTool(Tool):
    """Spawn a sub-agent with a fresh context."""
    
    name = "task"
    description = "Spawn a sub-agent with a fresh context to do a task; returns its final answer."
    parameters = {
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "Short description of the task."},
            "prompt": {"type": "string", "description": "Full instructions for the sub-agent."},
        },
        "required": ["description", "prompt"],
    }
    is_read_only = False

    def execute(self, args: dict[str, Any]) -> str:
        sub_tools = [t for t in get_all_tools() if t.name != self.name]
        sub_messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": args["prompt"]},
        ]
        return run_agent(sub_messages, sub_tools, config=AgentConfig(auto_approve=True))


# Global tool instances
_all_tools: Optional[list[Tool]] = None


def get_all_tools() -> list[Tool]:
    """Get all available tools (cached)."""
    global _all_tools
    if _all_tools is None:
        _all_tools = [
            ReadFileTool(),
            WriteFileTool(),
            EditFileTool(),
            GrepTool(),
            BashTool(),
            TodoWriteTool(),
            WebFetchTool(),
            WebSearchTool(),
            SpawnAgentTool(),
        ]
    return _all_tools


def get_system_prompt() -> str:
    """Generate the system prompt with environment info."""
    prompt = (
        "You are nanocode, a terminal coding agent. Be concise. Prefer tools over guessing.\n"
        "Use the todo_write tool to plan any task with more than a couple of steps.\n\n"
        f"Environment:\ncwd: {os.getcwd()}\nos: {os.uname().sysname}\n"
        f"files in cwd: {', '.join(sorted(os.listdir()))}"
    )
    
    if os.path.exists("NANOCODE.md"):
        try:
            with open("NANOCODE.md", encoding="utf-8") as f:
                prompt += f"\n\nProject instructions:\n{f.read()}"
        except Exception:
            pass
    
    return prompt


def parse_tool_calls(stream) -> tuple[str, list[ToolCall], Optional[str]]:
    """Parse streaming response for content and tool calls."""
    reply = ""
    tool_calls: list[ToolCall] = []
    finish_reason = None

    for chunk in stream:
        choice = chunk.choices[0]
        
        # Handle content
        if choice.delta.content:
            print(choice.delta.content, end="", flush=True)
            reply += choice.delta.content
        
        # Handle tool calls
        for tc in choice.delta.tool_calls or []:
            if tc.index >= len(tool_calls):
                tool_calls.append(ToolCall(id="", name=""))
            
            call = tool_calls[tc.index]
            call.id += tc.id or ""
            call.name += tc.function.name or ""
            call.arguments += tc.function.arguments or ""
        
        # Handle finish
        if choice.finish_reason:
            finish_reason = choice.finish_reason
    
    print()  # New line after streaming
    return reply, tool_calls, finish_reason


def get_user_approval(tool: Tool, args: dict[str, Any]) -> bool:
    """Prompt user for approval before executing a write tool."""
    try:
        answer = input(f"{tool.name}({json.dumps(args)}) [y/n] ")
        return answer.strip().lower() == "y"
    except (EOFError, KeyboardInterrupt):
        return False


def run_agent(
    messages: list[dict[str, Any]], 
    tools: list[Tool], 
    config: AgentConfig = None
) -> str:
    """Run the agent loop with the given messages and tools."""
    if config is None:
        config = AgentConfig()
    
    tools_by_name = {t.name: t for t in tools}
    tool_schemas = [t.to_schema() for t in tools]

    for _ in range(config.max_iterations):
        try:
            stream = client.chat.completions.create(
                model=MODEL, 
                messages=messages, 
                tools=tool_schemas, 
                stream=True
            )
        except Exception as e:
            print(f"\nError calling API: {e}", file=sys.stderr)
            return f"Error: {e}"

        reply, tool_calls, finish_reason = parse_tool_calls(stream)

        if finish_reason == "tool_calls" and tool_calls:
            # Add assistant message with tool calls
            assistant_msg = {
                "role": "assistant", 
                "content": reply,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments}
                    }
                    for tc in tool_calls
                ]
            }
            messages.append(assistant_msg)

            # Execute each tool call
            for tc in tool_calls:
                try:
                    args = json.loads(tc.arguments)
                except json.JSONDecodeError as e:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"Error parsing arguments: {e}"
                    })
                    continue

                tool = tools_by_name.get(tc.name)
                if not tool:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"Error: Unknown tool {tc.name}"
                    })
                    continue

                # Handle plan mode restrictions
                if config.plan_mode and not tool.is_read_only:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": (
                            "Plan mode is on: write tools are disabled. "
                            "Present a plan and ask the user to approve it."
                        ),
                    })
                    continue

                # Handle approval for write tools
                if not tool.is_read_only and not config.auto_approve:
                    if not get_user_approval(tool, args):
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": "User denied the tool call."
                        })
                        continue

                # Execute the tool
                try:
                    result = tool.execute(args)
                except Exception as e:
                    result = f"Error executing {tc.name}: {e}"
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })
        else:
            messages.append({"role": "assistant", "content": reply})
            return reply
    
    return reply


def print_banner():
    """Print the application banner."""
    print("┌──────────────────────────────────────────────┐")
    print("│ nanocode — a tiny coding agent               │")
    print("│ /plan toggles plan mode · ctrl-c/ctrl-d quits │")
    print("└──────────────────────────────────────────────┘")


def main():
    """Main entry point for the agent."""
    print_banner()
    
    plan_mode = False
    messages = [{"role": "system", "content": get_system_prompt()}]
    tools = get_all_tools()
    
    while True:
        try:
            prompt = "plan > " if plan_mode else "> "
            user_input = input(prompt)
            
            if user_input.strip() == "/plan":
                plan_mode = not plan_mode
                print(f"Plan mode {'on' if plan_mode else 'off'}")
                continue
            
            if not user_input.strip():
                continue
            
            messages.append({"role": "user", "content": user_input})
            config = AgentConfig(plan_mode=plan_mode)
            run_agent(messages, tools, config=config)
            
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()