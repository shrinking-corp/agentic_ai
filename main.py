import asyncio
import os
import sys
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import Callable, Sequence

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from graph import create_agent_graph
from gui import launch_gui

load_dotenv()


BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR.parent
DEFAULT_SERVER_SCRIPT = WORKSPACE_DIR / "mcp_server" / "server.py"
DEFAULT_MCP_TRANSPORT = "stdio"


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def build_server_parameters() -> StdioServerParameters:
    server_script = resolve_path(os.getenv("MCP_SERVER_SCRIPT", str(DEFAULT_SERVER_SCRIPT)))

    return StdioServerParameters(
        command=sys.executable,
        args=[str(server_script)],
        env=os.environ.copy(),
    )


def get_mcp_transport() -> str:
    return os.getenv("MCP_TRANSPORT", DEFAULT_MCP_TRANSPORT).strip().lower()


def get_mcp_server_url() -> str:
    return os.getenv("MCP_SERVER_URL", "").strip()


def build_conversation_messages(
    conversation_history: Sequence[dict[str, str]] | None,
    user_prompt: str,
):
    messages = []

    for item in conversation_history or []:
        role = (item.get("role") or "").strip().lower()
        content = (item.get("content") or "").strip()
        if not content:
            continue

        if role == "assistant":
            messages.append(AIMessage(content=content))
        elif role == "system":
            messages.append(SystemMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))

    prompt = user_prompt.strip()
    if not prompt:
        raise ValueError("The user message is empty.")

    messages.append(HumanMessage(content=prompt))
    return messages


def default_log_callback(message: str) -> None:
    print(message)


def _format_exception_message(exc: BaseException) -> str:
    if hasattr(exc, "exceptions"):
        nested_messages = []
        for nested_exc in exc.exceptions:
            nested_message = _format_exception_message(nested_exc)
            if nested_message:
                nested_messages.append(nested_message)
        if nested_messages:
            return " | ".join(nested_messages)

    message = str(exc).strip()
    if message:
        return message
    return exc.__class__.__name__


async def _run_graph(app, input_messages, log, metadata_callback=None) -> str:
    final_result = ""
    tool_metadata = {
        "tool_used": False,
        "tool_name": "",
        "algorithm": "",
    }

    async for event in app.astream(
        {
            "messages": input_messages,
            "plan": "",
            "review_notes": "",
            "final_answer": "",
            "finalizer_retries": 0,
            "review_rounds": 0,
        }
    ):
        for node_name, node_output in event.items():
            if node_name == "finalizer" and node_output.get("final_answer"):
                final_result = node_output["final_answer"]
                log(f"[Finalizer] {final_result}")

            for message in node_output.get("messages", []):
                tool_calls = getattr(message, "tool_calls", None)
                if not tool_calls:
                    continue

                tool_call = tool_calls[0]
                tool_args = tool_call.get("args", {})
                tool_metadata["tool_used"] = True
                tool_metadata["tool_name"] = tool_call["name"]
                tool_metadata["algorithm"] = tool_args.get("algorithm", "")
                log(f"[{node_name}] tool call: {tool_call['name']} -> {tool_args}")

    if metadata_callback:
        metadata_callback(tool_metadata)
    return final_result


class AgentRuntime:
    def __init__(self) -> None:
        self.transport = get_mcp_transport()
        self.server_url = get_mcp_server_url()
        self.server_params = build_server_parameters() if self.transport == "stdio" else None
        self.server_script_path = Path(self.server_params.args[0]) if self.server_params else None

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._startup_lock = threading.Lock()
        self._initialize_future: Future | None = None
        self._transport_context = None
        self._session_context = None
        self._session = None
        self._tools = []
        self._tool_names: list[str] = []
        self._mcp_available = False
        self._mcp_error = ""
        self._status_logged = False

    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def start(self) -> None:
        with self._startup_lock:
            if self._loop is not None:
                return

            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._run_loop, name="agent-runtime", daemon=True)
            self._thread.start()
            self._initialize_future = asyncio.run_coroutine_threadsafe(self._initialize_mcp(), self._loop)

    def get_status(self) -> dict:
        return {
            "mcp_available": self._mcp_available,
            "tool_names": list(self._tool_names),
            "error": self._mcp_error,
            "transport": self.transport,
            "server_script": str(self.server_script_path) if self.server_script_path else "",
            "server_script_exists": self.server_script_path.exists() if self.server_script_path else False,
            "server_url": self.server_url,
        }

    def initialize(self, log_callback: Callable[[str], None] | None = None) -> dict:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set in the .env file or the environment.")

        log = log_callback or default_log_callback
        self.start()

        if self._initialize_future is not None:
            self._initialize_future.result()

        status = self.get_status()
        if not self._status_logged:
            log(f"Using OpenAI model: {os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}")
            log(f"MCP transport: {self.transport}")
            if self.transport == "stdio" and self.server_script_path is not None:
                log(f"Starting local MCP server: {self.server_script_path}")
                log(f"MCP server script exists: {self.server_script_path.exists()}")
            elif self.server_url:
                log(f"Connecting to MCP server URL: {self.server_url}")

            if status["mcp_available"]:
                log(f"MCP server is connected via {self.transport}.")
                log(f"Loaded MCP tools: {status['tool_names']}")
            else:
                log(
                    "[Warning] The MCP server or tools are not available. "
                    f"Continuing without tools. Reason: {status['error']}"
                )
            self._status_logged = True
        return status

    async def _initialize_mcp(self) -> None:
        try:
            if self.transport == "stdio":
                assert self.server_params is not None
                self._transport_context = stdio_client(self.server_params)
            elif self.transport == "sse":
                if not self.server_url:
                    raise RuntimeError("MCP_SERVER_URL must be set for the SSE transport.")
                self._transport_context = sse_client(self.server_url)
            elif self.transport in {"streamable-http", "streamable_http", "http"}:
                if not self.server_url:
                    raise RuntimeError(
                        "MCP_SERVER_URL must be set for the streamable HTTP transport."
                    )
                self._transport_context = streamablehttp_client(self.server_url)
            else:
                raise RuntimeError(
                    "Unsupported MCP transport. Use stdio, sse, or streamable-http."
                )

            streams = await self._transport_context.__aenter__()
            self._session_context = ClientSession(streams[0], streams[1])
            self._session = await self._session_context.__aenter__()
            await self._session.initialize()

            self._tools = await load_mcp_tools(self._session)
            self._tool_names = [tool.name for tool in self._tools]
            self._mcp_available = True
            self._mcp_error = ""
        except Exception as exc:
            self._tools = []
            self._tool_names = []
            self._mcp_available = False
            self._mcp_error = _format_exception_message(exc)
            await self._close_mcp_handles()

    async def _close_mcp_handles(self) -> None:
        if self._session_context is not None:
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception:
                pass
            self._session_context = None
            self._session = None

        if self._transport_context is not None:
            try:
                await self._transport_context.__aexit__(None, None, None)
            except Exception:
                pass
            self._transport_context = None

    async def _run_chat_async(
        self,
        input_messages,
        log: Callable[[str], None],
        metadata_callback: Callable[[dict], None] | None,
    ) -> str:
        tools = list(self._tools) if self._mcp_available else []
        app = create_agent_graph(tools, log_callback=log)
        return await _run_graph(app, input_messages, log, metadata_callback)

    def run_chat(
        self,
        user_prompt: str,
        conversation_history: Sequence[dict[str, str]] | None = None,
        log_callback: Callable[[str], None] | None = None,
        metadata_callback: Callable[[dict], None] | None = None,
    ) -> str:
        log = log_callback or default_log_callback
        self.initialize(log)
        input_messages = build_conversation_messages(conversation_history, user_prompt)

        log("Sending the message to the agent.")

        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(
            self._run_chat_async(input_messages, log, metadata_callback),
            self._loop,
        )
        final_result = future.result()
        if self._mcp_available:
            log("Application finished.")
        else:
            log("Application finished without MCP tools.")
        return final_result

    async def _shutdown_async(self) -> None:
        await self._close_mcp_handles()

    def shutdown(self) -> None:
        if self._loop is None:
            return

        try:
            future = asyncio.run_coroutine_threadsafe(self._shutdown_async(), self._loop)
            future.result(timeout=5)
        except Exception:
            pass

        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)

        self._loop = None
        self._thread = None
        self._initialize_future = None
        self._tools = []
        self._tool_names = []
        self._mcp_available = False
        self._status_logged = False


if __name__ == "__main__":
    launch_gui(AgentRuntime())