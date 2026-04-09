# Agentic AI Chat Client for MCP Servers

This application is a general-purpose chat client built on top of OpenAI and LangGraph. It connects to an MCP server in the background, loads the available tools, and lets the agent decide whether they are useful for a given reply.

The MCP server currently exposes the `shrink_diagram` tool, but the client is no longer specialized only for PlantUML. The tool is available to the model, not mandatory.

## Workflow Architecture

The LangGraph workflow contains these nodes:

- `planner`: internally analyzes the latest request and prepares a short plan
- `executor`: decides the next step and optionally calls an available MCP tool
- `tools`: executes the MCP tool call chosen by the executor
- `finalizer`: writes the final user-facing answer without internal details
- `reviewer`: checks whether the final answer is coherent and good enough

The two newer logical nodes compared with the earlier version are `planner` and `finalizer`.

## Setup

1. Make sure `uv` is installed.

2. From the `agentic_ai` directory, sync the local environment from `pyproject.toml` and `uv.lock`:

```powershell
uv sync
```

This creates or updates the local `.venv` and installs the client dependencies from `pyproject.toml` and `uv.lock`.

3. If you want to use a local MCP server over `stdio`, make sure the same environment can also run the server side, because the client starts [mcp_server/server.py](../mcp_server/server.py) with the current Python interpreter. That means the same environment must also contain the server dependencies.

You can keep `requirements.txt` as a legacy compatibility file, but the source of truth for the client is now `pyproject.toml` together with `uv.lock`.

4. Configure the `.env` file with at least:

```env
OPENAI_API_KEY="your-openai-api-key"
OPENAI_MODEL="gpt-4o-mini"
```

You can also set:

```env
MCP_TRANSPORT="stdio"
MCP_SERVER_SCRIPT="../mcp_server/server.py"
MCP_SERVER_URL=""
```

Supported transports:

- `stdio`: the client starts a local Python MCP server script
- `sse`: the client connects to a remote MCP server through an SSE endpoint
- `streamable-http`: the client connects to a remote MCP server through a streamable HTTP endpoint

Configuration examples:

Local server over `stdio`:

```env
MCP_TRANSPORT="stdio"
MCP_SERVER_SCRIPT="../mcp_server/server.py"
MCP_SERVER_URL=""
```

Remote server over `sse`:

```env
MCP_TRANSPORT="sse"
MCP_SERVER_URL="http://localhost:8000/sse"
```

Remote server over `streamable-http`:

```env
MCP_TRANSPORT="streamable-http"
MCP_SERVER_URL="http://localhost:8000/mcp"
```

## Running the Application

Start the application from the `agentic_ai` directory:

```powershell
uv run python main.py
```

If `stdio` is selected, the client starts a local MCP server as a child process. If `sse` or `streamable-http` is selected, the client connects to the URL set in `MCP_SERVER_URL`. In both cases, MCP is initialized once at startup and then reused for subsequent messages. After that, the graphical chat interface opens.

## GUI Features

- conversation history between the user and the assistant
- a separate input area for a new message
- logs from the workflow nodes
- MCP connection status at startup
- an indicator showing whether an MCP tool was used for the current answer
- the ability to clear the full chat

## Manual Server Test

If you want to start the local server manually for `stdio`, run it in an environment where its dependencies are installed:

```powershell
uv run python ..\mcp_server\server.py
```

This mode waits for an MCP client over standard input/output, so it is mainly useful for diagnostics rather than direct terminal usage.