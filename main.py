import asyncio
import os
from langchain_core.messages import HumanMessage

from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain_mcp_adapters.tools import load_mcp_tools

from graph import create_agent_graph
from dotenv import load_dotenv

load_dotenv()  

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

async def run_app():
    server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000/sse")
    print(f"🔌 Pripájam sa k lokálnemu HTTP MCP serveru na {server_url}...")

    async with sse_client(server_url) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            print("✅ MCP Server pripojený!")

            tools = await load_mcp_tools(session)
            print(f"🛠️ Načítané nástroje: {[t.name for t in tools]}")

            app = create_agent_graph(tools)

            user_input = "Ahoj, tu je diagram. Zmenši ho prosím cez lokálny tool."
            print(f"\n👤 Používateľ: {user_input}\n")

            async for event in app.astream({"messages": [HumanMessage(content=user_input)]}):
                pass 
            
            print("\n🎉 Hotovo! Finálny stav aplikácie bol dosiahnutý.")

if __name__ == "__main__":
    asyncio.run(run_app())