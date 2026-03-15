from typing import Annotated, TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def create_agent_graph(tools: list):
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    executor_llm = llm.bind_tools(tools)
    reviewer_llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # --- DEFINÍCIA UZLOV ---
    
    async def executor_node(state: AgentState):
        print("👷 [Executor]: Rozmýšľam nad krokom...")
        response = await executor_llm.ainvoke(state["messages"])
        return {"messages": [response]}

    async def reviewer_node(state: AgentState):
        print("🕵️ [Reviewer]: Kontrolujem výsledok...")
        sys_prompt = SystemMessage(
            content="Si prísny revízor. Skontroluj históriu konverzácie. "
                    "Ak agent úspešne použil nástroj na zmenšenie diagramu a vrátil používateľovi hotový výsledok, "
                    "odpovedz presne a iba jedným slovom: 'SCHVALENE'. "
                    "Ak nástroj nepoužil, alebo je tam chyba, napíš mu spätnú väzbu, čo musí spraviť."
        )
        messages_for_review = [sys_prompt] + state["messages"]
        response = await reviewer_llm.ainvoke(messages_for_review)
        return {"messages": [response]}

    # --- DEFINÍCIA PODMIENOK (ROUTING) ---
    
    def route_from_executor(state: AgentState) -> Literal["tools", "reviewer"]:
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            print(f"🔧 [Smerovanie]: Executor volá nástroj: {last_message.tool_calls[0]['name']}")
            return "tools"
        return "reviewer"

    def route_from_reviewer(state: AgentState) -> Literal["END", "executor"]:
        last_message = state["messages"][-1]
        if "SCHVALENE" in last_message.content.upper():
            print("✅ [Smerovanie]: Práca schválená, končíme!")
            return "END"
        print(f"❌ [Smerovanie]: Práca zamietnutá, vraciam Executorovi na prepracovanie. Odôvodnenie: {last_message.content}")
        return "executor"

    # --- ZOSTAVENIE GRAFU ---
    
    workflow = StateGraph(AgentState)

    workflow.add_node("executor", executor_node)
    workflow.add_node("tools", ToolNode(tools)) 
    workflow.add_node("reviewer", reviewer_node)

    workflow.add_edge(START, "executor")
    workflow.add_conditional_edges("executor", route_from_executor, {"tools": "tools", "reviewer": "reviewer"})
    workflow.add_edge("tools", "executor")
    workflow.add_conditional_edges("reviewer", route_from_reviewer, {"END": END, "executor": "executor"})

    return workflow.compile()