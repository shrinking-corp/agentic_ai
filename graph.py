import os
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


MAX_FINALIZER_RETRIES = 1
MAX_REVIEW_ROUNDS = 3


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    plan: str
    review_notes: str
    final_answer: str
    finalizer_retries: int
    review_rounds: int


class ReviewResult(BaseModel):
    approved: bool = Field(description="Whether the candidate answer is acceptable for the user.")
    feedback: str = Field(
        description="Empty when approved. Otherwise one short sentence describing the most important fix."
    )


def _is_review_approved(review_notes: str) -> bool:
    return review_notes.strip().upper() == "APPROVED"


def _build_executor_prompt(state: AgentState, tools_available: bool) -> str:
    sections = [
        (
            "You are the main assistant in a chat application. Reply directly and helpfully. "
            "Use available tools only when they are genuinely useful, and decide that yourself. "
            "If required information is missing, ask a short clarifying question instead of guessing. "
            "Never claim you used a tool when you did not."
        )
    ]

    if not tools_available:
        sections.append("No external tools are available in this run, so solve the request without them.")

    plan = state.get("plan", "").strip()
    if plan:
        sections.append(f"Internal plan:\n{plan}")

    review_notes = state.get("review_notes", "").strip()
    if review_notes and not _is_review_approved(review_notes):
        sections.append(f"Review feedback to address:\n{review_notes}")

    return "\n\n".join(sections)


def _build_finalizer_prompt(state: AgentState) -> str:
    sections = [
        (
            "You are the finalizer for a user-facing answer. Produce the final answer for the user from the conversation history. "
            "Keep it clear and concise. Do not mention hidden workflow, internal nodes, or orchestration. "
            "If tool results are present in the message history, use them as factual input, but do not talk about tool usage unless the user explicitly asked about it."
        )
    ]

    review_notes = state.get("review_notes", "").strip()
    if review_notes and not _is_review_approved(review_notes):
        sections.append(f"Review feedback to address:\n{review_notes}")

    return "\n\n".join(sections)


def _build_reviewer_prompt(final_answer: str) -> str:
    return (
        "You are a quality reviewer for the candidate final answer. Review it against the latest user request and the conversation history. "
        "If tool results appear in the history, treat them as part of the available evidence, but do not require the answer to mention tools or internal processing. "
        "Be practical, not overly strict. Return your decision using the provided structured schema only. "
        "Set approved=true when the answer is good enough for the user. Otherwise set approved=false and provide one short sentence in feedback describing the most important issue to fix."
        f"\n\nCandidate final answer:\n{final_answer}"
    )


def _decide_reviewer_route(
    review_notes: str,
    finalizer_retries: int,
    review_rounds: int,
) -> tuple[Literal["END", "executor", "finalizer"], str]:
    if _is_review_approved(review_notes):
        return "END", "[Routing] Final answer approved. Workflow finished."

    if finalizer_retries <= MAX_FINALIZER_RETRIES:
        return (
            "finalizer",
            f"[Routing] Reviewer requested a finalizer revision. Reason: {review_notes}",
        )

    if review_rounds >= MAX_REVIEW_ROUNDS:
        return (
            "END",
            "[Routing] Maximum review rounds reached. Returning the latest final answer.",
        )

    return "executor", f"[Routing] Reviewer requested another executor pass. Reason: {review_notes}"


def create_agent_graph(tools: list, log_callback=None):
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def log(message: str) -> None:
        if log_callback:
            log_callback(message)
        else:
            print(message)

    planner_llm = ChatOpenAI(model=model_name, temperature=0)
    base_executor_llm = ChatOpenAI(model=model_name, temperature=0)
    executor_llm = base_executor_llm.bind_tools(tools) if tools else base_executor_llm
    finalizer_llm = ChatOpenAI(model=model_name, temperature=0)
    reviewer_llm = ChatOpenAI(model=model_name, temperature=0).with_structured_output(ReviewResult)

    planner_prompt = (
        "You are an internal planner for a general AI assistant. Analyze the latest user request and the conversation history. "
        "Produce a brief internal plan in exactly three lines: goal, missing information or blockers, and whether tools might help. "
        "This plan is not for the user."
    )

    async def planner_node(state: AgentState):
        log("[Planner] Building an internal plan.")
        response = await planner_llm.ainvoke([SystemMessage(content=planner_prompt)] + state["messages"])
        plan = (response.content or "").strip()
        if plan:
            log(f"[Planner] Plan:\n{plan}")
        return {
            "plan": plan,
            "review_notes": "",
            "final_answer": "",
            "finalizer_retries": 0,
            "review_rounds": 0,
        }

    async def executor_node(state: AgentState):
        log("[Executor] Generating the next assistant step.")
        prompt_content = _build_executor_prompt(state, tools_available=bool(tools))
        response = await executor_llm.ainvoke([SystemMessage(content=prompt_content)] + state["messages"])
        return {"messages": [response], "finalizer_retries": 0}

    async def finalizer_node(state: AgentState):
        log("[Finalizer] Writing the user-facing answer.")
        prompt_content = _build_finalizer_prompt(state)
        response = await finalizer_llm.ainvoke(
            [SystemMessage(content=prompt_content)] + state["messages"]
        )
        final_answer = (response.content or "").strip()
        return {
            "final_answer": final_answer,
            "finalizer_retries": state.get("finalizer_retries", 0) + 1,
        }

    async def reviewer_node(state: AgentState):
        log("[Reviewer] Reviewing the candidate answer.")
        response = await reviewer_llm.ainvoke(
            [SystemMessage(content=_build_reviewer_prompt(state.get("final_answer", "")))] + state["messages"]
        )
        review_notes = "APPROVED" if response.approved else (response.feedback or "The answer needs one more revision.").strip()
        log(f"[Reviewer] Verdict: {review_notes}")
        return {
            "review_notes": review_notes,
            "review_rounds": 0 if _is_review_approved(review_notes) else state.get("review_rounds", 0) + 1,
        }

    def route_from_executor(state: AgentState) -> Literal["tools", "finalizer"]:
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            log(f"[Routing] Executor selected a tool: {last_message.tool_calls[0]['name']}")
            return "tools"
        return "finalizer"

    def route_from_reviewer(state: AgentState) -> Literal["END", "executor", "finalizer"]:
        review_notes = state.get("review_notes", "")
        route, message = _decide_reviewer_route(
            review_notes=review_notes,
            finalizer_retries=state.get("finalizer_retries", 0),
            review_rounds=state.get("review_rounds", 0),
        )
        log(message)
        return route

    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    if tools:
        workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("finalizer", finalizer_node)
    workflow.add_node("reviewer", reviewer_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "executor")
    if tools:
        workflow.add_conditional_edges("executor", route_from_executor, {"tools": "tools", "finalizer": "finalizer"})
        workflow.add_edge("tools", "finalizer")
    else:
        workflow.add_edge("executor", "finalizer")
    workflow.add_edge("finalizer", "reviewer")
    workflow.add_conditional_edges(
        "reviewer",
        route_from_reviewer,
        {"END": END, "executor": "executor", "finalizer": "finalizer"},
    )

    return workflow.compile()