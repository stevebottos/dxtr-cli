import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage

from agents import main_agent


# =============================================================================
# MEMORY (simple session-based message history)
# =============================================================================

# TODO: Try mem0 for long-term semantic memory across sessions
# https://docs.mem0.ai - extracts facts, does semantic search, persists user context

# For now: simple in-memory session store (swap for Redis in production)
_sessions: dict[str, list[ModelMessage]] = {}


# =============================================================================
# REQUEST HANDLING
# =============================================================================


def get_session_key(user_id: str, session_id: str) -> str:
    return f"{user_id}:{session_id}"


async def handle_query(query: str, user_id: str, session_id: str) -> tuple[str, list[str]]:
    """
    Process a query through the main agent with conversation history.
    Returns (answer, list of delegation tools called).
    """
    session_key = get_session_key(user_id, session_id)

    # Get existing conversation history for this session
    history = _sessions.get(session_key, [])

    # Run agent with message history
    result = await main_agent.run(query, message_history=history)

    # Store updated history (all messages including new ones)
    _sessions[session_key] = result.all_messages()

    # Extract which tools were called (only from this turn, not history)
    tools_called = []
    for msg in result.new_messages():
        if hasattr(msg, "parts"):
            for part in msg.parts:
                if hasattr(part, "tool_name"):
                    tools_called.append(part.tool_name)

    return result.output, tools_called


# =============================================================================
# FASTAPI SERVER
# =============================================================================


class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    query: str


class ChatResponse(BaseModel):
    answer: str
    delegated_to: str | None


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Multi-agent system ready")
    yield
    print("Shutting down")


api = FastAPI(title="Multi-Agent Server", lifespan=lifespan)


@api.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    answer, tools_called = await handle_query(request.query, request.user_id, request.session_id)

    # Map delegation tools to agent names
    delegated_to = None
    for tool in tools_called:
        if tool.startswith("delegate_to_"):
            delegated_to = tool.replace("delegate_to_", "") + "_agent"
            break

    return ChatResponse(answer=answer, delegated_to=delegated_to)


@api.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("server:api", host="0.0.0.0", port=8000, reload=True)
