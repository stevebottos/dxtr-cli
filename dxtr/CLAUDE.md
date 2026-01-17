# LiteLLM + LangGraph Agent System

## Overview

Multi-agent system with a main chat agent that can delegate to specialist agents. Uses LiteLLM proxy for LLM routing and LangGraph for orchestration.

## Architecture

```
User → FastAPI (server.py) → Main Chat Agent → Response
                                    │
                                    ├── (direct response for simple queries)
                                    │
                                    └── delegate_to_agent(agent, task)
                                              │
                                    ┌─────────┴─────────┐
                                    ▼                   ▼
                            Addition Agent      Subtraction Agent
                              (add tool)         (subtract tool)
```

**Key distinction:**
- **Tools** = Pure functions (add, subtract, search, etc.)
- **Agents** = LLM-powered entities with their own tools and reasoning loops
- **Delegation** = Main agent hands off to specialist agents (not tools pretending to be agents)

## Key Files

- **server.py** - FastAPI server hosting the LangGraph multi-agent system
  - `POST /chat` - Main endpoint (user_id, session_id, query)
  - `GET /health` - Health check
  - Main agent can respond directly or delegate to specialists

- **mock_conversation.py** - HTTP client for testing the server

- **Makefile** - Commands to run services
  - `make dev` - Start both LiteLLM proxy and agent server
  - `make stop` - Stop all services
  - `make litellm` - Start just LiteLLM proxy (port 4000)
  - `make agent-server` - Start just agent server (port 8000)
  - `make mock-conversation` - Run test conversation

## Dependencies

- LiteLLM proxy must be running on `localhost:4000`
- Key packages: `langchain`, `langchain-openai`, `langgraph`, `fastapi`, `uvicorn`, `httpx`

## Running

```bash
# Start everything
make dev

# In another terminal, test it
make mock-conversation

# Stop everything
make stop
```

## Adding New Specialist Agents

1. Create the agent's tools:
```python
@tool
def my_tool(arg: str) -> str:
    """Tool description."""
    return do_something(arg)
```

2. Create the agent using the factory:
```python
my_agent = create_specialist_agent(
    llm=my_llm,
    tools=[my_tool],
    system_prompt="You are a specialist in X...",
)
```

3. Add to the graph in `build_graph()`:
```python
workflow.add_node("my_agent", my_agent)
workflow.add_conditional_edges("my_agent", route_next, {"main": "main", END: END})
```

4. Update the main agent's system prompt and `delegate_to_agent` tool to know about the new agent.

## Important: pydantic_ai + LangGraph Incompatibility

**Do NOT use pydantic_ai agents inside LangGraph nodes.** There's an intermittent deadlock caused by conflicting anyio task groups between the two libraries.

What works:
- pydantic_ai alone (no LangGraph)
- LangGraph + langchain_openai (current setup)
- LangGraph + raw httpx calls

What deadlocks:
- LangGraph nodes calling pydantic_ai agents

## LLM Configuration

Models are configured via LiteLLM proxy. Current setup uses:
- `main` - Main chat model (used by main agent and specialists)

Virtual model names are mapped in `litellm_config.yaml`.

## Conversation Memory

**Currently disabled** for testing. The `MemorySaver` checkpointer needs proper handling to avoid state bleeding between queries.

For production:
- Re-enable checkpointer with proper message handling
- Consider how delegation affects conversation history
- Use persistent checkpointer (PostgreSQL, Redis, etc.)

## Future Work

- Re-enable conversation memory with proper state handling
- Add more specialist agents (code, research, etc.)
- Streaming responses
- Agent-to-agent communication (not just main → specialist)
- Parallel tool execution within agents
