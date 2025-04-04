from async_lru import alru_cache
from langgraph.graph.graph import CompiledGraph
from .graph import get_compiled_graph
from .enums import Model

@alru_cache(maxsize=1)
async def get_agent() -> CompiledGraph:
    """Get or create an agent instance for the given session ID"""

    agent = await get_compiled_graph(Model.OPENAI)
    return agent

