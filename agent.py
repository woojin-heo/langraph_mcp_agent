"""
LangGraph + FastMCP Multi-Server Agent with Human-in-the-Loop
"""
import asyncio
import json
from typing import Annotated, TypedDict, Any, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, create_model
from datetime import datetime

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

load_dotenv()


# ============== Configuration ==============

SERVERS = {
    "calendar": "servers/gcalendar.py",
    # "gmail": "servers/gmail.py",      # Uncomment to enable
    "maps": "servers/maps.py",
}

# Tools that require human approval (data-changing operations)
TOOLS_REQUIRING_APPROVAL = ["create_event", "delete_event", "update_event"]

SYSTEM_PROMPT = f"""You are a helpful personal assistant with access to Google Calendar and Maps.

Current date and time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

=== Response Format ===
- Do NOT use markdown formatting (no **, __, `, ```, etc.)
- Use plain text with emojis for visual formatting
- Use line breaks and indentation for structure
- Always respond in the same language as the user

=== Calendar Tool Usage ===
When user asks for events, use get_events with:

1. Quick shortcuts (period parameter):
   - "today", "tomorrow", "week", "next_week", "last_week"
   - Example: get_events(period="week") for this week

2. Custom date ranges (start_date + end_date):
   - For months: start_date="2025-12-01", end_date="2025-12-31"
   - For specific periods: Calculate the exact dates from user's request
   - Example: "이번 달" (this month) → calculate first and last day of current month
   - Example: "다음 달" (next month) → calculate first and last day of next month
   - Example: "내년 1월" → start_date="2026-01-01", end_date="2026-01-31"
"""


# ============== Helper: JSON Schema to Pydantic ==============

def json_schema_to_pydantic(name: str, schema: dict) -> type[BaseModel]:
    """Convert JSON Schema to Pydantic model"""
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    
    fields = {}
    for field_name, field_schema in properties.items():
        field_type = _get_python_type(field_schema)
        description = field_schema.get("description", "")
        default = ... if field_name in required else field_schema.get("default", None)
        
        fields[field_name] = (field_type, Field(default=default, description=description))
    
    if not fields:
        fields["_placeholder"] = (Optional[str], Field(default=None))
    
    return create_model(name, **fields)


def _get_python_type(schema: dict) -> type:
    """Map JSON Schema type to Python type"""
    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    json_type = schema.get("type", "string")
    return type_map.get(json_type, str)


# ============== Human-in-the-Loop ==============

def get_human_approval(tool_name: str, args: dict) -> tuple[bool, dict]:
    """
    Request human approval for tool execution
    Returns: (approved, modified_args)
    """
    print("\n" + "="*50)
    print(f"Check required: {tool_name}")
    print("="*50)
    print("\nParameters to execute:")
    
    for key, value in args.items():
        print(f"   {key}: {value}")
    
    print("\nOptions:")
    print("   [Enter] Approve and execute")
    print("   [e] Modify parameters")
    print("   [n] Cancel")
    
    choice = input("\n선택: ").strip().lower()
    
    if choice == 'n':
        return False, args
    elif choice == 'e':
        # Modify parameters mode
        modified_args = args.copy()
        print("\n📝 Modify parameters (Enter if no changes)")
        
        for key, value in args.items():
            new_value = input(f"   {key} [{value}]: ").strip()
            if new_value:
                # Try to keep type
                if isinstance(value, int):
                    try:
                        modified_args[key] = int(new_value)
                    except:
                        modified_args[key] = new_value
                else:
                    modified_args[key] = new_value
        
        print("\n✏️ Modified parameters:")
        for key, value in modified_args.items():
            print(f"   {key}: {value}")
        
        confirm = input("\n이대로 실행할까요? [Y/n]: ").strip().lower()
        if confirm == 'n':
            return False, modified_args
        return True, modified_args
    else:
        # Enter or other input = approve
        return True, args


# ============== MCP Client ==============

class MCPConnection:
    """Single MCP server connection"""
    
    def __init__(self, name: str, script: str):
        self.name = name
        self.script = script
        self._client = None
        self._session = None
    
    async def connect(self):
        import sys
        # Use the same Python interpreter that's running this script
        params = StdioServerParameters(command=sys.executable, args=[self.script])
        self._client = stdio_client(params)
        read, write = await self._client.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()
    
    async def list_tools(self):
        return await self._session.list_tools()
    
    async def call_tool(self, name: str, args: dict) -> str:
        result = await self._session.call_tool(name, args)
        return result.content[0].text if result.content else ""
    
    async def disconnect(self):
        if self._session:
            await self._session.__aexit__(None, None, None)
        if self._client:
            await self._client.__aexit__(None, None, None)


class MultiMCPClient:
    """Manages multiple MCP server connections"""
    
    def __init__(self):
        self.connections: dict[str, MCPConnection] = {}
        self.tools: list[StructuredTool] = []
        self._tool_to_server: dict[str, str] = {}
    
    async def connect_all(self, servers: dict[str, str]):
        """Connect to all configured servers"""
        for name, script in servers.items():
            conn = MCPConnection(name, script)
            try:
                await conn.connect()
                self.connections[name] = conn
                
                # Get tools from this server
                resp = await conn.list_tools()
                for t in resp.tools:
                    self._tool_to_server[t.name] = name
                    
                    # Create Pydantic model from MCP tool schema
                    schema = t.inputSchema if hasattr(t, 'inputSchema') else {"properties": {}}
                    args_model = json_schema_to_pydantic(f"{t.name}Args", schema)
                    
                    # Create async function for this tool
                    async def _call(tn=t.name, **kwargs) -> str:
                        return await self.call_tool(tn, kwargs)
                    
                    self.tools.append(StructuredTool(
                        name=t.name,
                        description=t.description or "",
                        args_schema=args_model,
                        coroutine=_call,
                        func=lambda **kw: None,  # Dummy sync func
                    ))
                
                print(f"  ✅ {name}: {[t.name for t in resp.tools]}")
            except Exception as e:
                print(f"  ❌ {name}: {e}")
    
    async def call_tool(self, name: str, args: dict) -> str:
        server_name = self._tool_to_server.get(name)
        if not server_name:
            return f"Tool not found: {name}"
        return await self.connections[server_name].call_tool(name, args)
    
    async def disconnect_all(self):
        for conn in self.connections.values():
            await conn.disconnect()


# ============== LangGraph Agent ==============

class State(TypedDict):
    messages: Annotated[list, add_messages]


def create_graph(mcp: MultiMCPClient, use_cli_approval: bool = True):
    """
    Create LangGraph agent.
    
    Args:
        mcp: MultiMCPClient instance
        use_cli_approval: If True, use CLI-based human approval (for terminal).
                         If False, skip approval in graph (for Telegram/external approval).
    """
    llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(mcp.tools)
    
    async def chat(state: State):
        msgs = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in msgs):
            msgs = [SystemMessage(content=SYSTEM_PROMPT)] + msgs
        return {"messages": [await llm.ainvoke(msgs)]}
    
    async def tools(state: State):
        last = state["messages"][-1]
        results = []
        
        for tc in last.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            
            # Human-in-the-Loop: CLI approval (only if enabled)
            if use_cli_approval and tool_name in TOOLS_REQUIRING_APPROVAL:
                approved, modified_args = get_human_approval(tool_name, tool_args)
                
                if not approved:
                    results.append(ToolMessage(
                        content="❌ User cancelled the operation.",
                        tool_call_id=tc["id"]
                    ))
                    continue
                
                tool_args = modified_args
            
            # Execute tool
            r = await mcp.call_tool(tool_name, tool_args)
            results.append(ToolMessage(content=r, tool_call_id=tc["id"]))
        
        return {"messages": results}
    
    def router(state: State):
        last = state["messages"][-1]
        return "tools" if getattr(last, 'tool_calls', None) else "__end__"
    
    g = StateGraph(State)
    g.add_node("chat", chat)
    g.add_node("tools", tools)
    g.add_edge(START, "chat")
    g.add_conditional_edges("chat", router)
    g.add_edge("tools", "chat")
    
    return g.compile()


# ============== Main ==============

async def main():
    print("🤖 Multi-Service Agent (with Human-in-the-Loop)\n")
    print("Connecting to servers...")
    
    mcp = MultiMCPClient()
    await mcp.connect_all(SERVERS)
    
    if not mcp.tools:
        print("\nNo tools available. Check server configurations.")
        return
    
    print(f"\nTotal tools: {len(mcp.tools)}")
    print(f"🔒 Approval required for: {TOOLS_REQUIRING_APPROVAL}\n")
    
    graph = create_graph(mcp)
    messages = []
    
    try:
        while True:
            user = input("You: ").strip()
            if user.lower() in ['quit', 'exit', 'q']:
                break
            if not user:
                continue
            
            messages.append(HumanMessage(content=user))
            result = await graph.ainvoke({"messages": messages})
            messages = result["messages"]
            
            print(f"\nAssistant: {messages[-1].content}\n")
    finally:
        await mcp.disconnect_all()
        print("👋 Bye")


if __name__ == "__main__":
    asyncio.run(main())
