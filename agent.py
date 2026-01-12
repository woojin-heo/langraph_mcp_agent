"""
LangGraph + FastMCP Multi-Server Agent with Human-in-the-Loop
"""
import asyncio
import json
from typing import Annotated, TypedDict, Any, Optional, Literal
import re
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
- Use plain text for visual formatting
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


# ============== User Configuration ==============

USER_CONFIG = {
    "default_location": "redhill mrt station",           # Default origin for travel calculations
    "default_transport": "transit",         # transit, driving, walking, bicycling
    "buffer_minutes": 10,                   # Extra time buffer before appointments
}


# ============== LangGraph Agent ==============

class State(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str                             # User intent: check_schedule, create_event, general
    events: list[dict]                      # Fetched calendar events
    travel_info: list[dict]                 # Travel information for events with locations
    user_config: dict                       # User preferences


def create_graph(mcp: MultiMCPClient, use_cli_approval: bool = True):
    """
    Create LangGraph agent with intent-based workflow.
    
    Flow:
    START -> classify_intent -> route_by_intent:
        - check_schedule -> fetch_schedule -> check_locations:
            - has_location -> enrich_with_travel -> generate_response -> END
            - no_location -> generate_response -> END
        - create_event/general -> chat -> router:
            - has_tool_calls -> tools -> chat
            - no_tool_calls -> END
    """
    llm = ChatOpenAI(model="gpt-4o-mini")
    llm_with_tools = llm.bind_tools(mcp.tools)
    
    # ============== Intent Classification ==============
    
    async def classify_intent(state: State) -> dict:
        """
        Classify user intent from the last message.
        Returns intent: check_schedule, create_event, or general
        """
        last_msg = state["messages"][-1]
        user_message = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
        
        # Use LLM for intent classification
        classification_prompt = f"""Classify the user's intent. Return ONLY one of these words:
- check_schedule: asking about existing events/schedule (e.g., "what's my schedule?", "오늘 뭐해?", "이번주 일정")
- create_event: wants to create/add a new event (e.g., "add meeting", "일정 추가해줘")
- general: other requests (directions, place search, general questions)

User message: {user_message}

Intent:"""
        
        response = await llm.ainvoke([HumanMessage(content=classification_prompt)])
        intent_text = response.content.strip().lower()
        
        # Parse the intent from response
        if "check_schedule" in intent_text:
            intent = "check_schedule"
        elif "create_event" in intent_text:
            intent = "create_event"
        else:
            intent = "general"
        
        return {
            "intent": intent,
            "user_config": USER_CONFIG,
            "events": [],
            "travel_info": []
        }
    
    # ============== Schedule Workflow ==============
    
    async def fetch_schedule(state: State) -> dict:
        """
        Fetch calendar events based on user's request.
        Parses the time period from user message and calls get_events.
        """
        last_msg = state["messages"][-1]
        user_message = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
        
        # Determine period from user message
        period_prompt = f"""Extract the time period from this message. Return ONLY one of:
today, tomorrow, week, next_week, or a date range in format "YYYY-MM-DD to YYYY-MM-DD"

Message: {user_message}
Current date: {datetime.now().strftime("%Y-%m-%d")}

Period:"""
        
        response = await llm.ainvoke([HumanMessage(content=period_prompt)])
        period_text = response.content.strip().lower()
        
        # Call get_events tool
        try:
            if "to" in period_text:
                # Date range format
                dates = period_text.split("to")
                start_date = dates[0].strip()
                end_date = dates[1].strip()
                result = await mcp.call_tool("get_events", {
                    "start_date": start_date,
                    "end_date": end_date
                })
            else:
                # Period shortcut
                period = period_text if period_text in ["today", "tomorrow", "week", "next_week"] else "today"
                result = await mcp.call_tool("get_events", {"period": period})
            
            # Parse events from result
            events = parse_events_from_result(result)
            
        except Exception as e:
            events = []
            result = f"Error fetching events: {e}"
        
        return {"events": events}
    
    async def enrich_with_travel(state: State) -> dict:
        """
        Add travel information for events that have locations.
        Calculates travel time from user's default location.
        """
        events = state.get("events", [])
        user_config = state.get("user_config", USER_CONFIG)
        travel_info = []
        
        for event in events:
            location = event.get("location")
            if not location:
                continue
            
            try:
                # Get directions from default location to event location
                directions_result = await mcp.call_tool("get_directions", {
                    "origin": user_config["default_location"],
                    "destination": location,
                    "mode": user_config["default_transport"]
                })
                
                # Parse travel duration
                duration_minutes = parse_duration_minutes(directions_result)
                
                # Calculate suggested departure time
                event_time = event.get("start_time")
                if event_time and duration_minutes:
                    departure_time = calculate_departure_time(
                        event_time,
                        duration_minutes,
                        user_config.get("buffer_minutes", 10)
                    )
                else:
                    departure_time = None
                
                travel_info.append({
                    "event_summary": event.get("summary", ""),
                    "destination": location,
                    "origin": user_config["default_location"],
                    "duration_minutes": duration_minutes,
                    "duration_text": f"{duration_minutes}분" if duration_minutes else "알 수 없음",
                    "suggested_departure": departure_time,
                    "transport_mode": user_config["default_transport"],
                    "raw_directions": directions_result
                })
                
            except Exception as e:
                travel_info.append({
                    "event_summary": event.get("summary", ""),
                    "destination": location,
                    "error": str(e)
                })
        
        return {"travel_info": travel_info}
    
    async def generate_response(state: State) -> dict:
        """
        Generate final response with schedule and travel information.
        """
        events = state.get("events", [])
        travel_info = state.get("travel_info", [])
        user_config = state.get("user_config", USER_CONFIG)
        
        # Build context for response generation
        context_parts = ["Schedule information:"]
        
        if not events:
            context_parts.append("No events found for the requested period.")
        else:
            for event in events:
                event_str = f"- {event.get('summary', 'Untitled')}"
                if event.get('start_time'):
                    event_str += f" at {event['start_time']}"
                if event.get('location'):
                    event_str += f" @ {event['location']}"
                context_parts.append(event_str)
        
        if travel_info:
            context_parts.append("\nTravel information:")
            for ti in travel_info:
                if ti.get("error"):
                    context_parts.append(f"- {ti['destination']}: Could not calculate travel info")
                else:
                    travel_str = f"- To {ti['destination']}: {ti['duration_text']} by {ti['transport_mode']}"
                    if ti.get('suggested_departure'):
                        travel_str += f", leave by {ti['suggested_departure']}"
                    context_parts.append(travel_str)
        
        context = "\n".join(context_parts)
        
        response_prompt = f"""{SYSTEM_PROMPT}

Based on this information, respond to the user naturally in their language.
Include travel time and suggested departure time if available.

{context}

User's default location: {user_config['default_location']}
User's default transport: {user_config['default_transport']}"""
        
        msgs = [SystemMessage(content=response_prompt)] + state["messages"]
        response = await llm.ainvoke(msgs)
        
        return {"messages": [response]}
    
    # ============== General Chat (ReAct style) ==============
    
    async def chat(state: State) -> dict:
        """Standard chat node for general queries and tool usage."""
        msgs = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in msgs):
            msgs = [SystemMessage(content=SYSTEM_PROMPT)] + msgs
        return {"messages": [await llm_with_tools.ainvoke(msgs)]}
    
    async def tools(state: State) -> dict:
        """Execute tool calls from the LLM."""
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
    
    # ============== Routers ==============
    
    def route_by_intent(state: State) -> Literal["fetch_schedule", "chat"]:
        """Route based on classified intent."""
        intent = state.get("intent", "general")
        if intent == "check_schedule":
            return "fetch_schedule"
        # For create_event and general, use standard chat flow
        return "chat"
    
    def check_locations(state: State) -> Literal["enrich_with_travel", "generate_response"]:
        """Check if any events have locations that need travel info."""
        events = state.get("events", [])
        has_location = any(e.get("location") for e in events)
        return "enrich_with_travel" if has_location else "generate_response"
    
    def chat_router(state: State) -> Literal["tools", "__end__"]:
        """Route after chat: to tools if tool calls exist, else end."""
        last = state["messages"][-1]
        return "tools" if getattr(last, 'tool_calls', None) else "__end__"
    
    # ============== Build Graph ==============
    
    g = StateGraph(State)
    
    # Add nodes
    g.add_node("classify_intent", classify_intent)
    g.add_node("fetch_schedule", fetch_schedule)
    g.add_node("enrich_with_travel", enrich_with_travel)
    g.add_node("generate_response", generate_response)
    g.add_node("chat", chat)
    g.add_node("tools", tools)
    
    # Add edges
    g.add_edge(START, "classify_intent")
    g.add_conditional_edges("classify_intent", route_by_intent)
    
    # Schedule workflow edges
    g.add_conditional_edges("fetch_schedule", check_locations)
    g.add_edge("enrich_with_travel", "generate_response")
    g.add_edge("generate_response", "__end__")
    
    # Chat workflow edges (ReAct loop)
    g.add_conditional_edges("chat", chat_router)
    g.add_edge("tools", "chat")
    
    return g.compile()


# ============== Helper Functions ==============

def parse_events_from_result(result: str) -> list[dict]:
    """
    Parse calendar events from the get_events tool result string.
    Returns list of event dicts with summary, start_time, location, etc.
    """
    events = []
    
    if not result or "No events" in result or "Error" in result:
        return events
    
    # Parse the formatted output from gcalendar.py
    # Expected format: "• HH:MM - Summary\n  Location: ...\n"
    lines = result.split('\n')
    current_event = {}
    
    for line in lines:
        line = line.strip()
        
        # New event starts with bullet point and time
        if line.startswith('•') or line.startswith('-'):
            if current_event:
                events.append(current_event)
            
            # Parse time and summary: "• 14:00 - Meeting"
            match = re.match(r'[•\-]\s*(\d{1,2}:\d{2})?\s*[-–]?\s*(.+)', line)
            if match:
                time_str = match.group(1)
                summary = match.group(2).strip()
                current_event = {
                    "summary": summary,
                    "start_time": time_str,
                    "location": None
                }
            else:
                current_event = {"summary": line.lstrip('•- '), "start_time": None, "location": None}
        
        # Location line
        elif 'Location:' in line or '장소:' in line:
            if current_event:
                loc = line.split(':', 1)[-1].strip()
                if loc and loc.lower() != 'none' and loc != '없음':
                    current_event["location"] = loc
        
        # Alternative location format: "  📍 Location name"
        elif '📍' in line or line.startswith('  ') and current_event and not current_event.get("location"):
            loc = line.replace('📍', '').strip()
            if loc and not loc.startswith(('http', 'www')):
                current_event["location"] = loc
    
    # Don't forget the last event
    if current_event:
        events.append(current_event)
    
    return events


def parse_duration_minutes(directions_result: str) -> Optional[int]:
    """
    Parse travel duration in minutes from get_directions result.
    Handles formats like "Duration: 45 mins", "소요시간: 1시간 30분", etc.
    """
    if not directions_result:
        return None
    
    # Try to find duration patterns
    patterns = [
        r'Duration:\s*(\d+)\s*min',           # "Duration: 45 mins"
        r'Duration:\s*(\d+)\s*hour.*?(\d+)?\s*min',  # "Duration: 1 hour 30 mins"
        r'(\d+)\s*분',                          # "45분"
        r'(\d+)\s*시간\s*(\d+)?\s*분?',          # "1시간 30분" or "1시간"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, directions_result, re.IGNORECASE)
        if match:
            groups = match.groups()
            if len(groups) >= 2 and groups[1]:
                # Hours and minutes
                hours = int(groups[0])
                minutes = int(groups[1]) if groups[1] else 0
                return hours * 60 + minutes
            else:
                # Just minutes or just hours
                value = int(groups[0])
                if 'hour' in pattern or '시간' in pattern:
                    return value * 60
                return value
    
    return None


def calculate_departure_time(event_time: str, duration_minutes: int, buffer_minutes: int = 10) -> Optional[str]:
    """
    Calculate suggested departure time.
    
    Args:
        event_time: Event start time in "HH:MM" format
        duration_minutes: Travel duration in minutes
        buffer_minutes: Extra buffer time
    
    Returns:
        Suggested departure time in "HH:MM" format
    """
    if not event_time or not duration_minutes:
        return None
    
    try:
        # Parse event time
        if ':' in event_time:
            hour, minute = map(int, event_time.split(':'))
        else:
            return None
        
        # Calculate total minutes to subtract
        total_minutes = duration_minutes + buffer_minutes
        
        # Convert to minutes from midnight, subtract, convert back
        event_minutes = hour * 60 + minute
        departure_minutes = event_minutes - total_minutes
        
        # Handle negative (previous day)
        if departure_minutes < 0:
            departure_minutes += 24 * 60
        
        dep_hour = departure_minutes // 60
        dep_minute = departure_minutes % 60
        
        return f"{dep_hour:02d}:{dep_minute:02d}"
        
    except Exception:
        return None


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
