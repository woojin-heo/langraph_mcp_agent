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
from langgraph.checkpoint.memory import MemorySaver

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

RESPONSE_FORMAT_PROMPT = f"""You are a helpful personal assistant.
Current date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

=== Response Format ===
- Do NOT use markdown formatting
- Use plain text for visual formatting
- Always respond in the same language as the user
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
    Create LangGraph agent with Full Workflow pattern (no ReAct).
    
    Flow:
    START -> classify_intent -> route_by_intent:
        - check_schedule  -> fetch_schedule -> check_locations -> [enrich_travel] -> generate_response -> END
        - create_event    -> extract_event_info -> execute_create_event -> generate_response -> END
        - search_place    -> execute_search_place -> generate_response -> END
        - get_directions  -> execute_directions -> generate_response -> END
        - general         -> generate_response -> END (simple response, no tools)
    """
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    # ============== Intent Classification ==============
    
    async def classify_intent(state: State) -> dict:
        """
        Classify user intent from the last message.
        Intents: check_schedule, create_event, search_place, get_directions, general
        """
        last_msg = state["messages"][-1]
        user_message = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
        
        classification_prompt = f"""Classify the user's intent. Return ONLY one of these words:
- check_schedule: asking about existing events/schedule (e.g., "what's my schedule?", "오늘 뭐해?", "이번주 일정")
- create_event: wants to create/add a new event (e.g., "add meeting tomorrow", "일정 추가해줘")
- search_place: searching for places/restaurants/etc (e.g., "find restaurants near", "맛집 찾아줘")
- get_directions: asking for directions/travel time (e.g., "how to get to", "강남역 가는 법")
- general: other requests, greetings, questions that don't fit above

User message: {user_message}

Intent:"""
        
        response = await llm.ainvoke([HumanMessage(content=classification_prompt)])
        intent_text = response.content.strip().lower()
        
        # Parse intent
        if "check_schedule" in intent_text:
            intent = "check_schedule"
        elif "create_event" in intent_text:
            intent = "create_event"
        elif "search_place" in intent_text:
            intent = "search_place"
        elif "get_directions" in intent_text:
            intent = "get_directions"
        else:
            intent = "general"
        
        return {
            "intent": intent,
            "user_config": USER_CONFIG,
            "events": [],
            "travel_info": [],
        }
    
    # ============== Schedule Workflow ==============
    
    async def fetch_schedule(state: State) -> dict:
        """Fetch calendar events based on user's request."""
        last_msg = state["messages"][-1]
        user_message = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
        
        # Extract period from user message
        period_prompt = f"""Extract the time period from this message. Return ONLY one of:
today, tomorrow, week, next_week, or a date range in format "YYYY-MM-DD to YYYY-MM-DD"

Message: {user_message}
Current date: {datetime.now().strftime("%Y-%m-%d")}

Period:"""
        
        response = await llm.ainvoke([HumanMessage(content=period_prompt)])
        period_text = response.content.strip().lower()
        
        # Call get_events and parse JSON result
        try:
            if " to " in period_text:
                dates = period_text.split(" to ")
                result = await mcp.call_tool("get_events", {
                    "start_date": dates[0].strip(),
                    "end_date": dates[1].strip()
                })
            else:
                period = period_text if period_text in ["today", "tomorrow", "week", "next_week"] else "today"
                result = await mcp.call_tool("get_events", {"period": period})
            
            # Parse JSON result (new structured format)
            data = json.loads(result)
            events = data.get("events", [])
            
        except Exception as e:
            events = []
        
        return {"events": events}
    
    async def enrich_with_travel(state: State) -> dict:
        """Add travel information for events with locations."""
        events = state.get("events", [])
        user_config = state.get("user_config", USER_CONFIG)
        travel_info = []
        
        for event in events:
            location = event.get("location")
            if not location:
                continue
            
            try:
                directions_result = await mcp.call_tool("get_directions", {
                    "origin": user_config["default_location"],
                    "destination": location,
                    "mode": user_config["default_transport"]
                })
                
                duration_minutes = parse_duration_minutes(directions_result)
                event_time = event.get("start_time")
                
                if event_time and duration_minutes and event_time != "All day":
                    departure_time = calculate_departure_time(
                        event_time, duration_minutes, user_config.get("buffer_minutes", 10)
                    )
                else:
                    departure_time = None
                
                travel_info.append({
                    "event_summary": event.get("summary", ""),
                    "event_date": event.get("date", ""),
                    "destination": location,
                    "origin": user_config["default_location"],
                    "duration_minutes": duration_minutes,
                    "duration_text": f"{duration_minutes}분" if duration_minutes else "알 수 없음",
                    "suggested_departure": departure_time,
                    "transport_mode": user_config["default_transport"],
                })
                
            except Exception as e:
                travel_info.append({
                    "event_summary": event.get("summary", ""),
                    "destination": location,
                    "error": str(e)
                })
        
        return {"travel_info": travel_info}
    
    # ============== Create Event Workflow ==============
    
    async def extract_event_info(state: State) -> dict:
        """Extract event details from user message for creation."""
        last_msg = state["messages"][-1]
        user_message = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
        
        extract_prompt = f"""Extract event details from this message. Return in this exact JSON format:
{{"title": "event title", "date": "YYYY-MM-DD", "start_time": "HH:MM", "end_time": "HH:MM", "location": "place or null"}}

If any field is unclear, use reasonable defaults:
- end_time: 1 hour after start_time
- location: null if not specified

Current date: {datetime.now().strftime("%Y-%m-%d")}
Message: {user_message}

JSON:"""
        
        response = await llm.ainvoke([HumanMessage(content=extract_prompt)])
        
        try:
            # Extract JSON from response
            response_text = response.content.strip()
            if "```" in response_text:
                response_text = response_text.split("```")[1].replace("json", "").strip()
            event_info = json.loads(response_text)
        except:
            event_info = {"error": "Could not parse event details"}
        
        return {"events": [event_info]}
    
    async def execute_create_event(state: State) -> dict:
        """Execute event creation with human approval."""
        events = state.get("events", [])
        if not events:
            return {"events": [{"error": "No event info to create"}]}
        
        event_info = events[0]
        if event_info.get("error"):
            return {"events": events}
        
        # Build event parameters
        try:
            date = event_info.get("date", datetime.now().strftime("%Y-%m-%d"))
            start_time = event_info.get("start_time", "09:00")
            end_time = event_info.get("end_time", "10:00")
            
            tool_args = {
                "title": event_info.get("title", "New Event"),
                "start": f"{date}T{start_time}:00",
                "end": f"{date}T{end_time}:00",
                "location": event_info.get("location", "")
            }
            
            # Human approval for create_event
            if use_cli_approval:
                approved, modified_args = get_human_approval("create_event", tool_args)
                if not approved:
                    return {"events": [{"error": "User cancelled event creation"}]}
                tool_args = modified_args
            
            result = await mcp.call_tool("create_event", tool_args)
            event_info["result"] = result
            event_info["success"] = True
            
        except Exception as e:
            event_info["error"] = str(e)
            event_info["success"] = False
        
        return {"events": [event_info]}
    
    # ============== Search Place Workflow ==============
    
    async def execute_search_place(state: State) -> dict:
        """Execute place search."""
        last_msg = state["messages"][-1]
        user_message = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
        user_config = state.get("user_config", USER_CONFIG)
        
        # Extract search query
        extract_prompt = f"""Extract the place search query from this message.
Return JSON: {{"query": "search term", "location": "near location or null"}}

Message: {user_message}
User's default location: {user_config['default_location']}

JSON:"""
        
        response = await llm.ainvoke([HumanMessage(content=extract_prompt)])
        
        try:
            response_text = response.content.strip()
            if "```" in response_text:
                response_text = response_text.split("```")[1].replace("json", "").strip()
            search_info = json.loads(response_text)
        except:
            search_info = {"query": user_message}
        
        # Execute search
        try:
            result = await mcp.call_tool("search_places", {
                "query": search_info.get("query", user_message),
                "location": search_info.get("location", user_config["default_location"])
            })
            search_info["result"] = result
        except Exception as e:
            search_info["error"] = str(e)
        
        return {"events": [search_info]}
    
    # ============== Directions Workflow ==============
    
    async def execute_directions(state: State) -> dict:
        """Execute directions lookup."""
        last_msg = state["messages"][-1]
        user_message = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
        user_config = state.get("user_config", USER_CONFIG)
        
        # Extract origin/destination
        extract_prompt = f"""Extract travel information from this message.
Return JSON: {{"origin": "start point or null", "destination": "end point", "mode": "transit/driving/walking"}}

If origin is not specified, use null (will use default).
Default transport mode: {user_config['default_transport']}

Message: {user_message}

JSON:"""
        
        response = await llm.ainvoke([HumanMessage(content=extract_prompt)])
        
        try:
            response_text = response.content.strip()
            if "```" in response_text:
                response_text = response_text.split("```")[1].replace("json", "").strip()
            travel_params = json.loads(response_text)
        except:
            travel_params = {"destination": user_message}
        
        # Execute directions
        try:
            result = await mcp.call_tool("get_directions", {
                "origin": travel_params.get("origin") or user_config["default_location"],
                "destination": travel_params.get("destination", ""),
                "mode": travel_params.get("mode", user_config["default_transport"])
            })
            travel_params["result"] = result
        except Exception as e:
            travel_params["error"] = str(e)
        
        return {"travel_info": [travel_params]}
    
    # ============== Generate Response (Common) ==============
    
    async def generate_response(state: State) -> dict:
        """Generate final response based on intent and gathered data."""
        intent = state.get("intent", "general")
        events = state.get("events", [])
        travel_info = state.get("travel_info", [])
        user_config = state.get("user_config", USER_CONFIG)
        
        # Build context based on intent
        context_parts = []
        
        if intent == "check_schedule":
            context_parts.append("=== Schedule Information ===")
            if not events:
                context_parts.append("No events found for the requested period.")
            else:
                for event in events:
                    date = event.get("date", "")
                    day = event.get("day_of_week", "")
                    time = event.get("start_time", "")
                    summary = event.get("summary", "Untitled")
                    location = event.get("location", "")
                    
                    event_str = f"- {date} ({day}) {time}: {summary}"
                    if location:
                        event_str += f" @ {location}"
                    context_parts.append(event_str)
            
            if travel_info:
                context_parts.append("\n=== Travel Information ===")
                for ti in travel_info:
                    if ti.get("error"):
                        context_parts.append(f"- {ti.get('destination', 'Unknown')}: Could not calculate")
                    else:
                        line = f"- To {ti['destination']}: {ti.get('duration_text', '?')}"
                        if ti.get('suggested_departure'):
                            line += f" (leave by {ti['suggested_departure']})"
                        context_parts.append(line)
        
        elif intent == "create_event":
            context_parts.append("=== Event Creation ===")
            if events and events[0].get("success"):
                e = events[0]
                context_parts.append(f"Successfully created: {e.get('title', 'Event')}")
                context_parts.append(f"Date: {e.get('date', '')} {e.get('start_time', '')} - {e.get('end_time', '')}")
                if e.get("location"):
                    context_parts.append(f"Location: {e['location']}")
            elif events and events[0].get("error"):
                context_parts.append(f"Failed to create event: {events[0]['error']}")
            else:
                context_parts.append("Event creation status unknown")
        
        elif intent == "search_place":
            context_parts.append("=== Search Results ===")
            if events and events[0].get("result"):
                context_parts.append(events[0]["result"])
            elif events and events[0].get("error"):
                context_parts.append(f"Search failed: {events[0]['error']}")
        
        elif intent == "get_directions":
            context_parts.append("=== Directions ===")
            if travel_info and travel_info[0].get("result"):
                context_parts.append(travel_info[0]["result"])
            elif travel_info and travel_info[0].get("error"):
                context_parts.append(f"Could not get directions: {travel_info[0]['error']}")
        
        else:  # general
            context_parts.append("(No specific data - respond naturally to the user's message)")
        
        context = "\n".join(context_parts)
        
        response_prompt = f"""{RESPONSE_FORMAT_PROMPT}

Based on this information, respond to the user naturally in their language.

{context}

User's default location: {user_config['default_location']}"""
        
        msgs = [SystemMessage(content=response_prompt)] + state["messages"]
        response = await llm.ainvoke(msgs)
        
        return {"messages": [response]}
    
    # ============== Routers ==============
    
    def route_by_intent(state: State) -> Literal[
        "fetch_schedule", "extract_event_info", "execute_search_place", 
        "execute_directions", "generate_response"
    ]:
        """Route to appropriate workflow based on intent."""
        intent = state.get("intent", "general")
        routes = {
            "check_schedule": "fetch_schedule",
            "create_event": "extract_event_info",
            "search_place": "execute_search_place",
            "get_directions": "execute_directions",
            "general": "generate_response",
        }
        return routes.get(intent, "generate_response")
    
    def check_locations(state: State) -> Literal["enrich_with_travel", "generate_response"]:
        """Check if any events have locations that need travel info."""
        events = state.get("events", [])
        has_location = any(e.get("location") for e in events)
        return "enrich_with_travel" if has_location else "generate_response"
    
    # ============== Build Graph ==============
    
    g = StateGraph(State)
    
    # Add all nodes
    g.add_node("classify_intent", classify_intent)
    g.add_node("fetch_schedule", fetch_schedule)
    g.add_node("enrich_with_travel", enrich_with_travel)
    g.add_node("extract_event_info", extract_event_info)
    g.add_node("execute_create_event", execute_create_event)
    g.add_node("execute_search_place", execute_search_place)
    g.add_node("execute_directions", execute_directions)
    g.add_node("generate_response", generate_response)
    
    # Entry point
    g.add_edge(START, "classify_intent")
    g.add_conditional_edges("classify_intent", route_by_intent)
    
    # Schedule workflow
    g.add_conditional_edges("fetch_schedule", check_locations)
    g.add_edge("enrich_with_travel", "generate_response")
    
    # Create event workflow
    g.add_edge("extract_event_info", "execute_create_event")
    g.add_edge("execute_create_event", "generate_response")
    
    # Search/Directions workflows go directly to response
    g.add_edge("execute_search_place", "generate_response")
    g.add_edge("execute_directions", "generate_response")
    
    # All paths end at generate_response
    g.add_edge("generate_response", "__end__")
    
    return g


# ============== Helper Functions ==============

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
    print(f"🔒 Approval required for: {TOOLS_REQUIRING_APPROVAL}")
    
    # Create graph with memory checkpointer for state tracking
    memory = MemorySaver()
    graph_builder = create_graph(mcp)
    graph = graph_builder.compile(checkpointer=memory)
    messages = []
    
    # Config with thread_id for checkpointer
    config = {"configurable": {"thread_id": "main-session"}}
    
    try:
        while True:
            user = input("You: ").strip()
            if user.lower() in ['quit', 'exit', 'q']:
                break
            if not user:
                continue
            
            messages.append(HumanMessage(content=user))
            result = await graph.ainvoke({"messages": messages}, config=config)
            messages = result["messages"]
            
            print(f"\nAssistant: {messages[-1].content}\n")
    finally:
        await mcp.disconnect_all()
        print("👋 Bye")


if __name__ == "__main__":
    asyncio.run(main())
