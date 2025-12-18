# 🤖 Multi-Service Agent

A modular Google services agent built with **LangGraph** and **FastMCP**.

## Architecture

```
                         ┌─────────────────┐
                    ┌───▶│ Calendar Server │───▶ Google Calendar
                    │    └─────────────────┘
┌──────────────┐    │    ┌─────────────────┐
│   LangGraph  │────┼───▶│  Gmail Server   │───▶ Gmail API
│    Agent     │  MCP    └─────────────────┘
└──────────────┘    │    ┌─────────────────┐
                    └───▶│  Maps Server    │───▶ Google Maps
                         └─────────────────┘
```

## Available Servers

| Server | Status | Tools |
|--------|--------|-------|
| 📅 Calendar | ✅ Ready | `get_events`, `create_event` |
| 📧 Gmail | 🚧 Template | `search_emails`, `send_email`, `get_unread_emails` |
| 🗺️ Maps | 🚧 Template | `search_places`, `get_directions`, `get_place_details` |

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Google Calendar Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable **Google Calendar API**
3. Create OAuth 2.0 credentials (Desktop App)
4. Download as `credentials.json`

### 3. Environment

```bash
export OPENAI_API_KEY=your-api-key
```

### 4. Run

```bash
python agent.py
```

## Enable More Servers

Edit `agent.py` and uncomment the servers you want:

```python
SERVERS = {
    "calendar": "servers/calendar.py",
    "gmail": "servers/gmail.py",      # Uncomment to enable
    "maps": "servers/maps.py",        # Uncomment to enable
}
```

## Project Structure

```
agent/
├── agent.py                 # Multi-server LangGraph agent
├── servers/
│   ├── calendar.py          # Google Calendar (ready)
│   ├── gmail.py             # Gmail (template)
│   └── maps.py              # Google Maps (template)
├── requirements.txt
├── credentials.json         # Google OAuth (you create)
└── token.json               # Auto-generated
```

## Adding a New Server

1. Create `servers/your_service.py`:

```python
from fastmcp import FastMCP

mcp = FastMCP("YourService")

@mcp.tool()
def your_tool(param: str) -> str:
    """Tool description."""
    return "result"

if __name__ == "__main__":
    mcp.run()
```

2. Register in `agent.py`:

```python
SERVERS = {
    ...
    "your_service": "servers/your_service.py",
}
```

3. Done! The agent will auto-discover tools from the new server.

## Usage Example

```
🤖 Multi-Service Agent

Connecting to servers...
  ✅ calendar: ['get_events', 'create_event']

📦 Total tools: 2

You: What's on my calendar this week?
Assistant: 📅 7-day schedule:
• 2024-12-14T10:00: Team meeting
• 2024-12-15T14:00: Project review

You: q
👋 Bye
```
