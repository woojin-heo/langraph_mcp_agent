"""
FastMCP Google Calendar server
Supports multi-user OAuth tokens via user_id
"""
import os
import sys
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv
from fastmcp import FastMCP

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from user_token_manager import token_manager

load_dotenv()

mcp = FastMCP("Calendar")

SCOPES = ['https://www.googleapis.com/auth/calendar']

CREDENTIALS_PATH = os.getenv('GOOGLE_CREDENTIALS_PATH', 'gcalendar_credentials.json')
TOKEN_PATH = os.getenv('GOOGLE_TOKEN_PATH', 'gcalendar_token.json')

# Current user context (set by agent before calling tools)
_current_user_id = None


def set_current_user(user_id: int):
    """Set the current user context for multi-user support"""
    global _current_user_id
    _current_user_id = user_id


def get_current_user() -> int:
    """Get the current user context"""
    return _current_user_id


def get_service(user_id: int = None):
    """
    Get Google Calendar service for a specific user.
    If user_id is provided, use their token.
    Otherwise, fall back to the default token file (for single-user mode).
    """
    # Determine which user to use
    effective_user_id = user_id or _current_user_id
    
    if effective_user_id:
        # Multi-user mode: load from token manager
        creds = token_manager.load_credentials(effective_user_id)
        if not creds:
            raise ValueError(
                f"User {effective_user_id} has not connected their Google Calendar. "
                "Please use /connect command first."
            )
        return build('calendar', 'v3', credentials=creds)
    
    # Single-user mode: use default token file
    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, 'w') as f:
            f.write(creds.to_json())
    
    return build('calendar', 'v3', credentials=creds)

# === Timezone ===
TZ = pytz.timezone('Asia/Singapore')


def _get_week_range(base_date: datetime, offset_weeks: int = 0) -> tuple[datetime, datetime]:
    """
    Get Monday 00:00 ~ Sunday 23:59 for a given week.
    offset_weeks: 0 = this week, 1 = next week, -1 = last week
    """
    # Find Monday of the base week
    monday = base_date - timedelta(days=base_date.weekday())
    monday = monday.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Apply week offset
    monday = monday + timedelta(weeks=offset_weeks)
    sunday = monday + timedelta(days=6, hours=23, minutes=59, seconds=59)
    
    return monday, sunday


def _format_date_range(start: datetime, end: datetime) -> str:
    """Format date range for display."""
    weekdays = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']
    start_wd = weekdays[start.weekday()]
    end_wd = weekdays[end.weekday()]
    return f"{start.strftime('%m/%d')} {start_wd} ~ {end.strftime('%m/%d')} {end_wd}"


# === Actual functions ===
def _get_events(
    start_date: str = None,
    end_date: str = None,
    period: str = None
) -> str:
    """
    Get calendar events for a specific date range.
    
    Two ways to specify the range:
    1. Use start_date and end_date for flexible date ranges (LLM calculates)
    2. Use period shortcuts for common cases
    
    Args:
        start_date: Start date in "YYYY-MM-DD" format (e.g., "2025-12-01")
        end_date: End date in "YYYY-MM-DD" format (e.g., "2025-12-31")
        period: Shortcut - "today", "tomorrow", "week", "next_week", "last_week", or number of days
    
    Returns:
        Formatted string with events and date range
    """
    service = get_service()
    now = datetime.now(TZ)
    
    # Priority: start_date/end_date > period
    if start_date and end_date:
        # LLM provided specific dates
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=TZ)
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
                hour=23, minute=59, second=59, tzinfo=TZ
            )
            label = f"📅 {start_date} ~ {end_date}"
        except ValueError:
            return f"❌ Invalid date format. Use YYYY-MM-DD (e.g., 2025-12-01)"
    
    elif period:
        # Use period shortcuts
        if period == "today":
            start_dt = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end_dt = now.replace(hour=23, minute=59, second=59, microsecond=0)
            label = f"📅 Today ({now.strftime('%m/%d %a')})"
            
        elif period == "tomorrow":
            tomorrow = now + timedelta(days=1)
            start_dt = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
            end_dt = tomorrow.replace(hour=23, minute=59, second=59, microsecond=0)
            label = f"📅 Tomorrow ({tomorrow.strftime('%m/%d %a')})"
            
        elif period in ("week", "this_week"):
            start_dt, end_dt = _get_week_range(now, offset_weeks=0)
            label = f"📅 This week ({_format_date_range(start_dt, end_dt)})"
            
        elif period == "next_week":
            start_dt, end_dt = _get_week_range(now, offset_weeks=1)
            label = f"📅 Next week ({_format_date_range(start_dt, end_dt)})"
            
        elif period == "last_week":
            start_dt, end_dt = _get_week_range(now, offset_weeks=-1)
            label = f"📅 Last week ({_format_date_range(start_dt, end_dt)})"
            
        else:
            # Assume it's a number of days
            try:
                days = int(period)
                start_dt = now
                end_dt = now + timedelta(days=days)
                label = f"📅 Next {days} days ({now.strftime('%m/%d')} ~ {end_dt.strftime('%m/%d')})"
            except ValueError:
                return f"❌ Unknown period: {period}. Use 'today', 'tomorrow', 'week', 'next_week', or a number."
    
    else:
        # Default: next 7 days
        start_dt = now
        end_dt = now + timedelta(days=7)
        label = f"📅 Next 7 days ({now.strftime('%m/%d')} ~ {end_dt.strftime('%m/%d')})"
    
    # Fetch events from Google Calendar
    result = service.events().list(
        calendarId='primary',
        timeMin=start_dt.isoformat(),
        timeMax=end_dt.isoformat(),
        maxResults=50,
        singleEvents=True,
        orderBy='startTime'
    ).execute()
    
    events = result.get('items', [])
    
    if not events:
        return f"{label}\n\nNo events scheduled."
    
    # Format output
    output = f"{label}\n\n"
    current_date = None
    
    for e in events:
        # Parse event start time
        start_str = e['start'].get('dateTime', e['start'].get('date'))
        
        # Group by date
        if 'T' in start_str:
            event_dt = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
            event_date = event_dt.strftime('%m/%d (%a)')
            event_time = event_dt.strftime('%H:%M')
        else:
            event_date = start_str
            event_time = "All day"
        
        # Add date header if new date
        if event_date != current_date:
            if current_date is not None:
                output += "\n"
            output += f"📆 {event_date}\n"
            current_date = event_date
        
        # Add event
        title = e.get('summary', 'No title')
        location = e.get('location', '')
        loc_str = f" @ {location}" if location else ""
        output += f"  • {event_time} - {title}{loc_str}\n"
    
    return output


def _create_event(title: str, start: str, end: str, location: str = "") -> str:
    """
    Create new schedule.
    start/end format: 2025-12-15T10:00:00
    location: optional location of the event
    """
    service = get_service()
    event = {
        'summary': title,
        'start': {'dateTime': start, 'timeZone': 'Asia/Singapore'},
        'end': {'dateTime': end, 'timeZone': 'Asia/Singapore'},
    }
    if location:
        event['location'] = location
    created = service.events().insert(calendarId='primary', body=event).execute()
    return f"✅ Created: {title}\nLocation: {location or 'Not specified'}\nlink: {created.get('htmlLink')}"

# === MCP Tool wrapper ===
@mcp.tool()
def get_events(
    start_date: str = None,
    end_date: str = None,
    period: str = None
) -> str:
    """
    Get calendar events for a specific date range.
    
    Two ways to specify the range:
    1. Use start_date and end_date for flexible ranges (recommended for month, year, etc.)
       - Example: start_date="2025-12-01", end_date="2025-12-31" for December
       - Example: start_date="2026-01-01", end_date="2026-12-31" for year 2026
    
    2. Use period shortcuts for common cases:
       - "today": Today only
       - "tomorrow": Tomorrow only
       - "week" or "this_week": Monday ~ Sunday of current week
       - "next_week": Monday ~ Sunday of next week
       - "last_week": Monday ~ Sunday of last week
       - Number (e.g., "7"): Next N days from now
    
    Args:
        start_date: Start date in YYYY-MM-DD format (e.g., "2025-12-01")
        end_date: End date in YYYY-MM-DD format (e.g., "2025-12-31")
        period: Shortcut for common periods
    """
    return _get_events(start_date=start_date, end_date=end_date, period=period)

@mcp.tool()
def create_event(title: str, start: str, end: str, location: str = "") -> str:
    return _create_event(title, start, end, location)

if __name__ == "__main__":
    mcp.run()

