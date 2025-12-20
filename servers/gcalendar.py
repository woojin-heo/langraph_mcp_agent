"""
FastMCP Google Calendar server
"""
import os
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv
from fastmcp import FastMCP

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

load_dotenv()

mcp = FastMCP("Calendar")

SCOPES = ['https://www.googleapis.com/auth/calendar']

CREDENTIALS_PATH = os.getenv('GOOGLE_CREDENTIALS_PATH', 'gcalendar_credentials.json')
TOKEN_PATH = os.getenv('GOOGLE_TOKEN_PATH', 'gcalendar_token.json')

def get_service():
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

# === Actual functions ===
def _get_events(days: int = 7) -> str:
    """Get the next N days schedule of your calendar."""
    service = get_service()

    now_tz = datetime.now(pytz.timezone('Asia/Singapore'))

    now = now_tz.isoformat()
    end = (now_tz + timedelta(days=days)).isoformat()
    
    result = service.events().list(
        calendarId='primary', timeMin=now, timeMax=end,
        maxResults=20, singleEvents=True, orderBy='startTime'
    ).execute()
    
    events = result.get('items', [])
    if not events:
        return "No scheduled events."
    
    output = f"📅 {days}days schedule:\n"
    for e in events:
        start = e['start'].get('dateTime', e['start'].get('date'))
        location = e.get('location', 'No location')
        output += f"• {start}: {e.get('summary', 'No title')} @ {location}\n"
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
def get_events(days: int = 7) -> str:
    return _get_events(days)

@mcp.tool()
def create_event(title: str, start: str, end: str, location: str = "") -> str:
    return _create_event(title, start, end, location)

if __name__ == "__main__":
    mcp.run()

