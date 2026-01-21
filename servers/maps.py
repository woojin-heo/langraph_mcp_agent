"""
FastMCP Google Maps Server
"""
import os
import json
import re
from datetime import datetime
from typing import Optional, Union
from dotenv import load_dotenv
from fastmcp import FastMCP
import googlemaps
from langsmith import traceable

load_dotenv()

mcp = FastMCP("Maps")

# Google Maps API Key setup
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')

def get_client():
    if not GOOGLE_MAPS_API_KEY:
        raise ValueError("GOOGLE_MAPS_API_KEY not set in environment variables")
    return googlemaps.Client(key=GOOGLE_MAPS_API_KEY)


# === Actual functions (can be called directly) ===

def _search_places(query: str, location: str = "") -> str:
    """Search for places using Google Maps."""
    client = get_client()
    
    if location:
        geocode = client.geocode(location)
        if geocode:
            lat_lng = geocode[0]['geometry']['location']
            results = client.places(query=query, location=lat_lng, radius=5000)
        else:
            results = client.places(query=query)
    else:
        results = client.places(query=query)
    
    places = results.get('results', [])
    if not places:
        return f"No places found for: {query}"
    
    output = f"Search results for '{query}':\n"
    for i, place in enumerate(places, 1):
        name = place.get('name', 'Unknown')
        address = place.get('formatted_address', 'No address')
        rating = place.get('rating', 'N/A')
        output += f"{i}. {name}\n   {address}\n   Rating: {rating}\n"
    
    return output


def _parse_duration_minutes(duration_text: str) -> Optional[int]:
    """Parse duration text to minutes (e.g., '1 hour 30 mins' -> 90)"""
    if not duration_text:
        return None
    
    total_minutes = 0
    
    # Match hours
    hour_match = re.search(r'(\d+)\s*hour', duration_text, re.IGNORECASE)
    if hour_match:
        total_minutes += int(hour_match.group(1)) * 60
    
    # Match minutes
    min_match = re.search(r'(\d+)\s*min', duration_text, re.IGNORECASE)
    if min_match:
        total_minutes += int(min_match.group(1))
    
    return total_minutes if total_minutes > 0 else None


def _get_directions(
    origin: str, 
    destination: str, 
    mode: str = "driving",
    arrival_time: Optional[datetime] = None
) -> dict:
    """
    Get directions between two places.
    
    Args:
        origin: Starting location
        destination: Destination location
        mode: Transport mode (driving, walking, bicycling, transit)
        arrival_time: Target arrival time (datetime object, used for transit mode)
    
    Returns:
        dict with structured directions data:
        - origin, destination
        - distance_text, duration_text, duration_minutes
        - requested_mode, actual_mode, fallback_used
        - steps (list)
        - text (formatted string for display)
    """
    client = get_client()
    
    # Build API call parameters
    api_params = {
        "origin": origin,
        "destination": destination,
        "mode": mode,
    }
    
    # if transit mode, add arrival_time
    if mode == "transit" and arrival_time:
        api_params["arrival_time"] = arrival_time
    
    directions = client.directions(**api_params)
    
    # Fallback logic: walking -> driving sequentially
    fallback_mode = None
    if not directions and mode == "transit":
        # if close distance, walking mode is fallback
        directions = client.directions(origin, destination, mode="walking")
        if directions:
            fallback_mode = "walking"
        else:
            # driving mode is fallback
            directions = client.directions(origin, destination, mode="driving")
            if directions:
                fallback_mode = "driving"
    
    if not directions:
        return {
            "error": f"No route found from {origin} to {destination}",
            "origin": origin,
            "destination": destination,
            "requested_mode": mode,
        }
    
    route = directions[0]
    leg = route['legs'][0]
    
    actual_mode = fallback_mode if fallback_mode else mode
    distance_text = leg['distance']['text']
    duration_text = leg['duration']['text']
    duration_minutes = _parse_duration_minutes(duration_text)
    
    # Build steps list
    steps = []
    for i, step in enumerate(leg['steps'][:10], 1):
        instruction = step['html_instructions'].replace('<b>', '').replace('</b>', '')
        instruction = instruction.replace('<div style="font-size:0.9em">', ' ').replace('</div>', '')
        steps.append({
            "step": i,
            "instruction": instruction,
            "distance": step['distance']['text']
        })
    
    # Build formatted text output
    text = f"Directions: {origin} -> {destination}\n"
    text += f"Distance: {distance_text}\n"
    text += f"Duration: {duration_text}\n"
    
    if fallback_mode:
        text += f"Mode: {actual_mode} (transit replaced by {fallback_mode})\n\nSteps:\n"
    else:
        text += f"Mode: {actual_mode}\n\nSteps:\n"
    
    for s in steps:
        text += f"{s['step']}. {s['instruction']} ({s['distance']})\n"
    
    if len(leg['steps']) > 10:
        text += f"... and {len(leg['steps']) - 10} more steps\n"
    
    return {
        "origin": origin,
        "destination": destination,
        "distance_text": distance_text,
        "duration_text": duration_text,
        "duration_minutes": duration_minutes,
        "requested_mode": mode,
        "actual_mode": actual_mode,
        "fallback_used": fallback_mode is not None,
        "steps": steps,
        "text": text,
    }


def _get_place_details(place_name: str) -> str:
    """Get details about a specific place by name."""
    client = get_client()
    
    results = client.places(query=place_name)
    places = results.get('results', [])
    
    if not places:
        return f"No place found: {place_name}"
    
    place_id = places[0]['place_id']
    details = client.place(place_id)['result']
    
    output = f"Place Details: {details.get('name', 'Unknown')}\n"
    output += f"Address: {details.get('formatted_address', 'N/A')}\n"
    output += f"Phone: {details.get('formatted_phone_number', 'N/A')}\n"
    output += f"Website: {details.get('website', 'N/A')}\n"
    output += f"Rating: {details.get('rating', 'N/A')} ({details.get('user_ratings_total', 0)} reviews)\n"
    
    if details.get('opening_hours'):
        output += "\nOpening Hours:\n"
        for day in details['opening_hours'].get('weekday_text', []):
            output += f"   {day}\n"
    
    return output


# === MCP Tool wrapper ===

@mcp.tool()
def search_places(query: str, location: str = "") -> str:
    """Search for places using Google Maps."""
    return _search_places(query, location)

@mcp.tool()
def get_directions(origin: str, destination: str, mode: str = "driving", arrival_time: str = "") -> str:
    """
    Get directions between two places.
    
    Args:
        origin: Starting location
        destination: Destination location
        mode: Transport mode (driving, walking, bicycling, transit)
        arrival_time: Target arrival time in ISO format (e.g., "2024-01-20T09:00:00"), used for transit mode
    
    Returns:
        JSON string containing directions data with actual_mode field
    """
    parsed_arrival_time = None
    if arrival_time and mode == "transit":
        try:
            parsed_arrival_time = datetime.fromisoformat(arrival_time)
        except ValueError:
            pass  # Invalid format, ignore
    
    result = _get_directions(origin, destination, mode, parsed_arrival_time)
    return json.dumps(result, ensure_ascii=False)

@mcp.tool()
def get_place_details(place_name: str) -> str:
    """Get details about a specific place by name."""
    return _get_place_details(place_name)


if __name__ == "__main__":
    mcp.run()
