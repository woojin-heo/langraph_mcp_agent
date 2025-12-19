"""
FastMCP Google Maps Server
"""
import os
from dotenv import load_dotenv
from fastmcp import FastMCP
import googlemaps

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


def _get_directions(origin: str, destination: str, mode: str = "driving") -> str:
    """Get directions between two places."""
    client = get_client()
    
    directions = client.directions(origin, destination, mode=mode)
    
    if not directions:
        return f"No route found from {origin} to {destination}"
    
    route = directions[0]
    leg = route['legs'][0]
    
    output = f"Directions: {origin} → {destination}\n"
    output += f"Distance: {leg['distance']['text']}\n"
    output += f"Duration: {leg['duration']['text']}\n"
    output += f"Mode: {mode}\n\nSteps:\n"
    
    for i, step in enumerate(leg['steps'][:10], 1):
        instruction = step['html_instructions'].replace('<b>', '').replace('</b>', '')
        instruction = instruction.replace('<div style="font-size:0.9em">', ' ').replace('</div>', '')
        output += f"{i}. {instruction} ({step['distance']['text']})\n"
    
    if len(leg['steps']) > 10:
        output += f"... and {len(leg['steps']) - 10} more steps\n"
    
    return output


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
def get_directions(origin: str, destination: str, mode: str = "driving") -> str:
    """Get directions between two places. mode: driving, walking, bicycling, transit"""
    return _get_directions(origin, destination, mode)

@mcp.tool()
def get_place_details(place_name: str) -> str:
    """Get details about a specific place by name."""
    return _get_place_details(place_name)


if __name__ == "__main__":
    mcp.run()