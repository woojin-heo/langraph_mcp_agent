"""
FastMCP Google Maps Server (Template)
"""
from fastmcp import FastMCP

mcp = FastMCP("Maps")

# TODO: Add Google Maps API key setup


@mcp.tool()
def search_places(query: str, location: str = "") -> str:
    """Search for places."""
    # TODO: Implement
    return f"📍 Searching: {query}"


@mcp.tool()
def get_directions(origin: str, destination: str, mode: str = "driving") -> str:
    """Get directions between two places."""
    # TODO: Implement
    return f"🗺️ {origin} → {destination}"


@mcp.tool()
def get_place_details(place_id: str) -> str:
    """Get details about a specific place."""
    # TODO: Implement
    return f"ℹ️ Place details..."


if __name__ == "__main__":
    mcp.run()

