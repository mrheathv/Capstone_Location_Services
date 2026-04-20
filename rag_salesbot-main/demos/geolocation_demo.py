import duckdb
import pandas as pd
import json
from math import radians, cos, sin, asin, sqrt
from openai import OpenAI

# ======================================
# DuckDB Setup
# ======================================
data = {
    "account_id": [1, 2, 3, 4, 5, 6, 7],
    "name": [
        "Acme Corporation",
        "Bluth Company",
        "Donquadtech",
        "Goodsilron",
        "Mathtouch",
        "Singletechno",
        "Treequote",
    ],
    "lat": [42.001, 42.025, 42.022, 42.049, 41.5698, 41.7583, 41.7049],
    "lon": [-93.621, -93.612, -93.650, -93.618, -93.8065, -93.5732, -93.5792],
    "last_interaction": [
        "Quarterly review completed; discussed expansion.",
        "Issues with mobile banking portal reported last Tuesday.",
        "Follow-up on software renewal; waiting for signature.",
        "Walk-through of new retail site; client interested in POS upgrade.",
        "Shopping is done; now need to discuss the next shopping options.",
        "Grocery needs to be purchased; need to find the closest grocery store.",
        "House is fallen apart; contractor is coming tomorrow.",
    ],
    "industry": [
        "Education",
        "Finance",
        "Software",
        "Retail",
        "Shopping",
        "Food",
        "House Decor",
    ],
}

conn = duckdb.connect("crm.duckdb")
df_accounts = pd.DataFrame(data)
conn.execute("CREATE OR REPLACE TABLE accounts AS SELECT * FROM df_accounts")


# ======================================
# Distance Calculation (Haversine)
# ======================================
def calculate_distance(lat1, lon1, lat2, lon2):
    """Return distance in miles."""
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(
        radians, [lat1, lon1, lat2, lon2]
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * asin(sqrt(a)) * R


# ======================================
# OpenAI Tool Definition (Geocoding)
# ======================================
tools = [
    {
        "type": "function",
        "function": {
            "name": "extract_coordinates",
            "description": "Extract latitude and longitude from a user location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {
                        "type": "number",
                        "description": "Latitude of the location",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Longitude of the location",
                    },
                },
                "required": ["latitude", "longitude"],
            },
        },
    }
]


# ======================================
# LLM Tool-Based Geocoding
# ======================================
def get_coords_from_llm(user_prompt: str):
    """
    Uses OpenAI tool-calling to extract lat/lon.
    Returns (lat, lon)
    """
    client = OpenAI()  # reads OPENAI_API_KEY from environment

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Identify the user's geographic location and call the "
                    "`extract_coordinates` tool with the correct latitude "
                    "and longitude. Do not respond with text."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        tools=tools,
        tool_choice="required",
    )

    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)

    return args["latitude"], args["longitude"]


# ======================================
# Business Logic: Nearby Account Search
# ======================================
def find_nearby_accounts(
    user_prompt: str,
    radius_miles: float = 30,
    top_n: int = 3,
):
    user_lat, user_lon = get_coords_from_llm(user_prompt)

    df = conn.execute("SELECT * FROM accounts").df()

    df["distance_miles"] = df.apply(
        lambda row: calculate_distance(
            user_lat, user_lon, row.lat, row.lon
        ),
        axis=1,
    )

    results = (
        df[df["distance_miles"] <= radius_miles]
        .sort_values("distance_miles")
        .head(top_n)
        .reset_index(drop=True)
    )

    return results

# ======================================
# Example Execution
# ======================================
if __name__ == "__main__":
    prompt = (
        "I am in Des Moines, Iowa and I have 2 hours. "
        "Give me 3 accounts within 30 miles."
    )

    nearby_accounts = find_nearby_accounts(prompt)
    print(nearby_accounts)
