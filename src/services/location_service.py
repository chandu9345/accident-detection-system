# src/services/location_service.py
import requests
from datetime import datetime
from geopy.geocoders import Nominatim

def get_coordinates(place_name: str) -> dict:
    """Convert place name to lat/lon coordinates."""
    try:
        geolocator = Nominatim(user_agent="accident_detection_app")
        location   = geolocator.geocode(place_name, timeout=10)
        if location:
            return {
                "success"  : True,
                "lat"      : location.latitude,
                "lon"      : location.longitude,
                "address"  : location.address,
            }
        return {"success": False, "error": "Location not found"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_weather_risk(lat: float, lon: float) -> dict:
    """Fetch real weather data from Open-Meteo (free, no API key needed)."""
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,precipitation,windspeed_10m,"
            f"weathercode,visibility"
            f"&timezone=auto"
        )
        resp = requests.get(url, timeout=10)
        data = resp.json()
        curr = data.get("current", {})

        weathercode  = curr.get("weathercode",   0)
        precipitation= curr.get("precipitation", 0)
        windspeed    = curr.get("windspeed_10m", 0)
        visibility   = curr.get("visibility",    10000)
        temperature  = curr.get("temperature_2m",25)

        # Calculate weather risk score 0-100
        risk = 0
        if weathercode >= 95:   risk += 40  # thunderstorm
        elif weathercode >= 80: risk += 30  # heavy rain/snow
        elif weathercode >= 61: risk += 20  # moderate rain
        elif weathercode >= 51: risk += 10  # drizzle
        elif weathercode >= 40: risk += 25  # fog

        if precipitation > 10:  risk += 20
        elif precipitation > 5: risk += 10
        elif precipitation > 0: risk += 5

        if windspeed > 60:      risk += 20
        elif windspeed > 40:    risk += 10
        elif windspeed > 20:    risk += 5

        if visibility < 200:    risk += 25
        elif visibility < 500:  risk += 15
        elif visibility < 1000: risk += 8

        # Weather condition label
        if weathercode >= 95:   condition = "⛈ Thunderstorm"
        elif weathercode >= 80: condition = "🌨 Heavy Snow/Rain"
        elif weathercode >= 61: condition = "🌧 Rain"
        elif weathercode >= 51: condition = "🌦 Drizzle"
        elif weathercode >= 40: condition = "🌫 Fog"
        elif weathercode >= 1:  condition = "⛅ Cloudy"
        else:                   condition = "☀ Clear"

        return {
            "success"    : True,
            "risk"       : min(100, risk),
            "condition"  : condition,
            "temperature": temperature,
            "windspeed"  : windspeed,
            "visibility" : visibility,
            "precipitation": precipitation,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "risk": 20}


def get_time_risk() -> dict:
    """Calculate risk based on current time of day."""
    hour = datetime.now().hour

    if   4  <= hour < 8:   risk, label = 30, "🌅 Early Morning"
    elif 8  <= hour < 12:  risk, label = 45, "☀ Morning Peak"
    elif 12 <= hour < 17:  risk, label = 35, "🌤 Afternoon"
    elif 17 <= hour < 21:  risk, label = 60, "🌆 Evening Rush"
    elif 21 <= hour < 24:  risk, label = 50, "🌙 Night"
    else:                  risk, label = 55, "🌙 Late Night"

    return {"risk": risk, "label": label, "hour": hour}


def get_india_accident_risk(address: str) -> dict:
    """
    Estimate historical accident risk based on location keywords.
    Uses known high-risk zones in India.
    """
    address_lower = address.lower()

    high_risk_keywords = [
        "highway", "nh-", "national highway", "expressway",
        "flyover", "junction", "intersection", "ring road",
        "bypass", "ghat", "mountain", "hill"
    ]
    medium_risk_keywords = [
        "city", "urban", "market", "bazaar", "chowk",
        "road", "street", "avenue", "nagar", "colony"
    ]
    high_risk_cities = [
        "delhi", "mumbai", "bangalore", "hyderabad",
        "chennai", "pune", "ahmedabad", "jaipur",
        "lucknow", "kanpur", "nagpur", "indore"
    ]

    risk = 20
    factors = []

    for kw in high_risk_keywords:
        if kw in address_lower:
            risk += 25
            factors.append(f"High-risk road type: {kw}")
            break

    for kw in medium_risk_keywords:
        if kw in address_lower:
            risk += 10
            factors.append(f"Urban area detected")
            break

    for city in high_risk_cities:
        if city in address_lower:
            risk += 15
            factors.append(f"High accident city: {city.title()}")
            break

    return {"risk": min(100, risk), "factors": factors}


def predict_location_risk(place_name: str) -> dict:
    """
    Main function — combines all factors to predict accident risk
    for a given location.
    """
    # Step 1 — Get coordinates
    coords = get_coordinates(place_name)
    if not coords["success"]:
        return {"success": False, "error": coords["error"]}

    lat     = coords["lat"]
    lon     = coords["lon"]
    address = coords["address"]

    # Step 2 — Get weather risk
    weather = get_weather_risk(lat, lon)
    weather_risk = weather.get("risk", 20)

    # Step 3 — Get time risk
    time    = get_time_risk()
    time_risk = time["risk"]

    # Step 4 — Get historical risk
    historical = get_india_accident_risk(address)
    hist_risk  = historical["risk"]

    # Step 5 — Calculate final score (weighted average)
    final_score = round(
        (weather_risk  * 0.30) +
        (time_risk     * 0.25) +
        (hist_risk     * 0.30) +
        (20            * 0.15)   # base road risk
    )
    final_score = min(99, max(5, final_score))

    # Step 6 — Risk level
    if   final_score < 30: level, color = "LOW",      "🟢"
    elif final_score < 60: level, color = "MODERATE", "🟡"
    elif final_score < 80: level, color = "HIGH",     "🔴"
    else:                  level, color = "CRITICAL",  "🚨"

    return {
        "success"     : True,
        "score"       : final_score,
        "level"       : level,
        "color"       : color,
        "address"     : address,
        "lat"         : lat,
        "lon"         : lon,
        "weather"     : weather,
        "time"        : time,
        "historical"  : historical,
        "weather_risk": weather_risk,
        "time_risk"   : time_risk,
        "hist_risk"   : hist_risk,
    }