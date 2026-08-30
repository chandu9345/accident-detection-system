# src/services/map_service.py
import folium
from folium.plugins import HeatMap, MarkerCluster
import pandas as pd

# ── Global accident hotspot data ─────────────────────────────────────────
GLOBAL_ACCIDENT_DATA = [
    # India hotspots
    {"city": "Delhi",          "lat": 28.6139, "lon": 77.2090,  "risk": 85, "country": "India",   "deaths_year": 1516},
    {"city": "Mumbai",         "lat": 19.0760, "lon": 72.8777,  "risk": 78, "country": "India",   "deaths_year": 1198},
    {"city": "Bangalore",      "lat": 12.9716, "lon": 77.5946,  "risk": 72, "country": "India",   "deaths_year": 987},
    {"city": "Chennai",        "lat": 13.0827, "lon": 80.2707,  "risk": 70, "country": "India",   "deaths_year": 876},
    {"city": "Hyderabad",      "lat": 17.3850, "lon": 78.4867,  "risk": 68, "country": "India",   "deaths_year": 754},
    {"city": "Pune",           "lat": 18.5204, "lon": 73.8567,  "risk": 65, "country": "India",   "deaths_year": 643},
    {"city": "Jaipur",         "lat": 26.9124, "lon": 75.7873,  "risk": 74, "country": "India",   "deaths_year": 712},
    {"city": "Lucknow",        "lat": 26.8467, "lon": 80.9462,  "risk": 71, "country": "India",   "deaths_year": 689},
    {"city": "Nagpur",         "lat": 21.1458, "lon": 79.0882,  "risk": 66, "country": "India",   "deaths_year": 534},
    {"city": "Ahmedabad",      "lat": 23.0225, "lon": 72.5714,  "risk": 69, "country": "India",   "deaths_year": 678},
    {"city": "Kanpur",         "lat": 26.4499, "lon": 80.3319,  "risk": 73, "country": "India",   "deaths_year": 598},
    {"city": "Bhopal",         "lat": 23.2599, "lon": 77.4126,  "risk": 64, "country": "India",   "deaths_year": 445},
    {"city": "Patna",          "lat": 25.5941, "lon": 85.1376,  "risk": 76, "country": "India",   "deaths_year": 623},
    {"city": "Agra",           "lat": 27.1767, "lon": 78.0081,  "risk": 77, "country": "India",   "deaths_year": 589},
    {"city": "Kolkata",        "lat": 22.5726, "lon": 88.3639,  "risk": 67, "country": "India",   "deaths_year": 723},
    # Global hotspots
    {"city": "Beijing",        "lat": 39.9042, "lon": 116.4074, "risk": 72, "country": "China",   "deaths_year": 52000},
    {"city": "Shanghai",       "lat": 31.2304, "lon": 121.4737, "risk": 68, "country": "China",   "deaths_year": 38000},
    {"city": "Jakarta",        "lat": -6.2088, "lon": 106.8456, "risk": 80, "country": "Indonesia","deaths_year": 31000},
    {"city": "Bangkok",        "lat": 13.7563, "lon": 100.5018, "risk": 82, "country": "Thailand","deaths_year": 22000},
    {"city": "Cairo",          "lat": 30.0444, "lon": 31.2357,  "risk": 88, "country": "Egypt",   "deaths_year": 12000},
    {"city": "Lagos",          "lat": 6.5244,  "lon": 3.3792,   "risk": 90, "country": "Nigeria", "deaths_year": 18000},
    {"city": "Karachi",        "lat": 24.8607, "lon": 67.0011,  "risk": 84, "country": "Pakistan","deaths_year": 9000},
    {"city": "Dhaka",          "lat": 23.8103, "lon": 90.4125,  "risk": 86, "country": "Bangladesh","deaths_year": 7500},
    {"city": "Nairobi",        "lat": -1.2921, "lon": 36.8219,  "risk": 79, "country": "Kenya",   "deaths_year": 3200},
    {"city": "Mexico City",    "lat": 19.4326, "lon": -99.1332, "risk": 75, "country": "Mexico",  "deaths_year": 16000},
    {"city": "São Paulo",      "lat": -23.5505,"lon": -46.6333, "risk": 73, "country": "Brazil",  "deaths_year": 35000},
    {"city": "Rio de Janeiro", "lat": -22.9068,"lon": -43.1729, "risk": 71, "country": "Brazil",  "deaths_year": 18000},
    {"city": "Moscow",         "lat": 55.7558, "lon": 37.6173,  "risk": 62, "country": "Russia",  "deaths_year": 15000},
    {"city": "Istanbul",       "lat": 41.0082, "lon": 28.9784,  "risk": 66, "country": "Turkey",  "deaths_year": 7500},
    {"city": "Tehran",         "lat": 35.6892, "lon": 51.3890,  "risk": 83, "country": "Iran",    "deaths_year": 14000},
    {"city": "New York",       "lat": 40.7128, "lon": -74.0060, "risk": 42, "country": "USA",     "deaths_year": 270},
    {"city": "Los Angeles",    "lat": 34.0522, "lon": -118.2437,"risk": 48, "country": "USA",     "deaths_year": 350},
    {"city": "London",         "lat": 51.5074, "lon": -0.1278,  "risk": 35, "country": "UK",      "deaths_year": 130},
    {"city": "Paris",          "lat": 48.8566, "lon": 2.3522,   "risk": 38, "country": "France",  "deaths_year": 230},
    {"city": "Tokyo",          "lat": 35.6762, "lon": 139.6503, "risk": 30, "country": "Japan",   "deaths_year": 1500},
    {"city": "Sydney",         "lat": -33.8688,"lon": 151.2093, "risk": 28, "country": "Australia","deaths_year": 320},
    {"city": "Riyadh",         "lat": 24.7136, "lon": 46.6753,  "risk": 80, "country": "Saudi Arabia","deaths_year": 6800},
    {"city": "Johannesburg",   "lat": -26.2041,"lon": 28.0473,  "risk": 77, "country": "S. Africa","deaths_year": 14000},
]

def get_marker_color(risk: int) -> str:
    if   risk >= 80: return "#ff2d2d"
    elif risk >= 60: return "#ffd600"
    elif risk >= 40: return "#ff6b35"
    else:            return "#39ff14"

def get_risk_label(risk: int) -> str:
    if   risk >= 80: return "CRITICAL"
    elif risk >= 60: return "HIGH"
    elif risk >= 40: return "MODERATE"
    else:            return "LOW"

def build_world_map(filter_level: str = "All", center_india: bool = False) -> folium.Map:
    """Build the interactive world accident risk map."""

    # Center on India or World
    if center_india:
        start_location = [20.5937, 78.9629]
        start_zoom     = 5
    else:
        start_location = [20.0, 20.0]
        start_zoom     = 2

    # Create map
    m = folium.Map(
        location        = start_location,
        zoom_start      = start_zoom,
        tiles           = "CartoDB dark_matter",
        prefer_canvas   = True,
    )

    # Filter data
    filtered = GLOBAL_ACCIDENT_DATA
    if filter_level == "Critical (80+)":
        filtered = [d for d in GLOBAL_ACCIDENT_DATA if d["risk"] >= 80]
    elif filter_level == "High (60-79)":
        filtered = [d for d in GLOBAL_ACCIDENT_DATA if 60 <= d["risk"] < 80]
    elif filter_level == "Moderate (40-59)":
        filtered = [d for d in GLOBAL_ACCIDENT_DATA if 40 <= d["risk"] < 60]
    elif filter_level == "Low (<40)":
        filtered = [d for d in GLOBAL_ACCIDENT_DATA if d["risk"] < 40]

    # Add heatmap layer
    heat_data = [[d["lat"], d["lon"], d["risk"]/100] for d in filtered]
    HeatMap(
        heat_data,
        min_opacity = 0.3,
        radius      = 35,
        blur        = 25,
        gradient    = {
            "0.2": "#39ff14",
            "0.5": "#ffd600",
            "0.7": "#ff6b35",
            "1.0": "#ff2d2d"
        }
    ).add_to(m)

    # Add markers
    for d in filtered:
        color = get_marker_color(d["risk"])
        label = get_risk_label(d["risk"])

        popup_html = f"""
        <div style="background:#0d0d15;color:#eeeef8;padding:12px 16px;
                    font-family:monospace;min-width:200px;
                    border-left:3px solid {color}">
            <div style="font-size:16px;font-weight:bold;
                        color:{color};margin-bottom:8px">
                {d['city']}
            </div>
            <div style="font-size:11px;color:#7070a0;
                        margin-bottom:4px">{d['country']}</div>
            <div style="font-size:13px;margin:6px 0">
                Risk Score: <b style="color:{color}">{d['risk']}/100</b>
            </div>
            <div style="font-size:11px;color:{color};
                        border:1px solid {color};padding:2px 8px;
                        display:inline-block;margin-bottom:6px">
                {label}
            </div>
            <div style="font-size:11px;color:#7070a0">
                Deaths/year: {d['deaths_year']:,}
            </div>
        </div>
        """

        folium.CircleMarker(
            location     = [d["lat"], d["lon"]],
            radius       = max(6, d["risk"] // 10),
            color        = color,
            fill         = True,
            fill_color   = color,
            fill_opacity = 0.6,
            weight       = 1.5,
            popup        = folium.Popup(popup_html, max_width=250),
            tooltip      = f"{d['city']} — Risk: {d['risk']} ({label})"
        ).add_to(m)

    return m