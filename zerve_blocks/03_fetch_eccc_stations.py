# Zerve Block: fetch_eccc_stations
# DAG position: 3 of 4 (leftmost, parallel with 01_load_statcan_data)
# Inputs: none — hardcoded station configuration
# Outputs: PRAIRIE_STATIONS (list of 12 dicts with province, station_id, name, lat, lon),
#          stations_meta_df (DataFrame preview), YEAR_START=1990, YEAR_END=2023
#
# WHAT THIS DOES: Defines a list of 12 ECCC climate stations across Prairie provinces
#   (AB: 4, SK: 4, MB: 4) with their Climate ID strings and coordinates. Sets the
#   target year range to 1990-2023. Creates a preview DataFrame. No HTTP calls are made.
#
# DIVERGENCES FROM GROUND TRUTH (scripts/pipeline_eccc_weather.py):
#   - Uses 12 stations; ground truth uses 10 (no Red Deer, Moose Jaw, Portage La Prairie, The Pas;
#     includes Medicine Hat, Swift Current CDA, Indian Head CDA instead)
#   - Uses Climate ID strings (e.g. "3012205"); ground truth uses numeric station IDs
#     (e.g. 50430, 2205) which are the ECCC bulk-download IDs — these are different identifiers
#   - Does not handle station ID changes over time; ground truth uses (station_id, year_start,
#     year_end) tuples to account for stations that changed IDs mid-record
#   - Year range 1990-2023; ground truth uses 2000-2024
#   - Station selection: includes non-agricultural stations (The Pas at 54°N);
#     ground truth curates for agricultural zones


import requests
import pandas as pd

# Representative ECCC climate stations for Prairie provinces
# Selected for long records, agricultural zones, and data completeness
# Station IDs from ECCC Climate Data Inventory (climate.weather.gc.ca)
PRAIRIE_STATIONS = [
    # Alberta
    {"province": "AB", "station_id": "3012205", "name": "Calgary Int'l A",        "lat": 51.11, "lon": -114.02},
    {"province": "AB", "station_id": "3012475", "name": "Edmonton Int'l A",        "lat": 53.31, "lon": -113.58},
    {"province": "AB", "station_id": "3054519", "name": "Lethbridge A",            "lat": 49.63, "lon": -112.80},
    {"province": "AB", "station_id": "3011346", "name": "Red Deer A",              "lat": 52.18, "lon": -113.89},
    # Saskatchewan
    {"province": "SK", "station_id": "4016560", "name": "Regina Int'l A",          "lat": 50.43, "lon": -104.67},
    {"province": "SK", "station_id": "4057120", "name": "Saskatoon Diefenbaker A", "lat": 52.17, "lon": -106.72},
    {"province": "SK", "station_id": "4028040", "name": "Swift Current A",         "lat": 50.27, "lon": -107.69},
    {"province": "SK", "station_id": "4015560", "name": "Moose Jaw A",             "lat": 50.33, "lon": -105.55},
    # Manitoba
    {"province": "MB", "station_id": "5023222", "name": "Winnipeg Richardson A",  "lat": 49.92, "lon": -97.23},
    {"province": "MB", "station_id": "5010480", "name": "Brandon A",              "lat": 49.91, "lon": -99.95},
    {"province": "MB", "station_id": "5022800", "name": "Portage La Prairie",     "lat": 49.97, "lon": -98.27},
    {"province": "MB", "station_id": "5062922", "name": "The Pas A",              "lat": 53.97, "lon": -101.10},
]

# Target year range matching typical crop data from Statistics Canada
YEAR_START = 1990
YEAR_END   = 2023

print(f"Prairie climate stations defined: {len(PRAIRIE_STATIONS)}")
print(f"Year range target: {YEAR_START} – {YEAR_END} ({YEAR_END - YEAR_START + 1} years)\n")

# Preview station list
stations_meta_df = pd.DataFrame(PRAIRIE_STATIONS)
print(stations_meta_df.to_string(index=False))
