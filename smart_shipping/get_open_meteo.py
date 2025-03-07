# %%
import requests
import json


# %%
def get_marine_data(latitude, longitude, start_date, end_date):
    url = "https://marine-api.open-meteo.com/v1/marine"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "wave_height,wave_direction,wave_period,swell_wave_height,swell_wave_direction,swell_wave_period,wind_wave_height,wind_wave_direction,wind_wave_period",
        "timezone": "UTC"
    }
    #add header to avoid blocking of IP when scraping data
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"  # Example Chrome User-Agent
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        return data

    except requests.exceptions.RequestException as e:
        print(f"Error during HTTP request: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        return None

# Example usage:
latitude = 51.537111
longitude = 0.849485
start_date = "2025-03-07"
end_date = "2025-03-08"

marine_data = get_marine_data(latitude, longitude, start_date, end_date)

if marine_data:
    print(json.dumps(marine_data, indent=4)) #pretty print the json.
    # Access specific data (example):
    if "hourly" in marine_data and "time" in marine_data["hourly"]:
      times = marine_data["hourly"]["time"]
      if "wave_height" in marine_data["hourly"]:
          wave_heights = marine_data["hourly"]["wave_height"]
          for i in range(len(times)):
            print(f"Time: {times[i]}, Wave Height: {wave_heights[i]} meters")
else:
    print("Failed to retrieve marine data.")

# %%



