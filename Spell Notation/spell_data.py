import json
import os

import requests


base_url = "https://www.dnd5eapi.co"
payload = {}
headers = {
  'Accept': 'application/json'
}

def get_all_spell_paths(output_path: str):
    """Get all spell path info from the API, then save to a file for later use"""
    response = requests.request("GET", base_url + "/api/2014/spells", headers=headers, data=payload)
    results = response.json()["results"]
    with open(output_path, "w") as f:
        f.write("index,name,url")
        for result in results:
            f.write(f"{result['index']},{result['name']},{result["url"]}\n")

def get_spell_data(url: str):
    """Get the desired spell info from a provided spell API url, formatted nicely"""
    response = requests.request("GET", url, headers=headers, data=payload).json()
    standard_fields = ["range", "ritual", "duration", "concentration", "casting_time", "level"]
    custom_fields = ["components", "damage", "area_of_effect", "school"]
    # TODO: Parse this!
    # return {field: response[field] for field in fields}
    

def main():
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    # If needed, get all spell paths
    all_spell_path = "data/spell_paths.csv"
    if not os.path.exists(all_spell_path):
        get_all_spell_paths(all_spell_path)

    
    url = base_url + "/api/2014/spells/fireball"
    spell_data = get_spell_data(url)
    for key in spell_data.keys():
        print(key, ":", spell_data[key])

if __name__ == "__main__":
    main()