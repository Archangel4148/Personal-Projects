import os

import pandas as pd
import requests


base_url = "https://www.dnd5eapi.co"
payload = {}
headers = {
  'Accept': 'application/json'
}

def get_all_spell_paths() -> pd.DataFrame:
    """Get all spell path info from the API, then save to a file for later use"""
    response = requests.request("GET", base_url + "/api/2014/spells", headers=headers, data=payload)
    results = response.json()["results"]
    return pd.DataFrame(results)

def get_spell_data(url: str) -> pd.DataFrame:
    """Get the desired spell info from a provided spell API url, formatted nicely"""
    response = requests.request("GET", url, headers=headers, data=payload).json()
    # Parse data fields
    parsed_data = {}
    standard_fields = ["index", "name", "range", "ritual", "duration", "concentration", "casting_time", "level"]
    for field in standard_fields:
        parsed_data[field] = response[field]
    parsed_data["components"] = "".join(response.get("components"))
    try:
        parsed_data["damage_type"] = response.get("damage")["damage_type"]["name"]
    except (TypeError, KeyError):
        parsed_data["damage_type"] = None
    try:
        parsed_data["area_of_effect"] = "".join(map(str, response.get("area_of_effect").values()))
    except AttributeError:
        parsed_data["area_of_effect"] = None
    parsed_data["school"] = response["school"]["name"]
    return parsed_data


def load_and_get_all_spell_details(check_exists: bool = True, verbose: bool = False) -> pd.DataFrame:
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    # If needed, get all spell paths
    all_spell_path = "data/spell_paths.csv"
    if not check_exists or not os.path.exists(all_spell_path):
        # Get spell list from API and save locally
        spell_path_data = get_all_spell_paths()
        spell_path_data.to_csv(all_spell_path)
        if verbose:
            print(f"Downloaded {len(spell_path_data)} spell paths from the API!")
    else:
        # Read local spell list
        spell_path_data = pd.read_csv(all_spell_path)
        if verbose:
            print(f"Loaded {len(spell_path_data)} spell paths from '{all_spell_path}'")

    # Get all spell details
    spell_details_path = "data/spell_details.csv"
    if not check_exists or not os.path.exists(spell_details_path):
        # Get spell details from API and save locally
        paths = spell_path_data["url"]
        spell_detail_data = pd.DataFrame([get_spell_data(base_url + path) for path in paths])
        spell_detail_data.to_csv(spell_details_path)
        if verbose:
            print(f"Downloaded {len(spell_detail_data)} spell details from the API!")
    else:
        # Read local spell details
        spell_detail_data = pd.read_csv(spell_details_path)
        if verbose:
            print(f"Loaded {len(spell_detail_data)} spell details from '{spell_details_path}'")

    return spell_detail_data

def main():
    load_and_get_all_spell_details(verbose=True)


if __name__ == "__main__":
    main()
