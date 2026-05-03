import pandas as pd

from spell_data import load_all_spell_details


def get_ordered_attribute_lists(data) -> dict[str, list]:
    """Get the properly ordered columns to use to convert spell details to index values"""
    ordered_lists = {}
    
    # Sort all relevant columns alphabetically
    columns = data.columns[3:]
    for col in columns:
        ordered_lists[col] = sorted(map(str, data[col].unique()))

    # Area of Effect
    areas = ordered_lists["area_of_effect"]
    areas.remove("nan")
    areas.insert(0, "nan")
    ordered_lists["area_of_effect"] = areas

    # Duration
    durations = ordered_lists["duration"]
    durations.remove("Instantaneous")
    durations.insert(0, "Instantaneous")
    ordered_lists["duration"] = durations

    return ordered_lists

def get_spell_indices(order_lists: dict[str, list], spell_details: pd.DataFrame) -> pd.DataFrame:
    for row in list(spell_details.iterrows())[:3]:
        print(row[1].to_dict())
        # TODO: Build new dataframe with indices instead of values (probably keep name/index?)

if __name__ == "__main__":
    data = load_all_spell_details()
    ordered_lists = get_ordered_attribute_lists(data)
    get_spell_indices(ordered_lists, data)


