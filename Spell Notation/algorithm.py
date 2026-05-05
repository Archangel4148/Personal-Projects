import pandas as pd
from pandas import DataFrame

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


def get_spell_indices(order_lists: dict[str, list], spell_details: pd.DataFrame, skip_rows: list[str]) -> pd.DataFrame:
    """Convert spell details to index values"""
    all_dicts = []
    for row in list(spell_details.iterrows())[:3]:
        row_dict = row[1].to_dict()
        index_dict = {k: order_lists[k].index(str(v)) for k, v in row_dict.items() if
                      k in order_lists and k not in skip_rows}
        for key in row_dict.keys():
            if key not in index_dict.keys():
                index_dict[key] = row_dict[key]
        all_dicts.append(index_dict)
    return DataFrame(all_dicts)


def get_binary_indices(index_df: DataFrame, length: int = 0) -> DataFrame:
    """Convert index values to binary values"""
    def dec_to_bin(v):
        if type(v) == int:
            return bin(v)[2:].zfill(length)
        else:
            return v
    binary_df = index_df.copy().map(dec_to_bin)
    return binary_df


def get_processed_spells() -> DataFrame:
    data = load_all_spell_details()
    ordered_lists = get_ordered_attribute_lists(data)
    skip_rows = ["ritual", "concentration", "Unnamed: 0"]
    index_df = get_spell_indices(ordered_lists, data, skip_rows)
    binary_df = get_binary_indices(index_df, 13)
    return binary_df

def main():
    data = load_all_spell_details()
    ordered_lists = get_ordered_attribute_lists(data)

    for k, v in ordered_lists.items():
        print(k, ":", v)
    print()
    skip_rows = ["ritual", "concentration", "Unnamed: 0"]
    index_df = get_spell_indices(ordered_lists, data, skip_rows)

    print(index_df.head())

    binary_df = get_binary_indices(index_df, 13)
    print(binary_df.head())


if __name__ == "__main__":
    main()
