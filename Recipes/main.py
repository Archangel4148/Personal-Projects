import ast
import time
from pathlib import Path
from typing import cast

import pandas as pd

# https://www.kaggle.com/datasets/wilmerarltstrmberg/recipe-dataset-over-2m

DATASET_PATH = Path(__file__).parent / "data"
LOAD_BASE_DATASET = False
ROW_LOAD_LIMIT = None

def parse_string_list(s):
    return [x.strip().strip("'\"") for x in s[1:-1].split(",")]

def print_recipe(recipe_data: pd.Series):
    recipe_dict = recipe_data.to_dict()
    for k, v in recipe_dict.items():
        if isinstance(v, list) and k != "NER":
            print(f"{k}:")
            for item in v:
                print(f"\t- {item}")
        else:
            print(k, ":", v)

def find_recipes_with_ingredient(dataset, ingredient):
    """Get all recipes that contain the provided ingredient (in the NER column)"""
    mask = dataset["NER"].map(lambda x: ingredient in x)
    return list(dataset[mask].iterrows())

def main():
    # Data paths
    recipe_path = DATASET_PATH / "recipes_data.csv"
    cookie_path = DATASET_PATH / "cookie_recipes.pkl"

    if LOAD_BASE_DATASET or not cookie_path.exists():
        # Load the dataset
        print("Loading dataset...")
        t = time.perf_counter()
        df: pd.DataFrame = cast(pd.DataFrame, pd.read_csv(recipe_path, nrows=ROW_LOAD_LIMIT, encoding="utf-8"))
        print(f"Finished loading dataset in {time.perf_counter() - t:.3f} s.")

        # Parse/format list columns
        print("Parsing list variables...")
        list_columns = ["ingredients", "directions", "NER"]
        for col in list_columns:
            df[col] = df[col].apply(parse_string_list)

        # Separate cookie recipes
        print("Separating cookie recipes...")
        cookie_recipes = df[df["title"].str.contains("cookie", case=False, na=False)]
        cookie_recipes.reset_index(drop=True, inplace=True)
        cookie_recipes.to_pickle(cookie_path)
    else:
        cookie_recipes = cast(pd.DataFrame, pd.read_pickle(cookie_path))

    # Find list of unique ingredients
    unique_ingredients = set()
    for l in cookie_recipes["NER"]:
        unique_ingredients.update(l)
    print("Unique Ingredients:", len(unique_ingredients))

    # Find recipes with a specific ingredient
    ingredient = "lemon zest"
    matches = find_recipes_with_ingredient(cookie_recipes, ingredient)
    print(f"\nNumber of recipes with \'{ingredient}\': {len(matches)}\n")

    print_recipe(matches[6][1])

if __name__ == "__main__":
    main()