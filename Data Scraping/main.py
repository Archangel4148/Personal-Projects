import re
import pandas as pd

with open("raw_data.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Regex to parse the data
pattern = re.compile(
    r"(\d+)\t([^\n]+).*?\n"  # rank + tountry
    r"(UEFA|CONMEBOL|CONCACAF|AFC|CAF|OFC|Other)\n"  # confederation
    r"([WLDR—\s\n]{5,15})\n"  # form
    r"([\d,]+)(?:\t([^\n]*))?",  # rating + change
    re.DOTALL,
)

matches = pattern.findall(raw_text)

cleaned_data = []
for match in matches:
    # Get parts
    rank, country, confed, form_raw, points, change = match
    form = "".join(form_raw.strip().splitlines())

    # Handle missing change values
    change_raw = change.strip().replace("—", "")
    change_val =  change_raw if change_raw else None

    # Put the data together
    cleaned_data.append(
        {
            "Rank": int(rank),
            "Team": country.strip(),
            "Confederation": confed,
            "Form": form,
            "Rating": int(points.replace(",", "")),
            "Change": change_val,
        }
    )

# Create the data frame
df = pd.DataFrame(cleaned_data)
df.to_csv("football_ratings.csv")

print(df.tail(10))