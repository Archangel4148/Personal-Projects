
import csv
import sqlite3

DATASET_CSV_PATH = r"SQL Practice\Dataset.csv"
DB_PATH = r"SQL Practice\database.db"

# Create and connect to the database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Add the table
cursor.execute('''
CREATE TABLE IF NOT EXISTS music (
    track_id TEXT PRIMARY KEY,
    track_name TEXT NOT NULL,
    track_number INTEGER NOT NULL,
    track_popularity INTEGER NOT NULL,
    explicit BOOLEAN NOT NULL,
    artist_name TEXT NOT NULL,
    artist_popularity INTEGER NOT NULL,
    artist_followers INTEGER NOT NULL,
    artist_genres TEXT NOT NULL,
    album_id TEXT NOT NULL,
    album_name TEXT NOT NULL,
    album_release_date TEXT NOT NULL,
    album_total_tracks INTEGER NOT NULL,
    album_type TEXT NOT NULL,
    track_duration_min FLOAT NOT NULL
)
''')

# Load the CSV
with open(DATASET_CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)  # Skip the header
    for i, line in enumerate(reader):
        # Insert each row into the database
        cursor.execute("INSERT INTO music VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", line)

# Commit all changes to the database and close the connection
conn.commit()
conn.close()