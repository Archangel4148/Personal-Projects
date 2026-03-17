import os
from pathlib import Path
import subprocess

from music21 import environment

from file_tools import musicxml_to_pdf_musescore


import music21 as m21

from music_tools import ScoreData, clean_score, manual_add_chord_symbols, transpose_score

def main():
    score = """
    Title: Be Thou My Vision | Composer: Unknown (Irish Melody)
    Melody Eb 3/4 treble
    v1:
    Eb4q Eb4q F4e Eb4e -
    C4q Bb3q C4q -
    v2:
    Bb3q Ab3q Bb3q -
    Ab3q Bb3q C4q -
    | Bass Eb 3/4 bass
    v3:
    G3q F3q Eb3q -
    Eb3q D3q Eb3q -
    v4:
    Eb2q F2q G2q -
    Ab2q Bb2q Ab2q -
    """

    # Parse melody string
    score_obj = ScoreData.from_string(score)

    # Add chord letters above the melody
    melody_part = score_obj.parts[0]

    # chord_map = {
    #     (1, 0.0): "E-",
    #     (2, 0.0): "A-",
    #     (3, 0.0): "Cm",
    #     (3, 2.0): "B-/D",
    #     (4, 0.0): "E-",
    #     (5, 0.0): "B-",
        
    #     (6, 0.0): "E-",
    #     (7, 0.0): "A-",
    #     (8, 0.0): "B-",
    #     (9, 0.0): "A-",
    #     (10, 1.0): "E-/G",
        
    #     (11, 0.0): "Cm7",
    #     (12, 0.0): "A-",
    #     (12, 2.0): "B-",
    #     (13, 0.0): "Cm",
    #     (14, 2.0): "E-/G",
    #     (15, 0.0): "A-",
    #     (15, 2.0): "A-/B-",
    #     (16, 0.0): "E-",
    # }
    # melody_part.chords = chord_map

    # Convert to music21 object
    # score = score_obj.to_music21()
    # source_key = score_obj.key_signature.to_music21()

    # Load the score from the Audiveris mxl
    mxl_path = Path(r"imports\beethoven-symphony-no-5-1st-movement-piano-solo.mxl").resolve()
    score = m21.converter.parse(mxl_path)
    source_key = m21.key.Key("Eb", "major")

    # manual_add_chord_symbols(score, chord_map)

    # Transpose to C
    target_key = m21.key.Key("C", "major")
    score = transpose_score(score, source_key, target_key)
    score = clean_score(score)

    # Write to output files
    output_directory = Path("output/").resolve()
    os.makedirs(output_directory, exist_ok=True)
    score.write("musicxml", output_directory / "sheet_music")
    
    # Convert MusicXML to PDF
    musicxml_to_pdf_musescore(
        output_directory / "sheet_music.musicxml",
        r"D:\Musescore 4\bin\MuseScore4.exe"
    )

if __name__ == "__main__":
    main()
