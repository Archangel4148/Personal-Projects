from copy import deepcopy
import os
from pathlib import Path
from file_tools import musicxml_to_pdf_musescore
from music_tools import clean_score, manual_add_chord_symbols, transpose_score
import music21 as m21

def main():
    # Load the score from the Audiveris mxl
    mxl_path = Path(r"imports\beethoven-symphony-no-5-1st-movement-piano-solo.mxl").resolve()
    # mxl_path = Path(r"imports\be_thou_my_vision.mxl").resolve()
    
    score = m21.converter.parse(mxl_path)
    source_key = m21.key.Key("Eb", "major")

    # Transpose to C and clean up formatting
    target_key = m21.key.Key("C", "major")
    score = transpose_score(score, source_key, target_key)
    score = clean_score(score)

    if score.metadata is None:
        score.insert(0, m21.metadata.Metadata())

    if score.metadata.title is not None:
        score.metadata.title += "(in C Major)"
    score.metadata.movementName += "(in C Major)"

    # TODO: Work some simplification magic

    # Write to output files
    output_directory = Path("output/").resolve()
    os.makedirs(output_directory, exist_ok=True)
    score.write("musicxml", output_directory / "simplified_music")
    
    # Convert MusicXML to PDF
    musicxml_to_pdf_musescore(
        output_directory / "simplified_music.musicxml",
        r"D:\Musescore 4\bin\MuseScore4.exe"
    )

if __name__ == "__main__":
    main()