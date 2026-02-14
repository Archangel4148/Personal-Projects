import os
from pathlib import Path
import subprocess

from music_tools import ScoreData

def musicxml_to_pdf(musicxml_path: str, musescore_exe: str):
    """Use MuseScore to convert MusicXML to PDF"""
    out_dir = Path(musicxml_path).resolve().parent
    out_pdf = Path(musicxml_path).with_suffix(".pdf")
    subprocess.run(
        [
            musescore_exe,
            musicxml_path,
            "-o",
            out_dir / out_pdf,
        ],
        cwd=out_dir,
        check=True,
    )

def main():
    score = """
    Title: You're Mother | Composer: J. Robert Oppenheimer
    Melody C 4/4 treble
    E4s F4s G4s A4s B4s C5s D5s E5s F5q - 
    [B4e C5e B4e A4e] G4q (A4s B4s C5s D5s) -
    (D5s C5s B4s A4s G4s F4s E4s D4s) C4q D4q -
    (E4s F4s G4s A4s) B4h. -
    C5s B4s A4s G4s F4s E4s D4s C4s C4h -
    (D4e E4e F4e G4e A4e B4e C5e D5e) -
    [E5q D5q] C5h -
    C5w
    | Bass C 4/4 bass
    [C3q G2q C3q G2q] -
    F3e E3e D3e C3e B2e A2e G2e F2e -
    [D3q A2q D3q A2q] -
    G2e A2e B2e C3e D3e E3e F3e G3e -
    [C3h G2q C3q] -
    F3s G3s A3s B3s C4s B3s A3s G3s G3h -
    [D3q A2q] G2h -
    C3w
    """

    # Parse melody string
    score_obj = ScoreData.from_string(score)

    # Add chord letters above the melody
    melody_part = score_obj.parts[0]

    melody_part.chords = {
        (1, 0.0): "C",
        (1, 2.0): "C/E",
        (2, 0.0): "F",
        (2, 2.0): "G",
        (3, 0.0): "G7",
        (3, 2.0): "C/E",
        (4, 0.0): "Am",
        (4, 1.0): "Dm",
        (4, 2.0): "G",
        (4, 3.0): "G7",
        (5, 0.0): "Em",
        (5, 2.0): "Am",
        (6, 0.0): "Dm",
        (6, 2.0): "G",
        (7, 0.0): "C/G",
        (7, 2.0): "F",
        (7, 3.0): "G7",
        (8, 0.0): "C",
    }
    # Convert to music21 object
    score = score_obj.to_music21()

    # Write to output files
    output_directory = Path("output/").resolve()
    os.makedirs(output_directory, exist_ok=True)
    score.write("musicxml", output_directory / "sheet_music")
    
    # Convert MusicXML to PDF
    musicxml_to_pdf(
        output_directory / "sheet_music.musicxml",
        r"D:\Musescore 4\bin\MuseScore4.exe"
    )

if __name__ == "__main__":
    main()
