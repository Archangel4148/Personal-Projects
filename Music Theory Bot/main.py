import os
from pathlib import Path
import subprocess
from music21 import environment, lily

# ALWAYS set explicitly (do NOT guard with "if not")
environment.UserSettings()['lilypondPath'] = (
    r"D:\LilyPond\lilypond-2.24.4\bin\lilypond.exe"
)

from music_tools import Melody

def musicxml_to_lilypond_pdf(musicxml_path: str, lilypond_exe: str):
    """Use LilyPond to convert MusicXML to PDF"""
    musicxml_path = Path(musicxml_path).resolve()
    out_dir = musicxml_path.parent

    lilypond_exe = Path(lilypond_exe).resolve()
    bin_dir = lilypond_exe.parent

    lily_python = bin_dir / "python.exe"
    musicxml2ly = bin_dir / "musicxml2ly.py"

    ly_path = musicxml_path.with_suffix(".ly")

    subprocess.run(
        [
            str(lily_python),
            str(musicxml2ly),
            str(musicxml_path),
            "-o",
            str(ly_path),
        ],
        cwd=out_dir,
        check=True,
    )
    
    subprocess.run(
        [
            str(lilypond_exe),
            "--pdf",
            str(ly_path),
        ],
        cwd=out_dir,
        check=True,
    )

    return ly_path.with_suffix(".pdf")

def main():
    # 1. Add a melody (If notes are already adjusted for key, is_key_adjusted = True)
    melody = "C 4/4 Eq Dq Cq Dq - Eq Eq Eh - Dq Dq Dh - Eq Gq Gh - Eq Dq Cq Dq - Eq Eq Eq Eq - Dq Dq Eq Dq - Cw"

    # Parse melody string
    melody_obj = Melody.from_string(melody)

    # Add chord letters
    melody_obj.chords = {
        (1, 0): "C",
        (2, 1): "F"
    }

    # Convert to music21 object
    score = melody_obj.to_music21()

    # Write to output files
    output_directory = "output/"
    score.write("musicxml", output_directory + "sheet_music")
    
    # Use LilyPond to build clean pdf
    pdf_path = musicxml_to_lilypond_pdf(
        "output/sheet_music.musicxml",
        r"D:/LilyPond/lilypond-2.24.4/bin/lilypond.exe"
    )
    print("Generated:", pdf_path)

if __name__ == "__main__":
    main()
