import os
from pathlib import Path
import subprocess

from music21 import environment
# ALWAYS set explicitly (do NOT guard with "if not")
environment.UserSettings()['lilypondPath'] = (
    r"D:\LilyPond\lilypond-2.24.4\bin\lilypond.exe"
)

import music21 as m21

from music_tools import ScoreData, transpose_score

# def musicxml_to_pdf(musicxml_path: str, musescore_exe: str):
#     """Use MuseScore to convert MusicXML to PDF"""
#     out_dir = Path(musicxml_path).resolve().parent
#     out_pdf = Path(musicxml_path).with_suffix(".pdf")
#     subprocess.run(
#         [
#             musescore_exe,
#             musicxml_path,
#             "-o",
#             out_dir / out_pdf,
#         ],
#         cwd=out_dir,
#         check=True,
#     )



def musicxml_to_pdf(musicxml_path: str, lilypond_exe: str):
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
    ly_text = ly_path.read_text(encoding="utf-8")

    # Remove superscript chord qualifiers
    ly_text = ly_text.replace(":5", "")
    ly_path.write_text(ly_text, encoding="utf-8")

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

    melody_part.chords = {
        (1, 0.0): "E-",
        (2, 0.0): "A-",
        (3, 0.0): "Cm",
        (3, 2.0): "B-/D",
    }
    # Convert to music21 object
    score = score_obj.to_music21()

    source_key = score_obj.key_signature.to_music21()
    target_key = m21.key.Key("C", "major")
    score = transpose_score(score, source_key, target_key)


    # Write to output files
    output_directory = Path("output/").resolve()
    os.makedirs(output_directory, exist_ok=True)
    score.write("musicxml", output_directory / "sheet_music")

    # Convert MusicXML to PDF
    # musicxml_to_pdf(
    #     output_directory / "sheet_music.musicxml",
    #     r"D:\Musescore 4\bin\MuseScore4.exe"
    # )
    musicxml_to_pdf(
        output_directory / "sheet_music.musicxml",
        r"D:\LilyPond\lilypond-2.24.4\bin\lilypond.exe"
    )

if __name__ == "__main__":
    main()
