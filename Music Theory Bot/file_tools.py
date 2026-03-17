from pathlib import Path
import subprocess
from music21 import environment
environment.UserSettings()['lilypondPath'] = (
    r"D:\LilyPond\lilypond-2.24.4\bin\lilypond.exe"
)

def musicxml_to_pdf_musescore(musicxml_path: str, musescore_exe: str):
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



def musicxml_to_pdf_lily(musicxml_path: str, lilypond_exe: str, output_folder = None):
    """Use LilyPond to convert MusicXML to PDF"""
    musicxml_path = Path(musicxml_path).resolve()
    if output_folder is None:
        out_dir = musicxml_path.parent
    else:
        out_dir = Path(output_folder).resolve()

    lilypond_exe = Path(lilypond_exe).resolve()
    bin_dir = lilypond_exe.parent

    lily_python = bin_dir / "python.exe"
    musicxml2ly = bin_dir / "musicxml2ly.py"

    base_name = musicxml_path.stem
    ly_path = out_dir / f"{base_name}.ly"
    pdf_path = out_dir / f"{base_name}.pdf"
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

    # Remove unwanted marks
    ly_text = ly_text.replace(":5", "")
    ly_text = ly_text.replace('\\set Staff.instrumentName = "Voice"\n', "")
    ly_text = ly_text.replace('\\set Staff.shortInstrumentName = "Voice"\n', "")
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