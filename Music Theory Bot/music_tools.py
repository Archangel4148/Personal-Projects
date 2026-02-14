import dataclasses
import enum
import music21 as m21

# Required musical constants (these account for the tuning and note order)
_SEMITONE_MAP = {  # Positions of notes in an octave
    'C': 0, 'Cs': 1, 'Db': 1, 'D': 2, 'Ds': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'Fs': 6, 'Gb': 6, 'G': 7, 'Gs': 8,
    'Ab': 8, 'A': 9, 'As': 10, 'Bb': 10, 'B': 11
}
# Frequencies will be tuned to A4 ("equal temperament")
_TUNING_NOTE = "A"
_TUNING_OCTAVE = 4
_TUNING_FREQ = 440.0
_TUNING_SEMITONE_POS = _SEMITONE_MAP[_TUNING_NOTE]

# All notes not specified will be assumed to be in this octave
DEFAULT_PLAYING_OCTAVE = 4

BEAT_COUNT_MAP = {
    "t": 0.125,
    "s": 0.25,
    "e": 0.5,
    "q": 1,
    "h": 2,
    "w": 4,
}

KEY_SHARP_ORDER = "FCGDAEB"
KEY_FLAT_ORDER = KEY_SHARP_ORDER[::-1]
MAJOR_SCALE_NOTES = [0, 2, 4, 5, 7, 9, 11]

@dataclasses.dataclass
class Note:
    letter_name: str
    octave: int
    quarter_note_count: float

    @property
    def frequency(self) -> float:
        # Calculate the frequency from the tuning note (Hz)
        semitone_diff = _SEMITONE_MAP[self.letter_name] - _TUNING_SEMITONE_POS + (
                12 * abs(self.octave - _TUNING_OCTAVE))
        return _TUNING_FREQ * (2 ** (semitone_diff / 12))

    @classmethod
    def from_string(cls, note_str: str):
        """
        note_str should be formatted: "{note letter}{note octave (optional)}{beat count representation}"
        Example: C4h => Note C, octave 4, as a half note (2 beats)
        """
        beat_count = BEAT_COUNT_MAP[note_str[-1]]
        note_str = note_str[:-1]

        octave = DEFAULT_PLAYING_OCTAVE
        if (o := note_str[-1]).isdigit():
            octave = int(o)
            note_str = note_str[:-1]

        return cls(
            letter_name=note_str,
            octave=octave,
            quarter_note_count=beat_count,
        )
    
    def to_music21(self):
        """
        Convert this Note into a music21.note.Note
        """
        m21_note = m21.note.Note()
        m21_note.duration.quarterLength = self.quarter_note_count

        # Convert letter name (Cs, Bb, etc.) to music21 pitch spelling
        pitch_str = self.letter_name.replace("s", "#")
        m21_note.pitch.name = pitch_str
        m21_note.pitch.octave = self.octave

        return m21_note

class KeySignature(enum.IntEnum):
    # Positive value = sharp count, negative value = flat count
    C_MAJOR = 0
    G_MAJOR = 1
    D_MAJOR = 2
    A_MAJOR = 3
    E_MAJOR = 4
    B_MAJOR = 5
    Fs_MAJOR = 6
    F_MAJOR = -1
    Bb_MAJOR = -2
    Eb_MAJOR = -3
    Ab_MAJOR = -4
    Db_MAJOR = -5
    Gb_MAJOR = -6

    def get_scale(self):
        """Get the notes played in this key's scale (starting from the tonic)"""
        key_note = self.name.split("_")[0]
        all_notes = _SEMITONE_MAP.keys()
        # For flat keys, ignore sharp versions of notes, and vice versa
        if self.value < 0:
            all_notes = [note for note in all_notes if "s" not in note]
        else:
            all_notes = [note for note in all_notes if "b" not in note]
        key_idx = all_notes.index(key_note)
        # Order the notes, starting with the tonic
        ordered_notes = all_notes[key_idx:] + all_notes[:key_idx]
        # Build the scale using the major scale pattern
        scale = []
        for idx in MAJOR_SCALE_NOTES:
            # Handle the exceptions for enharmonics in Fs and Gb
            if self == KeySignature.Fs_MAJOR and idx == 11:
                scale.append("Es")
            elif self == KeySignature.Gb_MAJOR and idx == 4:
                scale.append("Cb")
            else:
                scale.append(ordered_notes[idx])
        return scale

    @classmethod
    def from_string(cls, key_str: str):
        for key_member in KeySignature:
            if key_member.name.split("_")[0] == key_str:
                return key_member
        return None
    
    def to_music21(self):
        """
        Convert KeySignature enum to music21.key.Key
        """
        tonic = self.name.split("_")[0].replace("s", "#")
        return m21.key.Key(tonic, "major")


@dataclasses.dataclass
class Melody:
    measures: list[list[Note]]
    time_signature: str
    key_signature: KeySignature
    chords: dict[tuple[int, float], str] | None = None


    @property
    def beats_per_measure(self):
        return int(self.time_signature.split("/")[0])

    @property
    def beats_per_quarter_note(self):
        return float(self.time_signature.split("/")[1]) / 4

    @classmethod
    def from_string(cls, melody_str: str):
        """
        Melody string should start with the key signature, followed by the time signature (Ex: "4/4"),
        and then a list of notes following the format:
        {note string} {note string} - {note string}
        (Notes are separated by spaces, measures are separated by hyphens)
        """
        measure_delimiter = "-"
        tokens = melody_str.split()

        # Read the key signature
        key_str = tokens[0]
        tokens = tokens[1:]

        # Read the time signature
        time_sig_str = tokens[0]
        tokens = tokens[1:]

        # Build the notes from the melody
        measures = [[] for _ in range(melody_str.count(measure_delimiter) + 1)]
        current_measure = 0
        for token in tokens:
            if token == measure_delimiter:
                current_measure += 1
                continue
            measures[current_measure].append(Note.from_string(token))

        return cls(
            measures=measures,
            time_signature=time_sig_str,
            key_signature=KeySignature.from_string(key_str)
        )

    def to_music21(self):
        """
        Convert Melody into a music21.stream.Score
        """
        score = m21.stream.Score()
        part = m21.stream.Part()

        for i, measure_notes in enumerate(self.measures, start=1):
            meas = m21.stream.Measure(number=i)

            # Put key + time signature ONLY in first measure
            if i == 1:
                meas.append(self.key_signature.to_music21())
                meas.append(m21.meter.TimeSignature(self.time_signature))

            for n in measure_notes:
                meas.append(n.to_music21())

            if self.chords:
                measure_length = m21.meter.TimeSignature(self.time_signature).barDuration.quarterLength

                for (m_idx, offset), chord_name in self.chords.items():
                    if m_idx == i:
                        cs = m21.harmony.ChordSymbol(chord_name)
                        cs.writeAsChord = False
                        print("Measure Length:", measure_length)
                        print(m_idx, offset)
                        print(f"max(0.25, {measure_length} - {offset})")

                        cs.duration.quarterLength = max(
                            0.25,
                            measure_length - offset
                        )
                        meas.insert(offset, cs)

            part.append(meas)

        score.append(part)
        return score